"""
app.py — FastAPI application (MQTT-powered v2).
"""

import asyncio
import threading
import os
import joblib
from collections import defaultdict
from datetime import timedelta

from fastapi import FastAPI, WebSocket, Depends, HTTPException, Request, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import tensorflow as tf
import paho.mqtt.publish as mqtt_publish

from src.auth import (
    Token, User,
    authenticate_user, create_access_token,
    get_current_user, require_admin, get_ws_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from src.database import init_db
from src.limiter import limiter
from src.sanitize import sanitize_mode, sanitize_machine_id, sanitize_string, sanitize_component_label
from src.metrics import (
    rul_gauge, rul_std_gauge, health_status_counter, inference_latency,
    ws_active_connections,
    simulation_mode_info,
    sensor_temperature, sensor_torque, sensor_tool_wear, sensor_speed,
    sensor_voltage, sensor_current, sensor_power_kw, sensor_vibration,
    machine_health_index, failure_probability, time_to_failure_hours,
    fault_component_counter, alarm_events,
    drift_score_gauge, drift_detected_flag, drift_rows_gauge,
    auto_retrain_runs, feedback_relabels_total, feedback_pending_gauge, feedback_ready_gauge,
)
from src.ws_manager import manager
from src.mqtt_subscriber import start_subscriber
from src.database import (
    fetch_latest_diagnosis,
    fetch_recent_diagnosis,
    fetch_fleet_overview,
    fetch_machine_ids,
    fetch_prediction_by_id,
    fetch_latest_prediction,
    insert_feedback_label,
    fetch_feedback_labels,
    resolve_feedback_label,
)
from src.fault_localization import FaultLocalizer
from src.retraining import AutoRetrainCoordinator

# ── Define BASE_DIR ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Paths ─────────────────────────────────────────────────────
DEFAULT_TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "models", "model_cmapss_int8.tflite")
DEFAULT_MULTIMODAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_multimodal_mtl.keras")
DEFAULT_MULTIMODAL_SAVEDMODEL_PATH = os.path.join(BASE_DIR, "models", "multimodal_savedmodel")
DEFAULT_MULTIMODAL_TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "models", "model_multimodal_portable.tflite")
RUNTIME_MODEL_PATH = os.getenv("RUNTIME_MODEL_PATH", "").strip()
RUNTIME_MODEL_MODE = os.getenv("RUNTIME_MODEL_MODE", "auto").strip().lower()
SCALER_PATH = os.getenv("RUNTIME_SCALER_PATH", os.path.join(BASE_DIR, "data", "scaler_cmapss.pkl"))

MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT   = int(os.getenv("MQTT_PORT", "1883"))

# ── Load model ────────────────────────────────────────────────
class _IdentityScaler:
    def transform(self, x):
        return x


def _compact_error(exc: Exception, limit: int = 280) -> str:
    msg = str(exc).replace("\n", " ").strip()
    if len(msg) > limit:
        msg = msg[: limit - 3].rstrip() + "..."
    return msg


def _load_scaler(path: str):
    try:
        loaded = joblib.load(path)
        print(f"[MODEL] Loaded scaler: {path}")
        return loaded
    except Exception as exc:
        print(f"[MODEL] Failed to load scaler '{path}' ({exc}); using identity scaler")
        return _IdentityScaler()


def _resolve_runtime_candidates() -> list[str]:
    candidates = []
    if RUNTIME_MODEL_PATH:
        candidates.append(RUNTIME_MODEL_PATH)
    else:
        if os.path.exists(DEFAULT_MULTIMODAL_MODEL_PATH):
            candidates.append(DEFAULT_MULTIMODAL_MODEL_PATH)
        if os.path.exists(os.path.join(DEFAULT_MULTIMODAL_SAVEDMODEL_PATH, "saved_model.pb")):
            candidates.append(DEFAULT_MULTIMODAL_SAVEDMODEL_PATH)
        if os.path.exists(DEFAULT_MULTIMODAL_TFLITE_MODEL_PATH):
            candidates.append(DEFAULT_MULTIMODAL_TFLITE_MODEL_PATH)
        candidates.append(DEFAULT_TFLITE_MODEL_PATH)

    deduped = []
    for path in candidates:
        if path not in deduped:
            deduped.append(path)
    return deduped


def _infer_mode(path: str) -> str:
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "saved_model.pb")):
        return "multimodal_savedmodel"

    ext = os.path.splitext(path)[1].lower()
    if ext in {".keras", ".h5"}:
        return "multimodal_keras"
    if ext == ".tflite" and "multimodal" in os.path.basename(path).lower():
        return "multimodal_tflite"
    return "tflite"


def _load_runtime_bundle():
    scaler_obj = _load_scaler(SCALER_PATH)
    candidates = _resolve_runtime_candidates()
    last_error = ""

    for path in candidates:
        mode = RUNTIME_MODEL_MODE if RUNTIME_MODEL_MODE != "auto" else _infer_mode(path)

        if not os.path.exists(path):
            last_error = f"missing model file: {path}"
            continue

        try:
            if mode == "multimodal_keras":
                keras_model = tf.keras.models.load_model(path, compile=False)
                print(f"[MODEL] Runtime mode=multimodal_keras, model={path}")
                return {
                    "mode": "multimodal_keras",
                    "model_path": path,
                    "keras_model": keras_model,
                    "scaler": scaler_obj,
                }

            if mode == "multimodal_savedmodel":
                saved_model = tf.saved_model.load(path)
                signature = saved_model.signatures.get("serving_default")
                if signature is None:
                    raise RuntimeError("SavedModel missing serving_default signature")
                print(f"[MODEL] Runtime mode=multimodal_savedmodel, model={path}")
                return {
                    "mode": "multimodal_savedmodel",
                    "model_path": path,
                    "saved_model": saved_model,
                    "saved_model_signature": signature,
                    "scaler": scaler_obj,
                }

            if mode in {"tflite", "multimodal_tflite"}:
                interpreter = tf.lite.Interpreter(model_path=path)
                interpreter.allocate_tensors()
                print(f"[MODEL] Runtime mode={mode}, model={path}")
                return {
                    "mode": mode,
                    "model_path": path,
                    "interpreter": interpreter,
                    "input_details": interpreter.get_input_details(),
                    "output_details": interpreter.get_output_details(),
                    "scaler": scaler_obj,
                }

            raise RuntimeError(f"unsupported runtime mode requested: {mode}")
        except Exception as exc:
            short_error = _compact_error(exc)
            if "Could not deserialize class 'Functional'" in short_error:
                short_error += " (Keras model/version mismatch likely)"
            last_error = short_error
            print(f"[MODEL] Failed loading {path} in mode={mode}: {short_error}")

    raise RuntimeError(
        "Unable to load a runtime model. "
        f"Checked: {candidates}. Last error: {last_error}"
    )


runtime_bundle = _load_runtime_bundle()
fault_localizer = FaultLocalizer()

# ── Metrics bundle ────────────────────────────────────────────
metrics = {
    "rul_gauge":             rul_gauge,
    "rul_std_gauge":         rul_std_gauge,
    "health_status_counter": health_status_counter,
    "inference_latency":     inference_latency,
    "sensor_temperature":    sensor_temperature,
    "sensor_torque":         sensor_torque,
    "sensor_tool_wear":      sensor_tool_wear,
    "sensor_speed":          sensor_speed,
    "sensor_voltage":        sensor_voltage,
    "sensor_current":        sensor_current,
    "sensor_power_kw":       sensor_power_kw,
    "sensor_vibration":      sensor_vibration,
    "machine_health_index":  machine_health_index,
    "failure_probability":   failure_probability,
    "time_to_failure_hours": time_to_failure_hours,
    "fault_component_counter": fault_component_counter,
    "alarm_events":          alarm_events,
    "drift_score_gauge":     drift_score_gauge,
    "drift_detected_flag":   drift_detected_flag,
    "drift_rows_gauge":      drift_rows_gauge,
    "auto_retrain_runs":     auto_retrain_runs,
    "feedback_relabels_total": feedback_relabels_total,
    "feedback_pending_gauge":  feedback_pending_gauge,
    "feedback_ready_gauge":    feedback_ready_gauge,
}

retrain_coordinator = AutoRetrainCoordinator(fault_localizer=fault_localizer, metrics=metrics)


class FeedbackRelabelRequest(BaseModel):
    machine_id: str = Field(min_length=1, max_length=32)
    corrected_component: str = Field(min_length=1, max_length=64)
    prediction_id: int | None = Field(default=None, ge=1)
    notes: str = Field(default="", max_length=400)
    resolved: bool = False


class ManualRetrainRequest(BaseModel):
    reason: str = Field(default="manual retraining", max_length=180)

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Edge AI Predictive Maintenance API",
    description="Real-time machine health monitoring via MQTT + TFLite.",
    version="2.0.0",
)

Instrumentator().instrument(app).expose(app)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

ws_connections: dict = defaultdict(int)
WS_MAX_PER_IP = 3

# ── Startup ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    loop = asyncio.get_running_loop()
    manager.set_loop(loop)

    thread = threading.Thread(
        target=start_subscriber,
        args=(runtime_bundle, manager, metrics, fault_localizer),
        daemon=True,
        name="mqtt-subscriber",
    )
    thread.start()
    retrain_coordinator.start()


@app.on_event("shutdown")
async def shutdown():
    retrain_coordinator.stop()

# ── Auth ──────────────────────────────────────────────────────
@app.post("/auth/login", response_model=Token, tags=["Auth"])
@limiter.limit("10/minute")
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username.strip().lower()
    password = form_data.password

    if len(username) > 64 or len(password) > 128:
        raise HTTPException(status_code=400, detail="Input exceeds maximum length")

    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": token, "token_type": "bearer"}

@app.get("/auth/me", response_model=User, tags=["Auth"])
@limiter.limit("60/minute")
def get_me(request: Request, current_user: User = Depends(get_current_user)):
    return current_user

# ── General ───────────────────────────────────────────────────
@app.get("/", tags=["General"])
@limiter.limit("30/minute")
def home(request: Request):
    return {"message": "Edge AI Predictive Maintenance API v2 🚀 (MQTT-powered)"}

@app.get("/health", tags=["General"])
def health():
    return {"status": "ok"}


@app.get("/diagnosis/latest/{machine_id}", tags=["Diagnosis"])
@limiter.limit("120/minute")
def diagnosis_latest(machine_id: str, request: Request):
    cleaned_id = sanitize_machine_id(machine_id)
    if not cleaned_id:
        raise HTTPException(status_code=400, detail="Invalid machine_id")

    row = fetch_latest_diagnosis(cleaned_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"No diagnosis data found for {cleaned_id}")
    return row


@app.get("/diagnosis/recent/{machine_id}", tags=["Diagnosis"])
@limiter.limit("60/minute")
def diagnosis_recent(machine_id: str, request: Request, limit: int = 100):
    cleaned_id = sanitize_machine_id(machine_id)
    if not cleaned_id:
        raise HTTPException(status_code=400, detail="Invalid machine_id")
    return fetch_recent_diagnosis(cleaned_id, limit=limit)


@app.get("/fleet/overview", tags=["Fleet"])
@limiter.limit("120/minute")
def fleet_overview(
    request: Request,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
):
    data = fetch_fleet_overview(limit=limit)
    return {
        "requested_by": current_user.username,
        "count": len(data),
        "machines": data,
    }


@app.get("/fleet/machines", tags=["Fleet"])
@limiter.limit("120/minute")
def fleet_machines(request: Request, current_user: User = Depends(get_current_user)):
    data = fetch_machine_ids()
    return {
        "requested_by": current_user.username,
        "count": len(data),
        "machine_ids": data,
    }


@app.post("/feedback/relabel", tags=["Feedback"])
@limiter.limit("90/minute")
def feedback_relabel(
    payload: FeedbackRelabelRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    machine_id = sanitize_machine_id(payload.machine_id)
    if not machine_id:
        raise HTTPException(status_code=400, detail="Invalid machine_id")

    corrected_component = sanitize_component_label(payload.corrected_component)
    if not corrected_component:
        raise HTTPException(status_code=400, detail="Invalid corrected_component")

    notes = sanitize_string(payload.notes, max_length=400) if payload.notes else ""
    prediction_id = int(payload.prediction_id) if payload.prediction_id else None
    prediction = None

    if prediction_id is not None:
        prediction = fetch_prediction_by_id(prediction_id)
        if not prediction:
            raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")
        if prediction.get("machine_id") and sanitize_machine_id(str(prediction.get("machine_id"))) != machine_id:
            raise HTTPException(
                status_code=400,
                detail="prediction_id does not belong to provided machine_id",
            )
    else:
        prediction = fetch_latest_prediction(machine_id)

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail=f"No diagnosis record found for machine {machine_id}",
        )

    if payload.resolved and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can mark relabels as resolved")

    created = insert_feedback_label(
        machine_id=machine_id,
        corrected_component=corrected_component,
        reviewer=current_user.username,
        notes=notes,
        prediction_id=int(prediction.get("id")) if prediction and prediction.get("id") else prediction_id,
        predicted_component=str(prediction.get("fault_component", "unknown")) if prediction else "unknown",
        metadata={
            "submitted_by_role": current_user.role,
            "source": "dashboard_or_api",
        },
        resolved=bool(payload.resolved and current_user.role == "admin"),
    )

    metrics["feedback_relabels_total"].labels(
        component=corrected_component,
        resolved="1" if created.get("resolved") else "0",
    ).inc()

    return {
        "message": "feedback relabel recorded",
        "feedback": created,
    }


@app.get("/feedback/relabels", tags=["Feedback"])
@limiter.limit("120/minute")
def feedback_list(
    request: Request,
    limit: int = 100,
    machine_id: str | None = None,
    resolved: bool | None = None,
    current_user: User = Depends(get_current_user),
):
    cleaned_machine_id = sanitize_machine_id(machine_id) if machine_id else None
    items = fetch_feedback_labels(limit=limit, machine_id=cleaned_machine_id, resolved=resolved)
    return {
        "requested_by": current_user.username,
        "count": len(items),
        "items": items,
    }


@app.post("/feedback/relabels/{feedback_id}/resolve", tags=["Feedback"])
@limiter.limit("60/minute")
def feedback_resolve(
    feedback_id: int,
    request: Request,
    current_user: User = Depends(require_admin),
):
    row = resolve_feedback_label(feedback_id=feedback_id, resolved_by=current_user.username)
    if not row:
        raise HTTPException(status_code=404, detail=f"Feedback {feedback_id} not found")

    metrics["feedback_relabels_total"].labels(
        component=sanitize_component_label(str(row.get("corrected_component", "unknown"))),
        resolved="1",
    ).inc()

    return {
        "message": "feedback resolved",
        "feedback": row,
    }


@app.get("/retraining/status", tags=["Retraining"])
@limiter.limit("120/minute")
def retraining_status(request: Request, current_user: User = Depends(get_current_user)):
    return {
        "requested_by": current_user.username,
        "status": retrain_coordinator.status(),
    }


@app.post("/retraining/run-now", tags=["Retraining"])
@limiter.limit("20/minute")
def retraining_run_now(
    payload: ManualRetrainRequest,
    request: Request,
    current_user: User = Depends(require_admin),
):
    status = retrain_coordinator.trigger_manual(
        requested_by=current_user.username,
        reason=sanitize_string(payload.reason, max_length=180),
    )
    return {
        "message": "manual retraining queued",
        "requested_by": current_user.username,
        "status": status,
    }


@app.get("/diagnosis/model/fault-localizer", tags=["Diagnosis"])
@limiter.limit("60/minute")
def fault_localizer_info(request: Request):
    return fault_localizer.info()


@app.post("/diagnosis/model/fault-localizer/reload", tags=["Diagnosis"])
@limiter.limit("20/minute")
def fault_localizer_reload(request: Request, current_user: User = Depends(require_admin)):
    fault_localizer.reload()
    return {
        "message": "fault localizer reloaded",
        "requested_by": current_user.username,
        "info": fault_localizer.info(),
    }

# ── Mode control ──────────────────────────────────────────────
@app.post("/set_mode/{new_mode}", tags=["Control"])
@limiter.limit("30/minute")
def set_mode(
    new_mode: str,
    request: Request,
    current_user: User = Depends(require_admin),
):
    try:
        cleaned = sanitize_mode(new_mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        mqtt_publish.single(
            topic="sensors/control/mode",
            payload=cleaned,
            hostname=MQTT_BROKER,
            port=MQTT_PORT,
            qos=1,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MQTT broker unavailable: {e}")

    simulation_mode_info.info({"mode": cleaned})
    print(f"[MODE] '{cleaned}' set by '{current_user.username}'")
    return {"mode": cleaned, "set_by": current_user.username}

# ── WebSocket ─────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    user = await get_ws_user(websocket)
    if not user:
        return

    client = websocket.client
    client_ip = client.host if client and client.host else "unknown"
    if ws_connections[client_ip] >= WS_MAX_PER_IP:
        await websocket.close(code=1008)
        return

    await manager.connect(websocket)
    ws_connections[client_ip] += 1
    ws_active_connections.inc()
    print(f"[WS] Connected — user: {user.username}")

    try:
        while True:
            # Keep a receive loop so disconnects are detected and cleaned up.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        manager.disconnect(websocket)
        ws_connections[client_ip] = max(0, ws_connections[client_ip] - 1)
        ws_active_connections.dec()
        print(f"[WS] Disconnected — user: {user.username}")