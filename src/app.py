"""
app.py — FastAPI application with:
  - JWT authentication
  - Rate limiting
  - Input validation + sanitization
  - Prometheus metrics exposed at /metrics
"""

import time
from fastapi import FastAPI, WebSocket, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import numpy as np
import tensorflow as tf
import os
import asyncio
import joblib
from collections import defaultdict
from datetime import timedelta

from src.auth import (
    Token, User,
    authenticate_user, create_access_token,
    get_current_user, require_admin, get_ws_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from src.database import init_db, insert_data
from src.limiter import limiter
from src.schemas import SensorReading, PredictionResponse
from src.sanitize import sanitize_mode, sanitize_sensor_dict
from src.metrics import (
    rul_gauge, health_status_counter, inference_latency,
    ws_active_connections, ws_messages_sent,
    simulation_mode_info, damage_level_gauge,
    sensor_temperature, sensor_torque, sensor_tool_wear, sensor_speed,
)

mode = "normal"
damage_level = 0

app = FastAPI(
    title="Edge AI Predictive Maintenance API",
    description="Real-time machine health monitoring with RUL prediction.",
    version="1.0.0",
)

# ── Prometheus: auto-instrument all HTTP endpoints ────────────
# Exposes /metrics with: request count, latency histograms,
# response sizes, status code breakdowns — all for free.
Instrumentator().instrument(app).expose(app)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "model_int8.tflite")
SCALER_PATH = os.path.join(BASE_DIR, "data",   "scaler.pkl")
DATA_PATH   = os.path.join(BASE_DIR, "data",   "X.npy")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = joblib.load(SCALER_PATH)

ws_connections: dict[str, int] = defaultdict(int)
WS_MAX_PER_IP = 3


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
    return {"message": "Edge AI Predictive Maintenance API 🚀"}


@app.get("/health", tags=["General"])
def health():
    """Health check endpoint — used by Docker and load balancers."""
    return {"status": "ok"}


# ── Predict ───────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
@limiter.limit("60/minute")
def predict(
    request: Request,
    reading: SensorReading,
    current_user: User = Depends(get_current_user),
):
    machine_id = reading.machine_id

    features = np.array([[
        reading.air_temperature,
        reading.temperature,
        reading.rotational_speed,
        reading.torque,
        reading.tool_wear,
    ]], dtype=np.float32)

    features_scaled = scaler.transform(features)
    sample = np.tile(features_scaled, (1, 30, 1)).reshape(1, 30, 5).astype(np.float32)

    # ── Time the inference and record it ──
    t_start = time.time()
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    prediction = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
    inference_latency.labels(machine_id=machine_id).observe(time.time() - t_start)

    prediction = max(0.0, prediction)

    if prediction < 60:
        status = "CRITICAL"
    elif prediction < 120:
        status = "WARNING"
    else:
        status = "HEALTHY"

    # ── Update metrics ──
    rul_gauge.labels(machine_id=machine_id).set(prediction)
    health_status_counter.labels(machine_id=machine_id, status=status).inc()
    sensor_temperature.labels(machine_id=machine_id).set(reading.temperature)
    sensor_torque.labels(machine_id=machine_id).set(reading.torque)
    sensor_tool_wear.labels(machine_id=machine_id).set(reading.tool_wear)
    sensor_speed.labels(machine_id=machine_id).set(reading.rotational_speed)

    record = sanitize_sensor_dict({
        "machine_id":      machine_id,
        "step":            0,
        "RUL":             prediction,
        "status":          status,
        "temperature":     reading.temperature,
        "air_temperature": reading.air_temperature,
        "torque":          reading.torque,
        "tool_wear":       reading.tool_wear,
        "speed":           reading.rotational_speed,
    })
    insert_data(record)

    return {**record, "RUL": prediction, "status": status}


# ── Mode control ──────────────────────────────────────────────

@app.post("/set_mode/{new_mode}", tags=["Control"])
@limiter.limit("30/minute")
def set_mode(
    new_mode: str,
    request: Request,
    current_user: User = Depends(require_admin),
):
    global mode, damage_level

    try:
        mode = sanitize_mode(new_mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if mode == "normal":
        damage_level = 0
    elif mode == "failure":
        damage_level = 30

    # ── Update mode metric ──
    simulation_mode_info.info({"mode": mode})
    damage_level_gauge.set(damage_level)

    print(f"[MODE] '{mode}' set by '{current_user.username}'")
    return {"mode": mode, "set_by": current_user.username}


# ── WebSocket ─────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global damage_level, mode

    user = await get_ws_user(websocket)
    if not user:
        return

    client_ip = websocket.client.host
    if ws_connections[client_ip] >= WS_MAX_PER_IP:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    ws_connections[client_ip] += 1
    ws_active_connections.inc()   # ← metric

    machine_id = "M1"
    X = np.load(DATA_PATH)
    i = 200

    try:
        while True:
            sample = X[i:i+1].astype(np.float32)

            if mode == "degrade":
                damage_level += 0.5
            elif mode == "failure":
                damage_level += 1
            damage_level = min(damage_level, 100)

            sample[:, :, 2] += damage_level * 0.5
            sample[:, :, 3] += damage_level * 0.3
            sample[:, :, 0] += damage_level * 0.2

            # ── Timed inference ──
            t_start = time.time()
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            prediction = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
            inference_latency.labels(machine_id=machine_id).observe(time.time() - t_start)

            prediction = max(0.0, prediction - damage_level * 0.5)

            if prediction < 60:
                status = "CRITICAL"
            elif prediction < 120:
                status = "WARNING"
            else:
                status = "HEALTHY"

            try:
                sensor_values   = sample[0][-1]
                original_values = scaler.inverse_transform([sensor_values])[0]
                temp   = float(original_values[0])
                air    = float(original_values[1])
                torque = float(original_values[2])
                wear   = float(original_values[3])
                speed  = float(original_values[4])
            except Exception as e:
                print(f"[SCALER ERROR] {e}")
                temp, air, torque, wear, speed = 300.0, 298.0, 40.0, 0.0, 1500.0

            # ── Update all metrics ──
            rul_gauge.labels(machine_id=machine_id).set(prediction)
            health_status_counter.labels(machine_id=machine_id, status=status).inc()
            damage_level_gauge.set(damage_level)
            simulation_mode_info.info({"mode": mode})
            sensor_temperature.labels(machine_id=machine_id).set(temp)
            sensor_torque.labels(machine_id=machine_id).set(torque)
            sensor_tool_wear.labels(machine_id=machine_id).set(wear)
            sensor_speed.labels(machine_id=machine_id).set(speed)
            ws_messages_sent.labels(machine_id=machine_id).inc()

            record = sanitize_sensor_dict({
                "machine_id":      machine_id,
                "step":            i,
                "RUL":             prediction,
                "status":          status,
                "temperature":     temp,
                "air_temperature": air,
                "torque":          torque,
                "tool_wear":       wear,
                "speed":           speed,
            })

            insert_data(record)
            await websocket.send_json({**record, "RUL": prediction})

            i += 1
            if i >= len(X):
                i = 0

            await asyncio.sleep(1)

    except Exception as e:
        print(f"[WS ERROR] {e}")

    finally:
        ws_connections[client_ip] = max(0, ws_connections[client_ip] - 1)
        ws_active_connections.dec()   # ← metric
        print(f"[WS] Disconnected — user: {user.username}")