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
from src.sanitize import sanitize_mode
from src.metrics import (
    rul_gauge, rul_std_gauge, health_status_counter, inference_latency,
    ws_active_connections,
    simulation_mode_info,
    sensor_temperature, sensor_torque, sensor_tool_wear, sensor_speed,
)
from src.ws_manager import manager
from src.mqtt_subscriber import start_subscriber

# ── Define BASE_DIR ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Paths ─────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(BASE_DIR, "models", "model_cmapss_int8.tflite")
SCALER_PATH = os.path.join(BASE_DIR, "data", "scaler_cmapss.pkl")

MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT   = int(os.getenv("MQTT_PORT", "1883"))

# ── Load model ────────────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = joblib.load(SCALER_PATH)

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
}

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
        args=(interpreter, input_details, output_details, scaler, manager, metrics),
        daemon=True,
        name="mqtt-subscriber",
    )
    thread.start()

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