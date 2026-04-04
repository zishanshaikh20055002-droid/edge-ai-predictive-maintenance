from fastapi import FastAPI, WebSocket, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
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

mode = "normal"
damage_level = 0

app = FastAPI(title="Edge AI Predictive Maintenance API")

# ── Register limiter + its error handler ──────────────────────
# Without this, hitting a limit raises an unhandled exception
# instead of returning a clean 429 response.
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

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "model_int8.tflite")
SCALER_PATH = os.path.join(BASE_DIR, "data",   "scaler.pkl")
DATA_PATH   = os.path.join(BASE_DIR, "data",   "X.npy")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = joblib.load(SCALER_PATH)

# ── WebSocket connection tracker (simple in-memory) ───────────
# Prevents a single IP from opening unlimited WebSocket connections.
# In production, use Redis for this so it works across multiple workers.
ws_connections: dict[str, int] = defaultdict(int)
WS_MAX_PER_IP = 3

# ── Auth routes ───────────────────────────────────────────────

@app.post("/auth/login", response_model=Token, tags=["Auth"])
@limiter.limit("10/minute")           # Brute-force protection
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login with username + password.
    Rate limit: 10 attempts per minute per IP.
    Returns a JWT access token valid for 60 minutes.
    """
    user = authenticate_user(form_data.username, form_data.password)
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
    """Returns info about the currently logged-in user."""
    return current_user


# ── General ───────────────────────────────────────────────────

@app.get("/", tags=["General"])
@limiter.limit("30/minute")
def home(request: Request):
    return {"message": "Edge AI Predictive Maintenance API 🚀"}


# ── Mode control ──────────────────────────────────────────────

@app.post("/set_mode/{new_mode}", tags=["Control"])
@limiter.limit("30/minute")           # Prevent rapid mode-spamming
def set_mode(
    new_mode: str,
    request: Request,
    current_user: User = Depends(require_admin),
):
    """
    Set simulation mode: normal | degrade | failure
    Rate limit: 30 per minute. Requires: admin role.
    """
    global mode, damage_level

    allowed = {"normal", "degrade", "failure"}
    if new_mode not in allowed:
        raise HTTPException(status_code=400, detail=f"Mode must be one of {allowed}")

    mode = new_mode
    if mode == "normal":
        damage_level = 0
    elif mode == "failure":
        damage_level = 30

    print(f"[MODE] '{mode}' set by '{current_user.username}'")
    return {"mode": mode, "set_by": current_user.username}


# ── WebSocket ─────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Real-time sensor stream.
    Connect with: ws://host/ws?token=<jwt>

    Per-IP connection limit: max 3 simultaneous WebSocket connections.
    """
    global damage_level, mode

    # ── Auth ──
    user = await get_ws_user(websocket)
    if not user:
        return

    # ── Per-IP connection limit ──
    client_ip = websocket.client.host
    if ws_connections[client_ip] >= WS_MAX_PER_IP:
        await websocket.close(code=1008)
        print(f"[WS] Rejected {client_ip} — too many connections ({WS_MAX_PER_IP} max)")
        return

    await websocket.accept()
    ws_connections[client_ip] += 1
    print(f"[WS] Connected — user: {user.username} | IP: {client_ip} | "
          f"total from IP: {ws_connections[client_ip]}")

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

            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

            prediction -= damage_level * 0.5
            prediction = max(prediction, 0)

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
                temp, air, torque, wear, speed = 0, 0, 0, 0, 0

            data = {
                "machine_id":      "M1",
                "step":            i,
                "RUL":             float(prediction),
                "status":          status,
                "temperature":     temp,
                "air_temperature": air,
                "torque":          torque,
                "tool_wear":       wear,
                "speed":           speed,
            }

            insert_data(data)
            await websocket.send_json(data)

            print(f"[WS] MODE={mode} DAMAGE={damage_level:.1f} "
                  f"RUL={prediction:.2f} STATUS={status}")

            i += 1
            if i >= len(X):
                i = 0

            await asyncio.sleep(1)

    except Exception as e:
        print(f"[WS ERROR] {e}")

    finally:
        # Always decrement counter, even on crash
        ws_connections[client_ip] = max(0, ws_connections[client_ip] - 1)
        print(f"[WS] Disconnected — user: {user.username} | IP: {client_ip}")