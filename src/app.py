from fastapi import FastAPI, WebSocket, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
import numpy as np
import tensorflow as tf
import os
import asyncio
import joblib

from src.auth import (
    Token, User,
    authenticate_user, create_access_token,
    get_current_user, require_admin, get_ws_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from src.database import init_db, insert_data
from datetime import timedelta

mode = "normal"
damage_level = 0

app = FastAPI(title="Edge AI Predictive Maintenance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "model_int8.tflite")
SCALER_PATH = os.path.join(BASE_DIR, "data",   "scaler.pkl")
DATA_PATH   = os.path.join(BASE_DIR, "data",   "X.npy")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = joblib.load(SCALER_PATH)

# ── Auth routes ───────────────────────────────────────────────

@app.post("/auth/login", response_model=Token, tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login with username + password.
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
def get_me(current_user: User = Depends(get_current_user)):
    """Returns info about the currently logged-in user."""
    return current_user

# ── Home ──────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def home():
    return {"message": "Edge AI Predictive Maintenance API 🚀"}

# ── Mode control (admin only) ─────────────────────────────────

@app.post("/set_mode/{new_mode}", tags=["Control"])
def set_mode(
    new_mode: str,
    current_user: User = Depends(require_admin),   # 🔐 admin only
):
    """
    Set simulation mode: normal | degrade | failure
    Requires: admin role
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

    print(f"[MODE] Set to '{mode}' by user '{current_user.username}'")
    return {"mode": mode, "set_by": current_user.username}

# ── WebSocket (authenticated via query param token) ───────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Real-time sensor stream.
    Connect with: ws://host/ws?token=<your_jwt>
    """
    global damage_level, mode

    # Authenticate before accepting the connection
    user = await get_ws_user(websocket)
    if not user:
        return   # get_ws_user already closed the socket

    await websocket.accept()
    print(f"[WS] Client connected — user: {user.username}")

    X = np.load(DATA_PATH)
    i = 200

    while True:
        try:
            sample = X[i:i+1].astype(np.float32)

            # ── Damage simulation ──
            if mode == "degrade":
                damage_level += 0.5
            elif mode == "failure":
                damage_level += 1
            damage_level = min(damage_level, 100)

            sample[:, :, 2] += damage_level * 0.5
            sample[:, :, 3] += damage_level * 0.3
            sample[:, :, 0] += damage_level * 0.2

            # ── Inference ──
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

            prediction -= damage_level * 0.5
            prediction = max(prediction, 0)

            # ── Status ──
            if prediction < 60:
                status = "CRITICAL"
            elif prediction < 120:
                status = "WARNING"
            else:
                status = "HEALTHY"

            # ── Sensor values ──
            try:
                sensor_values  = sample[0][-1]
                original_values = scaler.inverse_transform([sensor_values])[0]
                temp   = float(original_values[0])
                air    = float(original_values[1])
                torque = float(original_values[2])
                wear   = float(original_values[3])
                speed  = float(original_values[4])
            except Exception as scaler_err:
                print(f"[SCALER ERROR] {scaler_err}")
                temp, air, torque, wear, speed = 0, 0, 0, 0, 0

            data = {
                "machine_id": "M1",
                "step":        i,
                "RUL":         float(prediction),
                "status":      status,
                "temperature": temp,
                "air_temperature": air,
                "torque":      torque,
                "tool_wear":   wear,
                "speed":       speed,
            }

            insert_data(data)
            await websocket.send_json(data)

            print(f"[WS] MODE={mode} DAMAGE={damage_level:.1f} RUL={prediction:.2f} STATUS={status}")

            i += 1
            if i >= len(X):
                i = 0

            await asyncio.sleep(1)

        except Exception as e:
            print(f"[WS ERROR] {e}")
            break

    print(f"[WS] Client disconnected — user: {user.username}")