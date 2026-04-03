from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import os
import asyncio
import joblib
from src.database import init_db, insert_data

mode = "normal"
damage_level = 0

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model_int8.tflite")
SCALER_PATH = os.path.join(BASE_DIR, "data", "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "X.npy")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = joblib.load(SCALER_PATH)

@app.post("/set_mode/{new_mode}")
def set_mode(new_mode: str):
    global mode, damage_level
    mode = new_mode

    if mode == "normal":
        damage_level = 0
    elif mode == "failure":
        damage_level = 30

    print("MODE:", mode)
    return {"mode": mode}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global damage_level, mode

    await websocket.accept()

    X = np.load(DATA_PATH)
    i = 200

    while True:
        try:
            sample = X[i:i+1].astype(np.float32)

            # -------- DAMAGE --------
            if mode == "degrade":
                damage_level += 0.5
            elif mode == "failure":
                damage_level += 1

            damage_level = min(damage_level, 100)

            sample[:, :, 2] += damage_level * 0.5
            sample[:, :, 3] += damage_level * 0.3
            sample[:, :, 0] += damage_level * 0.2

            # -------- MODEL --------
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()

            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

            prediction -= damage_level * 0.5
            prediction = max(prediction, 0)

            # -------- STATUS --------
            if prediction < 60:
                status = "CRITICAL"
            elif prediction < 120:
                status = "WARNING"
            else:
                status = "HEALTHY"

            # -------- SENSOR VALUES --------
            try:
                sensor_values = sample[0][-1]
                original_values = scaler.inverse_transform([sensor_values])[0]

                temp    = float(original_values[0])
                air     = float(original_values[1])
                torque  = float(original_values[2])
                wear    = float(original_values[3])
                speed   = float(original_values[4])

            except Exception as scaler_err:
                print("Scaler error:", scaler_err)  # don't silently swallow
                temp, air, torque, wear, speed = 0, 0, 0, 0, 0

            data = {
                "machine_id": "M1",        # ✅ FIX: was missing, caused KeyError → WS crash
                "step": i,
                "RUL": float(prediction),
                "status": status,
                "temperature": temp,
                "air_temperature": air,
                "torque": torque,
                "tool_wear": wear,
                "speed": speed
            }

            insert_data(data)
            await websocket.send_json(data)

            print(f"MODE: {mode} | DAMAGE: {damage_level:.1f} | RUL: {prediction:.2f} | STATUS: {status}")

            i += 1
            if i >= len(X):
                i = 0

            await asyncio.sleep(1)

        except Exception as e:
            print("WebSocket ERROR:", e)
            break