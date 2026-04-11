import json
import logging
import os
import time
import warnings

import numpy as np
import paho.mqtt.client as mqtt

from src.ingestion import HardwareAgnosticBuffer

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT   = int(os.getenv("MQTT_PORT", "1883"))
WINDOW_SIZE = 30
NUM_FEATURES = 14  # CMAPSS uses 14
N_PASSES    = max(1, _env_int("MC_PASSES", 30))
NOISE_STD   = max(0.0, _env_float("MC_NOISE_STD", 0.01))
RNG         = np.random.default_rng()

# CMAPSS raw sensor ranges (FD001) for first 5 selected features.
RAW_SENSOR_BOUNDS = {
    "temperature": (641.21, 644.53),
    "air_temperature": (1571.04, 1616.91),
    "torque": (1382.25, 1441.49),
    "tool_wear": (549.85, 556.06),
    "speed": (2387.90, 2388.56),
}

# Dashboard target ranges.
UI_SENSOR_BOUNDS = {
    "temperature": (280.0, 360.0),
    "air_temperature": (270.0, 350.0),
    "torque": (10.0, 95.0),
    "tool_wear": (0.0, 300.0),
    "speed": (1000.0, 3000.0),
}


def _map_range(value: float, src: tuple[float, float], dst: tuple[float, float]) -> float:
    src_lo, src_hi = src
    dst_lo, dst_hi = dst
    if src_hi <= src_lo:
        return dst_lo
    norm = (value - src_lo) / (src_hi - src_lo)
    norm = float(np.clip(norm, 0.0, 1.0))
    return dst_lo + norm * (dst_hi - dst_lo)


def _to_ui_sensors(raw_features: np.ndarray) -> dict:
    return {
        "temperature": _map_range(float(raw_features[0]), RAW_SENSOR_BOUNDS["temperature"], UI_SENSOR_BOUNDS["temperature"]),
        "air_temperature": _map_range(float(raw_features[1]), RAW_SENSOR_BOUNDS["air_temperature"], UI_SENSOR_BOUNDS["air_temperature"]),
        "torque": _map_range(float(raw_features[2]), RAW_SENSOR_BOUNDS["torque"], UI_SENSOR_BOUNDS["torque"]),
        "tool_wear": _map_range(float(raw_features[3]), RAW_SENSOR_BOUNDS["tool_wear"], UI_SENSOR_BOUNDS["tool_wear"]),
        "speed": _map_range(float(raw_features[4]), RAW_SENSOR_BOUNDS["speed"], UI_SENSOR_BOUNDS["speed"]),
    }

ingestion_buffer = HardwareAgnosticBuffer(window_size=WINDOW_SIZE, num_features=NUM_FEATURES)


def _build_output_mapping(output_details):
    """Infer interpreter output indices by tensor shape/name."""
    mapping = {
        "rul_index": None,
        "rul_std_index": None,
        "stage_index": None,
        "direct_uncertainty": False,
    }
    scalar_candidates = []

    for detail in output_details:
        idx = detail["index"]
        name = str(detail.get("name", "")).lower()
        shape = detail.get("shape", [])
        size = int(np.prod(shape)) if shape is not None else 0

        if size == 3 and mapping["stage_index"] is None:
            mapping["stage_index"] = idx
            continue

        if size == 1:
            if "std" in name and mapping["rul_std_index"] is None:
                mapping["rul_std_index"] = idx
            elif "rul" in name and mapping["rul_index"] is None:
                mapping["rul_index"] = idx
            else:
                scalar_candidates.append(idx)

    if mapping["rul_index"] is None and scalar_candidates:
        mapping["rul_index"] = scalar_candidates.pop(0)
    if mapping["rul_std_index"] is None and scalar_candidates:
        mapping["rul_std_index"] = scalar_candidates.pop(0)

    # Fallbacks for simple 2-output models (rul + stage)
    if mapping["rul_index"] is None and output_details:
        mapping["rul_index"] = output_details[0]["index"]
    if mapping["stage_index"] is None and len(output_details) > 1:
        mapping["stage_index"] = output_details[1]["index"]

    mapping["direct_uncertainty"] = (
        mapping["rul_index"] is not None
        and mapping["rul_std_index"] is not None
        and mapping["stage_index"] is not None
        and len(output_details) >= 3
    )

    return mapping

def start_subscriber(interpreter, input_details, output_details, scaler, manager, metrics):
    from src.database import insert_data
    from src.sanitize import sanitize_sensor_dict

    output_map = _build_output_mapping(output_details)
    if output_map["direct_uncertainty"]:
        logger.info("[MQTT] Using direct uncertainty outputs from TFLite model")
    else:
        logger.info("[MQTT] Using Monte Carlo sampling over standard TFLite outputs")

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe("sensors/+/data")
            logger.info("[MQTT] Connected and subscribed to CMAPSS sensor stream")

    def on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
        except Exception:
            return
            
        machine_id = data.get("machine_id", "M1")
        try:
            step = int(data.get("step", 0))
        except (TypeError, ValueError):
            step = 0
        raw_features = data.get("features", [])
        
        ingestion_buffer.process_payload(machine_id, step, raw_features)
        raw_window = ingestion_buffer.get_valid_window(machine_id)
        
        if raw_window is None:
            return

        try:
            scaled_window = scaler.transform(raw_window[0])
        except Exception:
            return
        base_sample = scaled_window.reshape(1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
        clean_features = raw_window[0][-1]
        
        t0 = time.perf_counter()

        if output_map["direct_uncertainty"]:
            interpreter.set_tensor(input_details[0]["index"], base_sample)
            interpreter.invoke()

            rul_mean = float(
                interpreter.get_tensor(output_map["rul_index"]).reshape(-1)[0]
            )
            rul_std = float(abs(
                interpreter.get_tensor(output_map["rul_std_index"]).reshape(-1)[0]
            ))
            stage_probs = interpreter.get_tensor(output_map["stage_index"]).reshape(-1).tolist()
        else:
            rul_predictions = []
            stage_predictions = []

            for _ in range(N_PASSES):
                noise = RNG.normal(0.0, NOISE_STD, base_sample.shape).astype(np.float32)
                interpreter.set_tensor(input_details[0]["index"], base_sample + noise)
                interpreter.invoke()

                rul_predictions.append(
                    float(interpreter.get_tensor(output_map["rul_index"]).reshape(-1)[0])
                )
                if output_map["stage_index"] is not None:
                    stage_predictions.append(
                        interpreter.get_tensor(output_map["stage_index"]).reshape(-1)
                    )

            rul_mean = float(np.mean(rul_predictions))
            rul_std = float(np.std(rul_predictions))
            stage_probs = np.mean(stage_predictions, axis=0).tolist() if stage_predictions else []

        latency = time.perf_counter() - t0
        prediction = max(0.0, float(rul_mean))
        rul_std = float(rul_std)
        
        if len(stage_probs) == 3:
            pred_stage = np.argmax(stage_probs)
            status = "HEALTHY" if pred_stage == 0 else "WARNING" if pred_stage == 1 else "CRITICAL"
        else:
            status = "CRITICAL" if prediction < 60 else "WARNING" if prediction < 120 else "HEALTHY"
            stage_probs = [1.0, 0.0, 0.0] if status == "HEALTHY" else [0.0, 1.0, 0.0] if status == "WARNING" else [0.0, 0.0, 1.0]

        result = sanitize_sensor_dict({
            "machine_id": machine_id,
            "step": step,
            "RUL": prediction,
            "RUL_std": rul_std,
            "status": status,
            "stage_probs": [round(p, 3) for p in stage_probs],
            # Map first 5 CMAPSS features to UI so telemetry bars animate
            **_to_ui_sensors(clean_features),
        })

        try:
            metrics["rul_gauge"].labels(machine_id=machine_id).set(prediction)
            if "rul_std_gauge" in metrics:
                metrics["rul_std_gauge"].labels(machine_id=machine_id).set(rul_std)
            metrics["health_status_counter"].labels(machine_id=machine_id, status=status).inc()
            metrics["inference_latency"].labels(machine_id=machine_id).observe(latency)
            if "sensor_temperature" in metrics:
                metrics["sensor_temperature"].labels(machine_id=machine_id).set(result["temperature"])
            if "sensor_torque" in metrics:
                metrics["sensor_torque"].labels(machine_id=machine_id).set(result["torque"])
            if "sensor_tool_wear" in metrics:
                metrics["sensor_tool_wear"].labels(machine_id=machine_id).set(result["tool_wear"])
            if "sensor_speed" in metrics:
                metrics["sensor_speed"].labels(machine_id=machine_id).set(result["speed"])
        except Exception:
            pass
        
        try:
            insert_data(result)
        except Exception:
            pass

        manager.broadcast_from_thread(result)
        logger.info(f"[MQTT] {machine_id} Step={step} RUL={prediction:.1f}±{rul_std:.1f} {status} {latency*1000:.2f}ms")

    client = mqtt.Client(client_id="fastapi-subscriber")
    client.on_connect = on_connect
    client.on_message = on_message

    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            client.loop_forever()
        except Exception as e:
            logger.error(f"[MQTT] Error: {e} — retrying in 5s")
            time.sleep(5)