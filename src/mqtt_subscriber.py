import json
import logging
import os
import time
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
import paho.mqtt.client as mqtt

from src.ingestion import HardwareAgnosticBuffer, AsyncSensorFusionBuffer
from src.diagnostics import build_realtime_diagnosis, FAULT_PLAYBOOK
from src.alarm_policy import evaluate_alarm

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
MC_PASSES_MULTIMODAL = max(1, _env_int("MC_PASSES_MULTIMODAL", 12))
NOISE_STD   = max(0.0, _env_float("MC_NOISE_STD", 0.01))
ASYNC_RESAMPLE_HZ = max(0.1, _env_float("ASYNC_RESAMPLE_HZ", 1.0))
ASYNC_MAX_BUFFER_SECONDS = max(10.0, _env_float("ASYNC_MAX_BUFFER_SECONDS", 120.0))
RNG         = np.random.default_rng()

THERMAL_EMBED_DIM = 128
VIBRATION_WINDOW = 256
ACOUSTIC_WINDOW = 2048
ELECTRICAL_WINDOW = 64
ELECTRICAL_FEATURES = 4

FAULT_CLASS_TO_COMPONENT = {
    0: "stator",        # twf
    1: "cooling",       # hdf
    2: "power_supply",  # pwf
    3: "rotor",         # osf
    4: "lubrication",   # rnf
    5: "bearing",       # bearing_fault
}

CMAPSS_FEATURE_NAMES = [
    "sensor_measurement_2",
    "sensor_measurement_3",
    "sensor_measurement_4",
    "sensor_measurement_7",
    "sensor_measurement_8",
    "sensor_measurement_9",
    "sensor_measurement_11",
    "sensor_measurement_12",
    "sensor_measurement_13",
    "sensor_measurement_14",
    "sensor_measurement_15",
    "sensor_measurement_17",
    "sensor_measurement_20",
    "sensor_measurement_21",
]

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


def _resample_sequence(sequence: np.ndarray, target_len: int) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim == 1:
        sequence = sequence.reshape(-1, 1)
    if sequence.shape[0] == target_len:
        return sequence.astype(np.float32)
    if sequence.shape[0] <= 1:
        return np.repeat(sequence[:1], target_len, axis=0).astype(np.float32)

    src_x = np.linspace(0.0, 1.0, num=sequence.shape[0], dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    out = np.zeros((target_len, sequence.shape[1]), dtype=np.float32)

    for channel in range(sequence.shape[1]):
        out[:, channel] = np.interp(dst_x, src_x, sequence[:, channel]).astype(np.float32)

    return out.astype(np.float32)


def _thermal_embed_from_process(process_window_data: np.ndarray, dim: int = THERMAL_EMBED_DIM) -> np.ndarray:
    mean = np.mean(process_window_data, axis=0)
    std = np.std(process_window_data, axis=0)
    vmin = np.min(process_window_data, axis=0)
    vmax = np.max(process_window_data, axis=0)
    stats = np.concatenate([mean, std, vmin, vmax], axis=0).astype(np.float32)

    if stats.shape[0] >= dim:
        return stats[:dim].astype(np.float32)

    repeats = int(np.ceil(dim / max(1, stats.shape[0])))
    return np.tile(stats, repeats)[:dim].astype(np.float32)


def _build_multimodal_inputs(process_batch: np.ndarray):
    process = process_batch.astype(np.float32)
    process_window = process[0]

    vib_cols = [5, 9, 10]
    vib_cols = [c if c < process_window.shape[1] else c % process_window.shape[1] for c in vib_cols]
    vibration_src = process_window[:, vib_cols]
    vibration = _resample_sequence(vibration_src, VIBRATION_WINDOW).reshape(1, VIBRATION_WINDOW, 3)

    acoustic_src = np.mean(process_window, axis=1, keepdims=True)
    acoustic = _resample_sequence(acoustic_src, ACOUSTIC_WINDOW).reshape(1, ACOUSTIC_WINDOW, 1)

    elec_cols = [0, 1, 2, 4]
    elec_cols = [c if c < process_window.shape[1] else c % process_window.shape[1] for c in elec_cols]
    electrical_src = process_window[:, elec_cols]
    electrical = _resample_sequence(electrical_src, ELECTRICAL_WINDOW).reshape(
        1,
        ELECTRICAL_WINDOW,
        ELECTRICAL_FEATURES,
    )

    thermal = _thermal_embed_from_process(process_window).reshape(1, THERMAL_EMBED_DIM)

    return [
        process.astype(np.float32),
        vibration.astype(np.float32),
        acoustic.astype(np.float32),
        electrical.astype(np.float32),
        thermal.astype(np.float32),
    ]


def _parse_multimodal_outputs(outputs):
    if isinstance(outputs, dict):
        rul = outputs.get("head_rul")
        faults = outputs.get("head_faults")
        anomaly = outputs.get("head_anomaly_score")
    else:
        rul, faults, anomaly = outputs

    rul_value = float(np.asarray(rul).reshape(-1)[0])
    fault_vector = np.asarray(faults).reshape(-1).astype(np.float32)
    anomaly_value = float(np.asarray(anomaly).reshape(-1)[0])
    return rul_value, fault_vector, anomaly_value


def _stage_probs_from_rul_anomaly(rul: float, anomaly_score: float) -> list[float]:
    critical = float(np.clip(((90.0 - rul) / 90.0), 0.0, 1.0) * 0.70 + anomaly_score * 0.55)
    healthy = float(np.clip(((rul - 110.0) / 140.0), 0.0, 1.0) * (1.0 - anomaly_score))
    warning = float(max(0.0, 1.0 - critical - healthy))

    probs = np.asarray([healthy, warning, critical], dtype=np.float32)
    probs = np.clip(probs, 1e-6, None)
    probs = probs / np.sum(probs)
    return probs.tolist()


def _fault_component_probs_from_head(fault_probs: np.ndarray) -> dict[str, float]:
    vector = np.asarray(fault_probs, dtype=np.float32).reshape(-1)
    component_probs: dict[str, float] = defaultdict(float)

    for idx, value in enumerate(vector):
        component = FAULT_CLASS_TO_COMPONENT.get(idx)
        if component is None:
            continue
        component_probs[component] = max(component_probs[component], float(np.clip(value, 0.0, 1.0)))

    if not component_probs:
        return {}

    total = sum(component_probs.values())
    if total <= 0:
        return {}

    return {
        key: float(val / total)
        for key, val in component_probs.items()
    }


def _run_multimodal_prediction(keras_model, base_sample: np.ndarray) -> dict:
    rul_predictions = []
    fault_predictions = []
    anomaly_predictions = []

    for _ in range(MC_PASSES_MULTIMODAL):
        noisy = base_sample
        if NOISE_STD > 0.0:
            noisy = base_sample + RNG.normal(0.0, NOISE_STD, base_sample.shape).astype(np.float32)

        model_inputs = _build_multimodal_inputs(noisy.astype(np.float32))
        outputs = keras_model(model_inputs, training=True)
        rul, faults, anomaly = _parse_multimodal_outputs(outputs)

        rul_predictions.append(rul)
        fault_predictions.append(faults)
        anomaly_predictions.append(anomaly)

    rul_mean = float(np.mean(rul_predictions))
    rul_std = float(np.std(rul_predictions))
    fault_mean = np.mean(np.asarray(fault_predictions, dtype=np.float32), axis=0)
    anomaly_mean = float(np.mean(anomaly_predictions))
    stage_probs = _stage_probs_from_rul_anomaly(rul_mean, anomaly_mean)

    return {
        "rul_mean": rul_mean,
        "rul_std": rul_std,
        "stage_probs": stage_probs,
        "anomaly_score": anomaly_mean,
        "fault_component_probabilities": _fault_component_probs_from_head(fault_mean),
    }

ingestion_buffer = HardwareAgnosticBuffer(window_size=WINDOW_SIZE, num_features=NUM_FEATURES)
async_fusion_buffer = AsyncSensorFusionBuffer(
    feature_names=CMAPSS_FEATURE_NAMES,
    window_size=WINDOW_SIZE,
    target_hz=ASYNC_RESAMPLE_HZ,
    max_buffer_seconds=ASYNC_MAX_BUFFER_SECONDS,
)


def _parse_feature_payload(raw_payload: bytes):
    try:
        decoded = raw_payload.decode().strip()
    except Exception:
        return None

    if not decoded:
        return None

    try:
        data = json.loads(decoded)
        if isinstance(data, dict):
            value = data.get("value")
            timestamp = data.get("timestamp")
            if timestamp is None:
                timestamp = data.get("ts")
            if isinstance(timestamp, str):
                # Supports ISO timestamps from edge devices.
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
                except Exception:
                    timestamp = None
            return {"value": value, "timestamp": timestamp}
    except Exception:
        pass

    try:
        return {"value": float(decoded), "timestamp": None}
    except (TypeError, ValueError):
        return None


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


def _read_tflite_output_tensor(interpreter, tensor_detail: dict) -> np.ndarray:
    """Read an output tensor and dequantize integer outputs when needed."""
    tensor = interpreter.get_tensor(tensor_detail["index"])
    arr = np.asarray(tensor)

    if arr.dtype in (np.int8, np.uint8, np.int16, np.int32):
        qparams = tensor_detail.get("quantization_parameters") or {}
        scales = np.asarray(qparams.get("scales", []), dtype=np.float32).reshape(-1)
        zero_points = np.asarray(qparams.get("zero_points", []), dtype=np.float32).reshape(-1)

        if scales.size > 0 and float(scales[0]) > 0.0:
            scale = float(scales[0])
            zero_point = float(zero_points[0]) if zero_points.size > 0 else 0.0
            return (arr.astype(np.float32) - zero_point) * scale

    return arr.astype(np.float32)


def _build_multimodal_tflite_output_mapping(output_details, expected_fault_classes: int = 6):
    """Infer output tensor indices for multimodal TFLite models."""
    mapping = {
        "rul_index": None,
        "faults_index": None,
        "anomaly_index": None,
    }

    scalar_candidates = []
    vector_candidates = []

    for detail in output_details:
        idx = detail["index"]
        name = str(detail.get("name", "")).lower()
        shape = detail.get("shape", [])
        size = int(np.prod(shape)) if shape is not None else 0

        if size == 1:
            if "rul" in name and mapping["rul_index"] is None:
                mapping["rul_index"] = idx
            elif "anomaly" in name and mapping["anomaly_index"] is None:
                mapping["anomaly_index"] = idx
            else:
                scalar_candidates.append(idx)
            continue

        if size > 1:
            if "fault" in name and mapping["faults_index"] is None:
                mapping["faults_index"] = idx
            else:
                vector_candidates.append((idx, size))

    if mapping["faults_index"] is None and vector_candidates:
        for idx, size in vector_candidates:
            if size >= expected_fault_classes:
                mapping["faults_index"] = idx
                break
    if mapping["faults_index"] is None and vector_candidates:
        mapping["faults_index"] = max(vector_candidates, key=lambda x: x[1])[0]

    if mapping["rul_index"] is None and scalar_candidates:
        mapping["rul_index"] = scalar_candidates.pop(0)
    if mapping["anomaly_index"] is None and scalar_candidates:
        mapping["anomaly_index"] = scalar_candidates.pop(0)

    return mapping


def _build_multimodal_tflite_input_mapping(input_details):
    """Infer input tensor indices for multimodal TFLite models."""
    mapping = {
        "process": None,
        "vibration": None,
        "acoustic": None,
        "electrical": None,
        "thermal": None,
    }

    shape_to_key = {
        (WINDOW_SIZE, NUM_FEATURES): "process",
        (VIBRATION_WINDOW, 3): "vibration",
        (ACOUSTIC_WINDOW, 1): "acoustic",
        (ELECTRICAL_WINDOW, ELECTRICAL_FEATURES): "electrical",
        (THERMAL_EMBED_DIM,): "thermal",
    }

    for detail in input_details:
        idx = detail["index"]
        name = str(detail.get("name", "")).lower()
        shape = tuple(int(x) for x in detail.get("shape", []) if int(x) > 0)
        tail = shape[1:] if len(shape) > 1 else shape

        assigned = None
        if "process" in name:
            assigned = "process"
        elif "vibration" in name:
            assigned = "vibration"
        elif "acoustic" in name:
            assigned = "acoustic"
        elif "electrical" in name:
            assigned = "electrical"
        elif "thermal" in name:
            assigned = "thermal"
        elif tail in shape_to_key:
            assigned = shape_to_key[tail]

        if assigned and mapping[assigned] is None:
            mapping[assigned] = idx

    missing = [key for key, value in mapping.items() if value is None]
    if missing:
        raise RuntimeError(
            f"Could not infer multimodal TFLite input tensors for: {missing}. "
            f"Available tensor names: {[str(d.get('name', '')) for d in input_details]}"
        )

    return mapping


def _run_multimodal_tflite_prediction(
    interpreter,
    input_map,
    output_map,
    output_details_by_index,
    base_sample: np.ndarray,
) -> dict:
    rul_predictions = []
    fault_predictions = []
    anomaly_predictions = []

    for _ in range(MC_PASSES_MULTIMODAL):
        noisy = base_sample
        if NOISE_STD > 0.0:
            noisy = base_sample + RNG.normal(0.0, NOISE_STD, base_sample.shape).astype(np.float32)

        model_inputs = _build_multimodal_inputs(noisy.astype(np.float32))
        interpreter.set_tensor(input_map["process"], model_inputs[0])
        interpreter.set_tensor(input_map["vibration"], model_inputs[1])
        interpreter.set_tensor(input_map["acoustic"], model_inputs[2])
        interpreter.set_tensor(input_map["electrical"], model_inputs[3])
        interpreter.set_tensor(input_map["thermal"], model_inputs[4])
        interpreter.invoke()

        rul_detail = output_details_by_index[output_map["rul_index"]]
        anomaly_detail = output_details_by_index[output_map["anomaly_index"]]
        rul = float(_read_tflite_output_tensor(interpreter, rul_detail).reshape(-1)[0])
        anomaly = float(_read_tflite_output_tensor(interpreter, anomaly_detail).reshape(-1)[0])

        faults = None
        if output_map.get("faults_index") is not None:
            faults_detail = output_details_by_index[output_map["faults_index"]]
            faults = _read_tflite_output_tensor(interpreter, faults_detail).reshape(-1).astype(np.float32)

        rul_predictions.append(rul)
        anomaly_predictions.append(anomaly)
        if faults is not None:
            fault_predictions.append(faults)

    rul_mean = float(np.mean(rul_predictions))
    rul_std = float(np.std(rul_predictions))
    anomaly_mean = float(np.mean(anomaly_predictions))
    stage_probs = _stage_probs_from_rul_anomaly(rul_mean, anomaly_mean)

    fault_component_probabilities = {}
    if fault_predictions:
        fault_mean = np.mean(np.asarray(fault_predictions, dtype=np.float32), axis=0)
        fault_component_probabilities = _fault_component_probs_from_head(fault_mean)

    return {
        "rul_mean": rul_mean,
        "rul_std": rul_std,
        "stage_probs": stage_probs,
        "anomaly_score": anomaly_mean,
        "fault_component_probabilities": fault_component_probabilities,
    }


def _invoke_multimodal_savedmodel(signature, model_inputs):
    positional_specs, keyword_specs = signature.structured_input_signature

    if keyword_specs:
        kwargs = {}
        remaining = set(keyword_specs.keys())

        by_name = {
            "process": model_inputs[0],
            "vibration": model_inputs[1],
            "acoustic": model_inputs[2],
            "electrical": model_inputs[3],
            "thermal": model_inputs[4],
        }

        for key in list(remaining):
            lowered = str(key).lower()
            if "process" in lowered:
                kwargs[key] = by_name["process"]
                remaining.remove(key)
            elif "vibration" in lowered:
                kwargs[key] = by_name["vibration"]
                remaining.remove(key)
            elif "acoustic" in lowered:
                kwargs[key] = by_name["acoustic"]
                remaining.remove(key)
            elif "electrical" in lowered:
                kwargs[key] = by_name["electrical"]
                remaining.remove(key)
            elif "thermal" in lowered:
                kwargs[key] = by_name["thermal"]
                remaining.remove(key)

        shape_to_input = {
            (WINDOW_SIZE, NUM_FEATURES): model_inputs[0],
            (VIBRATION_WINDOW, 3): model_inputs[1],
            (ACOUSTIC_WINDOW, 1): model_inputs[2],
            (ELECTRICAL_WINDOW, ELECTRICAL_FEATURES): model_inputs[3],
            (THERMAL_EMBED_DIM,): model_inputs[4],
        }

        for key in list(remaining):
            spec = keyword_specs[key]
            shape = tuple(
                int(x)
                for x in getattr(spec, "shape", [])
                if x is not None and int(x) > 0
            )
            tail = shape[1:] if len(shape) > 1 else shape
            if tail in shape_to_input:
                kwargs[key] = shape_to_input[tail]
                remaining.remove(key)

        if remaining:
            raise RuntimeError(f"Unable to map SavedModel inputs for keys: {sorted(remaining)}")

        return signature(**kwargs)

    if positional_specs:
        if len(positional_specs) == 1 and isinstance(positional_specs[0], (list, tuple)):
            return signature(model_inputs)
        if len(positional_specs) == 5:
            return signature(*model_inputs)

    raise RuntimeError("Unsupported SavedModel serving signature layout")


def _parse_savedmodel_outputs(outputs):
    if isinstance(outputs, dict):
        rul = None
        faults = None
        anomaly = None
        scalar_candidates = []
        vector_candidates = []

        for name, value in outputs.items():
            arr = np.asarray(value)
            size = int(arr.size)
            lowered = str(name).lower()

            if size == 1:
                if "rul" in lowered and rul is None:
                    rul = arr
                elif "anomaly" in lowered and anomaly is None:
                    anomaly = arr
                else:
                    scalar_candidates.append(arr)
            elif size > 1:
                if "fault" in lowered and faults is None:
                    faults = arr
                else:
                    vector_candidates.append(arr)

        if faults is None and vector_candidates:
            faults = max(vector_candidates, key=lambda x: x.size)
        if rul is None and scalar_candidates:
            rul = scalar_candidates.pop(0)
        if anomaly is None and scalar_candidates:
            anomaly = scalar_candidates.pop(0)

        if rul is None or anomaly is None:
            raise RuntimeError("SavedModel outputs must include scalar RUL and anomaly tensors")

        rul_value = float(np.asarray(rul).reshape(-1)[0])
        fault_vector = np.asarray(faults).reshape(-1).astype(np.float32) if faults is not None else np.zeros((0,), dtype=np.float32)
        anomaly_value = float(np.asarray(anomaly).reshape(-1)[0])
        return rul_value, fault_vector, anomaly_value

    if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
        rul = float(np.asarray(outputs[0]).reshape(-1)[0])
        faults = np.asarray(outputs[1]).reshape(-1).astype(np.float32)
        anomaly = float(np.asarray(outputs[2]).reshape(-1)[0])
        return rul, faults, anomaly

    raise RuntimeError("Unsupported SavedModel output structure")


def _run_multimodal_savedmodel_prediction(saved_model_signature, base_sample: np.ndarray) -> dict:
    rul_predictions = []
    fault_predictions = []
    anomaly_predictions = []

    for _ in range(MC_PASSES_MULTIMODAL):
        noisy = base_sample
        if NOISE_STD > 0.0:
            noisy = base_sample + RNG.normal(0.0, NOISE_STD, base_sample.shape).astype(np.float32)

        model_inputs = _build_multimodal_inputs(noisy.astype(np.float32))
        outputs = _invoke_multimodal_savedmodel(saved_model_signature, model_inputs)
        rul, faults, anomaly = _parse_savedmodel_outputs(outputs)

        rul_predictions.append(rul)
        anomaly_predictions.append(anomaly)
        if faults.size:
            fault_predictions.append(faults)

    rul_mean = float(np.mean(rul_predictions))
    rul_std = float(np.std(rul_predictions))
    anomaly_mean = float(np.mean(anomaly_predictions))
    stage_probs = _stage_probs_from_rul_anomaly(rul_mean, anomaly_mean)

    fault_component_probabilities = {}
    if fault_predictions:
        fault_mean = np.mean(np.asarray(fault_predictions, dtype=np.float32), axis=0)
        fault_component_probabilities = _fault_component_probs_from_head(fault_mean)

    return {
        "rul_mean": rul_mean,
        "rul_std": rul_std,
        "stage_probs": stage_probs,
        "anomaly_score": anomaly_mean,
        "fault_component_probabilities": fault_component_probabilities,
    }

def start_subscriber(runtime_bundle, manager, metrics, fault_localizer=None):
    from src.database import insert_data
    from src.sanitize import sanitize_sensor_dict

    runtime_mode = str(runtime_bundle.get("mode", "tflite")).strip().lower()
    model_path = str(runtime_bundle.get("model_path", ""))
    scaler = runtime_bundle.get("scaler")

    if scaler is None:
        raise RuntimeError("runtime_bundle missing scaler")

    interpreter = runtime_bundle.get("interpreter")
    input_details = runtime_bundle.get("input_details", [])
    output_details = runtime_bundle.get("output_details", [])
    keras_model = runtime_bundle.get("keras_model")
    saved_model_signature = runtime_bundle.get("saved_model_signature")

    output_map = None
    multimodal_input_map = None
    output_details_by_index = {d["index"]: d for d in output_details}
    if runtime_mode == "tflite":
        if interpreter is None or not input_details or not output_details:
            raise RuntimeError("runtime_bundle missing TFLite interpreter details")
        output_map = _build_output_mapping(output_details)
    elif runtime_mode == "multimodal_keras":
        if keras_model is None:
            raise RuntimeError("runtime_bundle missing Keras model")
    elif runtime_mode == "multimodal_savedmodel":
        if saved_model_signature is None:
            raise RuntimeError("runtime_bundle missing SavedModel serving signature")
    elif runtime_mode == "multimodal_tflite":
        if interpreter is None or not input_details or not output_details:
            raise RuntimeError("runtime_bundle missing multimodal TFLite interpreter details")
        multimodal_input_map = _build_multimodal_tflite_input_mapping(input_details)
        output_map = _build_multimodal_tflite_output_mapping(output_details)
        if output_map["rul_index"] is None or output_map["anomaly_index"] is None:
            raise RuntimeError(
                "multimodal_tflite outputs must include RUL and anomaly score tensors"
            )
    else:
        raise RuntimeError(f"Unsupported runtime mode: {runtime_mode}")

    last_inferred_step = defaultdict(int)

    if runtime_mode == "tflite":
        if output_map["direct_uncertainty"]:
            logger.info(f"[MQTT] Runtime=tflite direct-uncertainty model={model_path}")
        else:
            logger.info(f"[MQTT] Runtime=tflite MC-sampling model={model_path}")
    elif runtime_mode == "multimodal_tflite":
        logger.info(
            f"[MQTT] Runtime=multimodal_tflite MC-passes={MC_PASSES_MULTIMODAL} "
            f"model={model_path}"
        )
    elif runtime_mode == "multimodal_savedmodel":
        logger.info(
            f"[MQTT] Runtime=multimodal_savedmodel MC-passes={MC_PASSES_MULTIMODAL} "
            f"model={model_path}"
        )
    else:
        logger.info(
            f"[MQTT] Runtime=multimodal_keras MC-passes={MC_PASSES_MULTIMODAL} "
            f"model={model_path}"
        )

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe("sensors/+/data")
            client.subscribe("sensors/+/feature/+")
            logger.info(
                "[MQTT] Connected and subscribed to array + async feature streams"
            )

    def on_message(client, userdata, msg):
        topic_parts = msg.topic.split("/")
        raw_window = None
        machine_id = "M1"
        step = 0

        is_feature_stream = (
            len(topic_parts) >= 4
            and topic_parts[0] == "sensors"
            and topic_parts[2] == "feature"
        )

        if is_feature_stream:
            machine_id = topic_parts[1]
            feature_name = topic_parts[3]
            parsed = _parse_feature_payload(msg.payload)
            if not parsed:
                return

            accepted = async_fusion_buffer.process_feature(
                machine_id=machine_id,
                feature_name=feature_name,
                value=parsed.get("value"),
                timestamp=parsed.get("timestamp"),
            )
            if not accepted:
                return

            current_step = async_fusion_buffer.get_latest_step(machine_id)
            if current_step <= last_inferred_step[machine_id]:
                return

            last_inferred_step[machine_id] = current_step
            step = current_step
            raw_window = async_fusion_buffer.get_valid_window(machine_id)
        else:
            try:
                data = json.loads(msg.payload.decode())
            except Exception:
                return

            machine_id = data.get("machine_id", topic_parts[1] if len(topic_parts) > 1 else "M1")
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
        fault_component_probabilities = {}

        if runtime_mode == "multimodal_keras":
            inference = _run_multimodal_prediction(keras_model, base_sample)
            rul_mean = float(inference["rul_mean"])
            rul_std = float(inference["rul_std"])
            stage_probs = list(inference["stage_probs"])
            fault_component_probabilities = dict(inference["fault_component_probabilities"])
        elif runtime_mode == "multimodal_savedmodel":
            inference = _run_multimodal_savedmodel_prediction(saved_model_signature, base_sample)
            rul_mean = float(inference["rul_mean"])
            rul_std = float(inference["rul_std"])
            stage_probs = list(inference["stage_probs"])
            fault_component_probabilities = dict(inference["fault_component_probabilities"])
        elif runtime_mode == "multimodal_tflite":
            inference = _run_multimodal_tflite_prediction(
                interpreter,
                multimodal_input_map,
                output_map,
                output_details_by_index,
                base_sample,
            )
            rul_mean = float(inference["rul_mean"])
            rul_std = float(inference["rul_std"])
            stage_probs = list(inference["stage_probs"])
            fault_component_probabilities = dict(inference["fault_component_probabilities"])
        elif output_map["direct_uncertainty"]:
            interpreter.set_tensor(input_details[0]["index"], base_sample)
            interpreter.invoke()

            rul_detail = output_details_by_index[output_map["rul_index"]]
            rul_std_detail = output_details_by_index[output_map["rul_std_index"]]
            stage_detail = output_details_by_index[output_map["stage_index"]]

            rul_mean = float(_read_tflite_output_tensor(interpreter, rul_detail).reshape(-1)[0])
            rul_std = float(abs(_read_tflite_output_tensor(interpreter, rul_std_detail).reshape(-1)[0]))
            stage_probs = _read_tflite_output_tensor(interpreter, stage_detail).reshape(-1).tolist()
        else:
            rul_predictions = []
            stage_predictions = []

            for _ in range(N_PASSES):
                noise = RNG.normal(0.0, NOISE_STD, base_sample.shape).astype(np.float32)
                interpreter.set_tensor(input_details[0]["index"], base_sample + noise)
                interpreter.invoke()

                rul_detail = output_details_by_index[output_map["rul_index"]]
                rul_predictions.append(
                    float(_read_tflite_output_tensor(interpreter, rul_detail).reshape(-1)[0])
                )
                if output_map["stage_index"] is not None:
                    stage_detail = output_details_by_index[output_map["stage_index"]]
                    stage_predictions.append(
                        _read_tflite_output_tensor(interpreter, stage_detail).reshape(-1)
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

        ui_sensors = _to_ui_sensors(clean_features)
        diagnosis = build_realtime_diagnosis(
            machine_id=machine_id,
            step=step,
            prediction=prediction,
            rul_std=rul_std,
            status=status,
            stage_probs=stage_probs,
            ui_sensors=ui_sensors,
            raw_features=clean_features,
        )

        if fault_component_probabilities:
            diagnosis["fault_component_probabilities"] = {
                k: round(float(v), 4)
                for k, v in fault_component_probabilities.items()
            }

            top_component = max(
                fault_component_probabilities,
                key=fault_component_probabilities.get,
            )
            top_confidence = float(fault_component_probabilities[top_component])
            if top_confidence >= float(diagnosis.get("fault_confidence", 0.0)):
                diagnosis["fault_component"] = top_component
                diagnosis["fault_confidence"] = round(top_confidence, 4)
                if top_component in FAULT_PLAYBOOK:
                    diagnosis["fault_type"] = FAULT_PLAYBOOK[top_component]["fault_type"]
                    diagnosis["probable_causes"] = FAULT_PLAYBOOK[top_component]["probable_causes"]
                    diagnosis["recommended_actions"] = FAULT_PLAYBOOK[top_component]["recommended_actions"]

            diagnosis.setdefault("fault_model_source", runtime_mode)
            diagnosis.setdefault(
                "fault_model_version",
                os.path.basename(model_path) if model_path else runtime_mode,
            )

        ml_fault = fault_localizer.predict({
            "machine_id": machine_id,
            "RUL": prediction,
            "RUL_std": rul_std,
            "status": status,
            "stage_probs": stage_probs,
            **ui_sensors,
            **diagnosis,
        }) if fault_localizer else None

        if ml_fault:
            diagnosis.update(ml_fault)
            component = diagnosis.get("fault_component", "")
            if component in FAULT_PLAYBOOK:
                diagnosis["fault_type"] = FAULT_PLAYBOOK[component]["fault_type"]
                diagnosis["probable_causes"] = FAULT_PLAYBOOK[component]["probable_causes"]
                diagnosis["recommended_actions"] = FAULT_PLAYBOOK[component]["recommended_actions"]
            else:
                diagnosis["fault_type"] = "ml_predicted_component_fault"
                diagnosis["probable_causes"] = [
                    "Fault pattern recognized by supervised classifier",
                    "Component not mapped in current playbook",
                ]
                diagnosis["recommended_actions"] = [
                    "Review model explanation and raw telemetry",
                    "Add component playbook entry and retrain if needed",
                ]
            diagnosis["diagnosis_version"] = "v2.0-hybrid-ml"
        else:
            diagnosis.setdefault("fault_model_source", "rules")
            diagnosis.setdefault("fault_model_version", "rules-only")
            diagnosis["diagnosis_version"] = "v1.0-rule-fusion"

        diagnosis.update(evaluate_alarm({
            "failure_probability": diagnosis.get("failure_probability", 0.0),
            "time_to_failure_hours": diagnosis.get("time_to_failure_hours", 0.0),
            "fault_severity": diagnosis.get("fault_severity", "LOW"),
            "fault_confidence": diagnosis.get("fault_confidence", 0.0),
            "fault_component": diagnosis.get("fault_component", "unknown"),
        }))

        result = sanitize_sensor_dict({
            "machine_id": machine_id,
            "step": step,
            "RUL": prediction,
            "RUL_std": rul_std,
            "status": status,
            "stage_probs": [round(p, 3) for p in stage_probs],
            # Map first 5 CMAPSS features to UI so telemetry bars animate
            **ui_sensors,
            **diagnosis,
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
            if "sensor_voltage" in metrics:
                metrics["sensor_voltage"].labels(machine_id=machine_id).set(result["voltage"])
            if "sensor_current" in metrics:
                metrics["sensor_current"].labels(machine_id=machine_id).set(result["current"])
            if "sensor_power_kw" in metrics:
                metrics["sensor_power_kw"].labels(machine_id=machine_id).set(result["power_kw"])
            if "sensor_vibration" in metrics:
                metrics["sensor_vibration"].labels(machine_id=machine_id).set(result["vibration"])
            if "machine_health_index" in metrics:
                metrics["machine_health_index"].labels(machine_id=machine_id).set(result["health_index"])
            if "failure_probability" in metrics:
                metrics["failure_probability"].labels(machine_id=machine_id).set(result["failure_probability"])
            if "time_to_failure_hours" in metrics:
                metrics["time_to_failure_hours"].labels(machine_id=machine_id).set(result["time_to_failure_hours"])
            if "fault_component_counter" in metrics:
                metrics["fault_component_counter"].labels(
                    machine_id=machine_id,
                    component=result["fault_component"],
                    severity=result["fault_severity"],
                ).inc()
            if "alarm_events" in metrics:
                metrics["alarm_events"].labels(
                    machine_id=machine_id,
                    level=result.get("alarm_level", "INFO"),
                    priority=result.get("maintenance_priority", "P4"),
                ).inc()
        except Exception:
            pass
        
        try:
            insert_data(result)
        except Exception:
            pass

        manager.broadcast_from_thread(result)
        source = "feature" if is_feature_stream else "array"
        logger.info(
            f"[MQTT] {machine_id} Step={step} src={source} "
            f"RUL={prediction:.1f}±{rul_std:.1f} {status} {latency*1000:.2f}ms"
        )

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