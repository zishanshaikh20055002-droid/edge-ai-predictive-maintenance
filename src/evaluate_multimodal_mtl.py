"""
evaluate_multimodal_mtl.py

Post-training sanity evaluation for the multimodal MTL model.

Outputs:
- models/multimodal_eval_report.json
- models/training_result_card.md
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_multimodal_mtl.keras")
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "data", "multisource_train.npz")
DEFAULT_TRAINING_REPORT_PATH = os.path.join(BASE_DIR, "models", "multisource_training_report.json")
DEFAULT_TRAINING_HISTORY_PATH = os.path.join(BASE_DIR, "models", "multisource_training_history.json")
DEFAULT_EVAL_REPORT_PATH = os.path.join(BASE_DIR, "models", "multimodal_eval_report.json")
DEFAULT_RESULT_CARD_PATH = os.path.join(BASE_DIR, "models", "training_result_card.md")

FAULT_CLASS_NAMES = ["twf", "hdf", "pwf", "osf", "rnf", "bearing_fault"]

THERMAL_EMBED_DIM = 128
VIBRATION_WINDOW = 256
ACOUSTIC_WINDOW = 2048
ELECTRICAL_WINDOW = 64
ELECTRICAL_FEATURES = 4
EVAL_MC_PASSES = 12
EVAL_NOISE_STD = 0.01
RNG = np.random.default_rng(123)


def _compact_error(exc: Exception, limit: int = 280) -> str:
    msg = str(exc).replace("\n", " ").strip()
    if len(msg) > limit:
        msg = msg[: limit - 3].rstrip() + "..."
    return msg


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
    vibration = _resample_sequence(process_window[:, vib_cols], VIBRATION_WINDOW).reshape(1, VIBRATION_WINDOW, 3)

    acoustic_src = np.mean(process_window, axis=1, keepdims=True)
    acoustic = _resample_sequence(acoustic_src, ACOUSTIC_WINDOW).reshape(1, ACOUSTIC_WINDOW, 1)

    elec_cols = [0, 1, 2, 4]
    elec_cols = [c if c < process_window.shape[1] else c % process_window.shape[1] for c in elec_cols]
    electrical = _resample_sequence(process_window[:, elec_cols], ELECTRICAL_WINDOW).reshape(
        1,
        ELECTRICAL_WINDOW,
        ELECTRICAL_FEATURES,
    )

    thermal = _thermal_embed_from_process(process_window).reshape(1, THERMAL_EMBED_DIM)
    return [process, vibration.astype(np.float32), acoustic.astype(np.float32), electrical.astype(np.float32), thermal.astype(np.float32)]


def _parse_multimodal_outputs(outputs):
    if isinstance(outputs, dict):
        rul = outputs.get("head_rul")
        faults = outputs.get("head_faults")
        anomaly = outputs.get("head_anomaly_score")
    else:
        rul, faults, anomaly = outputs

    return (
        float(np.asarray(rul).reshape(-1)[0]),
        np.asarray(faults).reshape(-1).astype(np.float32),
        float(np.asarray(anomaly).reshape(-1)[0]),
    )


def _stage_probs_from_rul_anomaly(rul: float, anomaly_score: float) -> list[float]:
    critical = float(np.clip(((90.0 - rul) / 90.0), 0.0, 1.0) * 0.70 + anomaly_score * 0.55)
    healthy = float(np.clip(((rul - 110.0) / 140.0), 0.0, 1.0) * (1.0 - anomaly_score))
    warning = float(max(0.0, 1.0 - critical - healthy))

    probs = np.asarray([healthy, warning, critical], dtype=np.float32)
    probs = np.clip(probs, 1e-6, None)
    probs = probs / np.sum(probs)
    return probs.tolist()


def _run_multimodal_prediction_local(keras_model, base_sample: np.ndarray) -> dict:
    rul_predictions = []
    anomaly_predictions = []

    for _ in range(EVAL_MC_PASSES):
        noisy = base_sample + RNG.normal(0.0, EVAL_NOISE_STD, base_sample.shape).astype(np.float32)
        outputs = keras_model(_build_multimodal_inputs(noisy), training=True)
        rul, _, anomaly = _parse_multimodal_outputs(outputs)
        rul_predictions.append(rul)
        anomaly_predictions.append(anomaly)

    rul_mean = float(np.mean(rul_predictions))
    rul_std = float(np.std(rul_predictions))
    anomaly_mean = float(np.mean(anomaly_predictions))
    stage_probs = _stage_probs_from_rul_anomaly(rul_mean, anomaly_mean)

    return {
        "rul_mean": rul_mean,
        "rul_std": rul_std,
        "stage_probs": stage_probs,
        "anomaly_score": anomaly_mean,
    }


def _read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    if y_true.size == 0:
        return None
    uniques = np.unique(y_true)
    if uniques.size < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def _macro_fault_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float | None, dict[str, float | None]]:
    per_class: dict[str, float | None] = {}
    valid = []

    for idx, class_name in enumerate(FAULT_CLASS_NAMES):
        auc = _safe_binary_auc(y_true[:, idx], y_score[:, idx])
        per_class[class_name] = auc
        if auc is not None:
            valid.append(auc)

    if not valid:
        return None, per_class
    return float(np.mean(valid)), per_class


def _subsample_indices(total: int, max_samples: int, seed: int) -> np.ndarray:
    if total <= max_samples:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=max_samples, replace=False).astype(np.int64))


def _best_epoch_and_loss(training_report: dict, history_report: dict) -> tuple[int | None, float | None]:
    tr = training_report.get("training", {}) if isinstance(training_report.get("training"), dict) else {}

    best_epoch = tr.get("best_epoch")
    best_val_loss = tr.get("best_val_loss")

    if best_epoch is None:
        best_epoch = history_report.get("best_epoch")
    if best_val_loss is None:
        best_val_loss = history_report.get("best_val_loss")

    try:
        best_epoch = int(best_epoch) if best_epoch is not None else None
    except Exception:
        best_epoch = None

    try:
        best_val_loss = float(best_val_loss) if best_val_loss is not None else None
    except Exception:
        best_val_loss = None

    return best_epoch, best_val_loss


def _auc_trend(history_report: dict) -> dict:
    metrics = history_report.get("metrics", {}) if isinstance(history_report.get("metrics"), dict) else {}
    for key in ["val_head_faults_auc", "val_head_anomaly_score_auc"]:
        series = metrics.get(key)
        if not isinstance(series, list) or not series:
            continue
        floats = [float(v) for v in series]
        return {
            "metric": key,
            "start": float(floats[0]),
            "best": float(max(floats)),
            "end": float(floats[-1]),
            "delta": float(floats[-1] - floats[0]),
        }
    return {}


def evaluate(
    model_path: str,
    dataset_path: str,
    training_report_path: str,
    training_history_path: str,
    eval_report_path: str,
    result_card_path: str,
    max_samples: int,
    runtime_samples: int,
    batch_size: int,
    seed: int,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = np.load(dataset_path)
    X_process = data["X_process"].astype(np.float32)
    X_vibration = data["X_vibration"].astype(np.float32)
    X_acoustic = data["X_acoustic"].astype(np.float32)
    X_electrical = data["X_electrical"].astype(np.float32)
    X_thermal = data["X_thermal"].astype(np.float32)
    y_rul = data["y_rul"].astype(np.float32)
    y_faults = data["y_faults"].astype(np.float32)
    y_anomaly = data["y_anomaly"].astype(np.float32)

    subset_idx = _subsample_indices(total=len(X_process), max_samples=max_samples, seed=seed)

    Xp = X_process[subset_idx]
    Xv = X_vibration[subset_idx]
    Xa = X_acoustic[subset_idx]
    Xe = X_electrical[subset_idx]
    Xt = X_thermal[subset_idx]
    yr = y_rul[subset_idx]
    yf = y_faults[subset_idx]
    ya = y_anomaly[subset_idx]

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        short_error = _compact_error(exc)
        raise RuntimeError(
            "Failed to load multimodal Keras model. "
            "If this model was trained in WSL/GPU with a different Keras stack, "
            "run this evaluator in that same environment. "
            f"Details: {short_error}"
        ) from None

    pred_rul, pred_faults, pred_anomaly = model.predict(
        [Xp, Xv, Xa, Xe, Xt],
        batch_size=batch_size,
        verbose=0,
    )

    pred_rul = np.asarray(pred_rul).reshape(-1)
    pred_faults = np.asarray(pred_faults)
    pred_anomaly = np.asarray(pred_anomaly).reshape(-1)

    rul_mae = float(mean_absolute_error(yr, pred_rul))
    rul_rmse = float(np.sqrt(mean_squared_error(yr, pred_rul)))
    anomaly_auc = _safe_binary_auc(ya, pred_anomaly)
    fault_auc_macro, fault_auc_per_class = _macro_fault_auc(yf, pred_faults)

    runtime_n = int(max(1, min(runtime_samples, len(Xp))))
    runtime_rul = []
    runtime_std = []
    runtime_stage = []
    runtime_anomaly = []

    for i in range(runtime_n):
        result = _run_multimodal_prediction_local(model, Xp[i : i + 1])
        runtime_rul.append(float(result["rul_mean"]))
        runtime_std.append(float(result["rul_std"]))
        runtime_stage.append(result["stage_probs"])
        runtime_anomaly.append(float(result["anomaly_score"]))

    runtime_rul = np.asarray(runtime_rul, dtype=np.float32)
    runtime_std = np.asarray(runtime_std, dtype=np.float32)
    runtime_anomaly = np.asarray(runtime_anomaly, dtype=np.float32)
    runtime_stage = np.asarray(runtime_stage, dtype=np.float32)

    corr = None
    if runtime_n >= 2:
        corr = float(np.corrcoef(pred_rul[:runtime_n], runtime_rul)[0, 1])

    training_report = _read_json(training_report_path)
    history_report = _read_json(training_history_path)
    best_epoch, best_val_loss = _best_epoch_and_loss(training_report, history_report)
    auc_trend = _auc_trend(history_report)

    eval_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": model_path,
        "dataset_path": dataset_path,
        "num_samples_evaluated": int(len(subset_idx)),
        "metrics": {
            "rul_mae": rul_mae,
            "rul_rmse": rul_rmse,
            "fault_auc_macro": float(fault_auc_macro) if fault_auc_macro is not None else None,
            "fault_auc_per_class": {
                k: (float(v) if v is not None else None)
                for k, v in fault_auc_per_class.items()
            },
            "anomaly_auc": float(anomaly_auc) if anomaly_auc is not None else None,
            "mean_predicted_anomaly_score": float(np.mean(pred_anomaly)),
        },
        "runtime_adapter_sanity": {
            "num_samples": runtime_n,
            "mean_rul": float(np.mean(runtime_rul)),
            "mean_rul_std": float(np.mean(runtime_std)),
            "mean_anomaly_score": float(np.mean(runtime_anomaly)),
            "stage_probs_mean": [float(x) for x in np.mean(runtime_stage, axis=0)],
            "rul_correlation_direct_vs_runtime": corr,
        },
        "training_summary": {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "auc_trend": auc_trend,
        },
        "deployment": {
            "runtime_model_mode": "multimodal_keras",
            "runtime_model_path": model_path,
            "fallback_model": os.path.join(BASE_DIR, "models", "model_cmapss_int8.tflite"),
            "env_notes": {
                "RUNTIME_MODEL_MODE": "multimodal_keras",
                "RUNTIME_MODEL_PATH": model_path,
                "MC_PASSES_MULTIMODAL": "12 (default)",
            },
        },
    }

    os.makedirs(os.path.dirname(eval_report_path), exist_ok=True)
    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    auc_text = "unavailable"
    if auc_trend:
        auc_text = (
            f"{auc_trend['metric']}: "
            f"{auc_trend['start']:.4f} -> {auc_trend['end']:.4f} "
            f"(best {auc_trend['best']:.4f}, delta {auc_trend['delta']:+.4f})"
        )

    best_epoch_text = str(best_epoch) if best_epoch is not None else "unavailable"
    best_val_loss_text = f"{best_val_loss:.4f}" if best_val_loss is not None else "unavailable"
    anomaly_auc_text = f"{anomaly_auc:.4f}" if anomaly_auc is not None else "unavailable"
    fault_auc_text = f"{fault_auc_macro:.4f}" if fault_auc_macro is not None else "unavailable"
    corr_text = f"{corr:.4f}" if corr is not None else "unavailable"

    card = f"""# Multimodal Training Result Card

- Generated: {eval_report['generated_at']}
- Model: {model_path}
- Dataset: {dataset_path}
- Samples evaluated: {len(subset_idx)}

## Core Results

- Best epoch: {best_epoch_text}
- Best val_loss: {best_val_loss_text}
- RUL MAE: {rul_mae:.4f}
- RUL RMSE: {rul_rmse:.4f}
- Fault AUC (macro): {fault_auc_text}
- Anomaly AUC: {anomaly_auc_text}
- AUC trend: {auc_text}

## Runtime Sanity (Process-only Adapter)

- Adapter sample count: {runtime_n}
- Mean runtime RUL: {float(np.mean(runtime_rul)):.4f}
- Mean runtime RUL std: {float(np.mean(runtime_std)):.4f}
- Mean runtime anomaly score: {float(np.mean(runtime_anomaly)):.4f}
- Direct-vs-runtime RUL correlation: {corr_text}

## Deployment-ready Notes

- Runtime model mode: multimodal_keras
- Runtime model path: {model_path}
- Fallback model: {os.path.join(BASE_DIR, 'models', 'model_cmapss_int8.tflite')}
- Set env vars:
  - RUNTIME_MODEL_MODE=multimodal_keras
  - RUNTIME_MODEL_PATH={model_path}
  - MC_PASSES_MULTIMODAL=12

## Artifacts

- Evaluation report: {eval_report_path}
- Training card: {result_card_path}
"""

    with open(result_card_path, "w", encoding="utf-8") as f:
        f.write(card)

    print(f"Saved evaluation report: {eval_report_path}")
    print(f"Saved result card: {result_card_path}")
    print(f"RUL MAE={rul_mae:.4f} | RUL RMSE={rul_rmse:.4f} | Fault AUC={fault_auc_text} | Anomaly AUC={anomaly_auc_text}")

    return eval_report


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multimodal MTL model and emit a compact result card")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--training-report", default=DEFAULT_TRAINING_REPORT_PATH)
    parser.add_argument("--training-history", default=DEFAULT_TRAINING_HISTORY_PATH)
    parser.add_argument("--out", default=DEFAULT_EVAL_REPORT_PATH)
    parser.add_argument("--card", default=DEFAULT_RESULT_CARD_PATH)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--runtime-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        model_path=args.model,
        dataset_path=args.dataset,
        training_report_path=args.training_report,
        training_history_path=args.training_history,
        eval_report_path=args.out,
        result_card_path=args.card,
        max_samples=args.max_samples,
        runtime_samples=args.runtime_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )
