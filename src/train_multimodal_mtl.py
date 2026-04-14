"""
train_multimodal_mtl.py

Production-oriented multimodal multi-task training pipeline.

Improvements:
- Multi-dataset fusion (ai4i2020 + CWRU + MIMII + MetroPT-3 + Edge-IIoTset)
- Class-balanced focal losses for fault/anomaly heads
- Minority-aware sample weighting
- Weighted bootstrap resampling for long-tail robustness
- Real-sensor-ready multimodal shape contract preserved for runtime expansion
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.imbalance import (
    binary_sample_weights,
    multilabel_sample_weights,
    weighted_bootstrap_indices,
)
from src.model_multimodal_mtl import (
    build_multimodal_mtl_model,
    compile_multimodal_mtl_model,
)
from src.multisource_dataset import (
    MultiSourceConfig,
    build_multisource_dataset,
    save_multisource_dataset,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = os.path.join(BASE_DIR, "data", "multisource_train.npz")
DEFAULT_EXTERNAL_ROOT = os.path.join(BASE_DIR, "data", "external")
GPU_TARGET_BATCH_SIZE = 128
GPU_STEPS_PER_EXECUTION = 32


def _select_training_device(prefer_gpu: bool = True, require_gpu: bool = False) -> str:
    """
    Prefer NVIDIA/RTX GPU when available; otherwise use CPU.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if prefer_gpu and gpus:
        selected = None
        for gpu in gpus:
            name = str(getattr(gpu, "name", "")).upper()
            if "NVIDIA" in name or "RTX" in name:
                selected = gpu
                break
        if selected is None:
            selected = gpus[0]

        try:
            tf.config.set_visible_devices(selected, "GPU")
        except RuntimeError:
            # If context was initialized, continue with default visibility.
            pass

        try:
            tf.config.experimental.set_memory_growth(selected, True)
        except Exception:
            pass

        print(f"Using GPU device: {selected}")
        return "gpu"

    if require_gpu:
        raise RuntimeError(
            "No TensorFlow GPU device detected, but --require-gpu was specified."
        )

    print("No TensorFlow GPU detected; using CPU")
    return "cpu"


def _load_dataset(npz_path: str):
    data = np.load(npz_path)
    return {
        "X_process": data["X_process"].astype(np.float32),
        "X_vibration": data["X_vibration"].astype(np.float32),
        "X_acoustic": data["X_acoustic"].astype(np.float32),
        "X_electrical": data["X_electrical"].astype(np.float32),
        "X_thermal": data["X_thermal"].astype(np.float32),
        "y_rul": data["y_rul"].astype(np.float32),
        "y_faults": data["y_faults"].astype(np.float32),
        "y_anomaly": data["y_anomaly"].astype(np.float32),
    }


def _build_dataset_from_sources(dataset_path: str, external_root: str, random_seed: int):
    config = MultiSourceConfig(
        ai4i_csv=os.path.join(BASE_DIR, "data", "ai4i2020.csv"),
        cwru_dir=os.path.join(external_root, "CWRU"),
        mimii_dir=os.path.join(external_root, "MIMII"),
        metropt3_path=os.path.join(external_root, "MetroPT-3"),
        edgeiiot_path=os.path.join(external_root, "Edge-IIoTset"),
        random_seed=random_seed,
    )

    dataset, report = build_multisource_dataset(config)
    report_path = save_multisource_dataset(dataset_path, dataset, report)
    print(f"Saved fused training dataset: {dataset_path}")
    print(f"Saved dataset report: {report_path}")
    return dataset, report


def _apply_bootstrap(ds: dict[str, np.ndarray], size_multiplier: float, random_seed: int):
    indices = weighted_bootstrap_indices(
        y_faults=ds["y_faults"],
        y_anomaly=ds["y_anomaly"],
        size_multiplier=size_multiplier,
        seed=random_seed,
    )
    boot = {
        key: value[indices]
        for key, value in ds.items()
    }
    return boot


def _train_val_split(ds: dict[str, np.ndarray], random_seed: int):
    stratify = None
    if len(np.unique(ds["y_anomaly"])) > 1:
        stratify = ds["y_anomaly"]

    (
        Xp_train,
        Xp_val,
        Xv_train,
        Xv_val,
        Xa_train,
        Xa_val,
        Xe_train,
        Xe_val,
        Xt_train,
        Xt_val,
        yr_train,
        yr_val,
        yf_train,
        yf_val,
        ya_train,
        ya_val,
    ) = train_test_split(
        ds["X_process"],
        ds["X_vibration"],
        ds["X_acoustic"],
        ds["X_electrical"],
        ds["X_thermal"],
        ds["y_rul"],
        ds["y_faults"],
        ds["y_anomaly"],
        test_size=0.2,
        random_state=random_seed,
        stratify=stratify,
    )

    return {
        "Xp_train": Xp_train,
        "Xp_val": Xp_val,
        "Xv_train": Xv_train,
        "Xv_val": Xv_val,
        "Xa_train": Xa_train,
        "Xa_val": Xa_val,
        "Xe_train": Xe_train,
        "Xe_val": Xe_val,
        "Xt_train": Xt_train,
        "Xt_val": Xt_val,
        "yr_train": yr_train,
        "yr_val": yr_val,
        "yf_train": yf_train,
        "yf_val": yf_val,
        "ya_train": ya_train,
        "ya_val": ya_val,
    }


def train(
    dataset_path: str = DEFAULT_DATASET,
    external_root: str = DEFAULT_EXTERNAL_ROOT,
    epochs: int = 30,
    batch_size: int = 32,
    rebuild_dataset: bool = False,
    use_bootstrap: bool = True,
    bootstrap_multiplier: float = 1.25,
    random_seed: int = 42,
    prefer_gpu: bool = True,
    require_gpu: bool = False,
):
    tf.keras.utils.set_random_seed(random_seed)
    device_kind = _select_training_device(prefer_gpu=prefer_gpu, require_gpu=require_gpu)
    print(f"Training device mode: {device_kind}")

    use_mixed_precision = False
    if device_kind == "gpu":
        if batch_size < GPU_TARGET_BATCH_SIZE:
            print(f"Auto-tuning GPU batch size from {batch_size} to {GPU_TARGET_BATCH_SIZE}")
            batch_size = GPU_TARGET_BATCH_SIZE

        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            use_mixed_precision = True
            print("Enabled mixed precision policy: mixed_float16")
        except Exception as exc:
            print(f"Mixed precision not enabled: {exc}")

    steps_per_execution = GPU_STEPS_PER_EXECUTION if device_kind == "gpu" else 1
    print(f"Steps per execution: {steps_per_execution}")

    dataset_report: dict[str, object] = {}

    if (not rebuild_dataset) and os.path.exists(dataset_path):
        print(f"Loading fused dataset from: {dataset_path}")
        ds = _load_dataset(dataset_path)
    else:
        print("Building fused dataset from ai4i2020 + CWRU + MIMII + MetroPT-3 + Edge-IIoTset")
        ds, dataset_report = _build_dataset_from_sources(
            dataset_path=dataset_path,
            external_root=external_root,
            random_seed=random_seed,
        )

    if use_bootstrap:
        print(f"Applying weighted bootstrap resampling (x{bootstrap_multiplier:.2f})")
        ds = _apply_bootstrap(ds, size_multiplier=bootstrap_multiplier, random_seed=random_seed)

    split = _train_val_split(ds, random_seed=random_seed)

    fault_class_weights, sw_fault_train = multilabel_sample_weights(
        split["yf_train"],
        beta=0.9995,
        min_weight=1.0,
        max_weight=8.0,
    )
    anomaly_pos_weight, sw_anomaly_train = binary_sample_weights(
        split["ya_train"],
        beta=0.999,
        min_weight=1.0,
        max_weight=8.0,
    )
    sw_rul_train = np.clip(0.5 * sw_fault_train + 0.5 * sw_anomaly_train, 1.0, 8.0)

    model = build_multimodal_mtl_model(
        process_window=split["Xp_train"].shape[1],
        process_features=split["Xp_train"].shape[2],
        vibration_window=split["Xv_train"].shape[1],
        acoustic_window=split["Xa_train"].shape[1],
        electrical_window=split["Xe_train"].shape[1],
        electrical_features=split["Xe_train"].shape[2],
        thermal_embedding_dim=split["Xt_train"].shape[1],
        num_fault_classes=split["yf_train"].shape[1],
    )
    compile_multimodal_mtl_model(
        model,
        fault_class_weights=fault_class_weights,
        anomaly_pos_weight=anomaly_pos_weight,
        use_focal_losses=True,
        focal_gamma=2.0,
        steps_per_execution=steps_per_execution,
        jit_compile=False,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(BASE_DIR, "models", "best_multimodal_mtl.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    def _fit_for_batch(current_batch_size: int):
        return model.fit(
            [split["Xp_train"], split["Xv_train"], split["Xa_train"], split["Xe_train"], split["Xt_train"]],
            [split["yr_train"], split["yf_train"], split["ya_train"]],
            sample_weight=[sw_rul_train, sw_fault_train, sw_anomaly_train],
            validation_data=(
                [split["Xp_val"], split["Xv_val"], split["Xa_val"], split["Xe_val"], split["Xt_val"]],
                [split["yr_val"], split["yf_val"], split["ya_val"]],
            ),
            epochs=epochs,
            batch_size=current_batch_size,
            callbacks=callbacks,
            verbose=1,
        )

    try:
        history = _fit_for_batch(batch_size)
    except tf.errors.ResourceExhaustedError:
        if device_kind != "gpu" or batch_size <= 16:
            raise
        fallback_batch_size = max(16, batch_size // 2)
        print(
            f"GPU memory limit at batch_size={batch_size}; "
            f"retrying with batch_size={fallback_batch_size}"
        )
        batch_size = fallback_batch_size
        history = _fit_for_batch(batch_size)

    history_data = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }
    val_losses = history_data.get("val_loss", [])
    best_epoch = int(np.argmin(val_losses) + 1) if val_losses else int(len(history_data.get("loss", [])))
    best_val_loss = float(np.min(val_losses)) if val_losses else None

    val_fault_auc_key = None
    for key in [
        "val_head_faults_auc",
        "val_head_faults_accuracy",
        "val_head_anomaly_score_auc",
    ]:
        if key in history_data:
            val_fault_auc_key = key
            break

    history_report = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "metrics": history_data,
    }
    if val_fault_auc_key is not None and history_data.get(val_fault_auc_key):
        history_report["best_val_aux_metric"] = {
            "name": val_fault_auc_key,
            "value": float(np.max(history_data[val_fault_auc_key])),
        }

    history_path = os.path.join(BASE_DIR, "models", "multisource_training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_report, f, indent=2)

    training_report = {
        "dataset": {
            "path": dataset_path,
            "num_samples_after_bootstrap": int(len(split["Xp_train"]) + len(split["Xp_val"])),
            "anomaly_rate": float(np.mean(np.concatenate([split["ya_train"], split["ya_val"]]) > 0.5)),
        },
        "training": {
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
            "history_path": history_path,
        },
        "imbalance": {
            "fault_class_weights": [float(x) for x in fault_class_weights],
            "anomaly_pos_weight": float(anomaly_pos_weight),
            "train_fault_positive_rate": [
                float(np.mean(split["yf_train"][:, i] > 0.5))
                for i in range(split["yf_train"].shape[1])
            ],
            "train_anomaly_rate": float(np.mean(split["ya_train"] > 0.5)),
        },
        "runtime": {
            "device_kind": device_kind,
            "batch_size": int(batch_size),
            "mixed_precision": bool(use_mixed_precision),
            "steps_per_execution": int(steps_per_execution),
        },
        "source_report": dataset_report,
    }

    report_path = os.path.join(BASE_DIR, "models", "multisource_training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(training_report, f, indent=2)

    print("Training complete. Saved best model to models/best_multimodal_mtl.keras")
    print(f"Saved training history to {history_path}")
    print(f"Saved training report to {report_path}")
    return model, history, training_report


def _parse_args():
    parser = argparse.ArgumentParser(description="Train multimodal MTL model with multi-dataset fusion")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to fused .npz dataset")
    parser.add_argument("--external-root", default=DEFAULT_EXTERNAL_ROOT, help="Root dir containing CWRU/MIMII/MetroPT-3/Edge-IIoTset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild fused .npz from source datasets")
    parser.add_argument("--no-bootstrap", action="store_true", help="Disable weighted bootstrap resampling")
    parser.add_argument("--bootstrap-multiplier", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(prefer_gpu=True)
    parser.add_argument("--cpu-only", action="store_false", dest="prefer_gpu", help="Force CPU mode")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if no GPU is available")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        dataset_path=args.dataset,
        external_root=args.external_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        rebuild_dataset=args.rebuild_dataset,
        use_bootstrap=not args.no_bootstrap,
        bootstrap_multiplier=args.bootstrap_multiplier,
        random_seed=args.seed,
        prefer_gpu=args.prefer_gpu,
        require_gpu=args.require_gpu,
    )
