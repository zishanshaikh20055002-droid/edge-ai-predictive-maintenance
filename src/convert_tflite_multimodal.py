"""
convert_tflite_multimodal.py

Converts the multimodal Keras model into a portable TFLite artifact so
Windows runtime can run full multimodal inference without loading `.keras`.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_multimodal_mtl.keras")
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "data", "multisource_train.npz")
DEFAULT_OUT_PATH = os.path.join(BASE_DIR, "models", "model_multimodal_portable.tflite")


def _representative_data_gen(dataset_path: str, sample_count: int = 200, seed: int = 42):
    if os.path.exists(dataset_path):
        data = np.load(dataset_path)
        X_process = data["X_process"].astype(np.float32)
        X_vibration = data["X_vibration"].astype(np.float32)
        X_acoustic = data["X_acoustic"].astype(np.float32)
        X_electrical = data["X_electrical"].astype(np.float32)
        X_thermal = data["X_thermal"].astype(np.float32)

        rng = np.random.default_rng(seed)
        indices = np.arange(len(X_process), dtype=np.int64)
        if len(indices) > sample_count:
            indices = np.sort(rng.choice(indices, size=sample_count, replace=False))

        for i in indices:
            yield [
                X_process[i : i + 1],
                X_vibration[i : i + 1],
                X_acoustic[i : i + 1],
                X_electrical[i : i + 1],
                X_thermal[i : i + 1],
            ]
        return

    print(f"[WARN] Dataset not found for representative data: {dataset_path}")
    print("[WARN] Falling back to random representative samples")
    rng = np.random.default_rng(seed)
    for _ in range(sample_count):
        yield [
            rng.normal(size=(1, 30, 14)).astype(np.float32),
            rng.normal(size=(1, 256, 3)).astype(np.float32),
            rng.normal(size=(1, 2048, 1)).astype(np.float32),
            rng.normal(size=(1, 64, 4)).astype(np.float32),
            rng.normal(size=(1, 128)).astype(np.float32),
        ]


def convert_model(
    model_path: str,
    out_path: str,
    dataset_path: str,
    quantization: str = "float16",
    sample_count: int = 200,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading multimodal model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    # BiLSTM-heavy graphs are more reliable with this setting during conversion.
    converter._experimental_lower_tensor_list_ops = False

    quantization = quantization.strip().lower()
    if quantization == "float16":
        converter.target_spec.supported_types = [tf.float16]
        print("Using float16 quantization")
    elif quantization == "int8":
        converter.representative_dataset = lambda: _representative_data_gen(
            dataset_path=dataset_path,
            sample_count=sample_count,
        )
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        print("Using int8 quantization with representative dataset")
    else:
        raise ValueError("quantization must be one of: float16, int8")

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved TFLite model: {out_path} ({len(tflite_model) / 1024:.1f} KB)")

    # Quick runtime verification of tensor signatures.
    interpreter = tf.lite.Interpreter(model_path=out_path)
    interpreter.allocate_tensors()
    print("Input tensors:")
    for detail in interpreter.get_input_details():
        print(f"  - {detail['name']}: shape={detail['shape']} dtype={detail['dtype']}")
    print("Output tensors:")
    for detail in interpreter.get_output_details():
        print(f"  - {detail['name']}: shape={detail['shape']} dtype={detail['dtype']}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Convert multimodal Keras model to portable TFLite")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to .keras multimodal model")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="Path to output .tflite model")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Path to fused dataset .npz for int8 calibration")
    parser.add_argument("--quantization", choices=["float16", "int8"], default="float16")
    parser.add_argument("--sample-count", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert_model(
        model_path=args.model,
        out_path=args.out,
        dataset_path=args.dataset,
        quantization=args.quantization,
        sample_count=args.sample_count,
    )
