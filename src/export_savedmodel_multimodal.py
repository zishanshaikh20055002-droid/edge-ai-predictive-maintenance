"""
export_savedmodel_multimodal.py

Exports the multimodal Keras model to TensorFlow SavedModel format.
This avoids `.keras` class-deserialization issues across environments.
"""

from __future__ import annotations

import argparse
import os

import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_multimodal_mtl.keras")
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "models", "multimodal_savedmodel")


def export_saved_model(model_path: str, out_dir: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading multimodal model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    # Remove existing export to avoid stale signatures.
    if os.path.isdir(out_dir):
        import shutil

        shutil.rmtree(out_dir)

    exported = False
    if hasattr(model, "export"):
        try:
            model.export(out_dir)
            exported = True
        except Exception as exc:
            print(f"[WARN] model.export failed, retrying with tf.saved_model.save: {exc}")

    if not exported:
        tf.saved_model.save(model, out_dir)

    if not os.path.exists(os.path.join(out_dir, "saved_model.pb")):
        raise RuntimeError("SavedModel export did not produce saved_model.pb")

    loaded = tf.saved_model.load(out_dir)
    sig = loaded.signatures.get("serving_default")
    if sig is None:
        raise RuntimeError("SavedModel missing serving_default signature")

    print(f"Saved multimodal SavedModel: {out_dir}")
    print("Serving input keys:")
    for key, tensor in sig.structured_input_signature[1].items():
        print(f"  - {key}: shape={tensor.shape} dtype={tensor.dtype}")
    print("Serving output keys:")
    for key, tensor in sig.structured_outputs.items():
        print(f"  - {key}: shape={tensor.shape} dtype={tensor.dtype}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Export multimodal Keras model to SavedModel")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to .keras multimodal model")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="Output SavedModel directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export_saved_model(args.model, args.out)
