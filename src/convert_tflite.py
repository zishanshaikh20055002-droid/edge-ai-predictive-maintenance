import tensorflow as tf
import numpy as np
import os

print("Loading trained model...")

# adjust path if needed
model = tf.keras.models.load_model("models/best_model.keras")

print("Loading sample input...")

# dummy data for calibration
def representative_data_gen():
    for _ in range(100):
        yield [np.random.rand(1, 30, 5).astype(np.float32)]

print("Converting to INT8 TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

tflite_model = converter.convert()

os.makedirs("models", exist_ok=True)

with open("models/model_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete")

with open("models/model_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved at models/model_int8.tflite")