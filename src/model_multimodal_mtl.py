"""
model_multimodal_mtl.py

Multimodal multi-task architecture for whole-machine health monitoring.

Heads:
- head_rul: overall system RUL regression
- head_faults: multi-label fault classification
- head_anomaly_score: unknown-behavior anomaly score
"""

import tensorflow as tf
from tensorflow.keras import Model, layers

from src.imbalance import build_weighted_binary_focal_loss, build_weighted_focal_bce


def _temporal_encoder(inputs, filters=64, lstm_units=64, dropout=0.2, mc_dropout=False, name_prefix="enc"):
    x = layers.Conv1D(filters, kernel_size=5, padding="same", activation="relu", name=f"{name_prefix}_conv1")(inputs)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Conv1D(filters, kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=dropout),
        name=f"{name_prefix}_bilstm",
    )(x)
    x = layers.GlobalAveragePooling1D(name=f"{name_prefix}_gap")(x)
    dropout_layer = layers.Dropout(dropout, name=f"{name_prefix}_dropout")
    if mc_dropout:
        # MC Dropout: apply dropout even at inference time for uncertainty
        dropout_layer = layers.Dropout(dropout, name=f"{name_prefix}_mc_dropout")
    x = dropout_layer(x, training=True) if mc_dropout else dropout_layer(x)
    return x


def build_multimodal_mtl_model(
    process_window=30,
    process_features=14,
    vibration_window=256,
    acoustic_window=2048,
    electrical_window=64,
    electrical_features=4,
    thermal_embedding_dim=128,
    num_fault_classes=6,
    mc_dropout=False,
):
    process_input = layers.Input(shape=(process_window, process_features), name="process_input")
    vibration_input = layers.Input(shape=(vibration_window, 3), name="vibration_input")
    acoustic_input = layers.Input(shape=(acoustic_window, 1), name="acoustic_input")
    electrical_input = layers.Input(shape=(electrical_window, electrical_features), name="electrical_input")
    thermal_input = layers.Input(shape=(thermal_embedding_dim,), name="thermal_input")

    process_vec = _temporal_encoder(process_input, filters=64, lstm_units=64, mc_dropout=mc_dropout, name_prefix="process")
    vibration_vec = _temporal_encoder(vibration_input, filters=64, lstm_units=64, mc_dropout=mc_dropout, name_prefix="vibration")
    acoustic_vec = _temporal_encoder(acoustic_input, filters=32, lstm_units=32, mc_dropout=mc_dropout, name_prefix="acoustic")
    electrical_vec = _temporal_encoder(electrical_input, filters=32, lstm_units=32, mc_dropout=mc_dropout, name_prefix="electrical")

    thermal_vec = layers.Dense(64, activation="relu", name="thermal_dense1")(thermal_input)
    thermal_vec = layers.Dropout(0.2, name="thermal_dropout")(thermal_vec)

    fused = layers.Concatenate(name="sensor_fusion_concat")(
        [process_vec, vibration_vec, acoustic_vec, electrical_vec, thermal_vec]
    )

    shared = layers.Dense(256, activation="relu", name="shared_dense1")(fused)
    shared = layers.Dropout(0.3, name="shared_dropout1")(shared)
    shared = layers.Dense(128, activation="relu", name="shared_dense2")(shared)
    shared = layers.Dropout(0.3, name="shared_dropout2")(shared)

    # Head 1: system-level RUL
    head_rul = layers.Dense(64, activation="relu", name="rul_dense")(shared)
    # Keep output heads in float32 for numerical stability under mixed precision.
    head_rul = layers.Dense(1, activation="relu", name="head_rul", dtype="float32")(head_rul)

    # Head 2: multi-label fault diagnosis
    head_faults = layers.Dense(64, activation="relu", name="fault_dense")(shared)
    head_faults = layers.Dense(
        num_fault_classes,
        activation="sigmoid",
        name="head_faults",
        dtype="float32",
    )(head_faults)

    # Head 3: anomaly score for unknown behaviors
    head_anomaly_score = layers.Dense(32, activation="relu", name="anomaly_dense")(shared)
    head_anomaly_score = layers.Dense(
        1,
        activation="sigmoid",
        name="head_anomaly_score",
        dtype="float32",
    )(head_anomaly_score)

    return Model(
        inputs=[process_input, vibration_input, acoustic_input, electrical_input, thermal_input],
        outputs=[head_rul, head_faults, head_anomaly_score],
        name="Multimodal_MTL_Health",
    )


def compile_multimodal_mtl_model(
    model,
    rul_weight=1.0,
    faults_weight=2.0,
    anomaly_weight=1.0,
    learning_rate=1e-3,
    fault_class_weights=None,
    anomaly_pos_weight=1.0,
    use_focal_losses=True,
    focal_gamma=2.0,
    steps_per_execution=1,
    jit_compile=False,
):
    if fault_class_weights is None:
        fault_class_weights = [1.0] * int(model.get_layer("head_faults").output_shape[-1])

    if use_focal_losses:
        faults_loss = build_weighted_focal_bce(
            class_weights=fault_class_weights,
            gamma=focal_gamma,
            label_smoothing=0.01,
        )
        anomaly_loss = build_weighted_binary_focal_loss(
            pos_weight=float(anomaly_pos_weight),
            gamma=focal_gamma,
            label_smoothing=0.01,
        )
    else:
        faults_loss = "binary_crossentropy"
        anomaly_loss = "binary_crossentropy"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "head_rul": "mse",
            "head_faults": faults_loss,
            "head_anomaly_score": anomaly_loss,
        },
        loss_weights={
            "head_rul": rul_weight,
            "head_faults": faults_weight,
            "head_anomaly_score": anomaly_weight,
        },
        metrics={
            "head_rul": ["mae"],
            "head_faults": ["accuracy", tf.keras.metrics.AUC(name="auc")],
            "head_anomaly_score": [tf.keras.metrics.AUC(name="auc")],
        },
        steps_per_execution=steps_per_execution,
        jit_compile=jit_compile,
    )
    return model


if __name__ == "__main__":
    model = build_multimodal_mtl_model()
    compile_multimodal_mtl_model(model)
    model.summary()
