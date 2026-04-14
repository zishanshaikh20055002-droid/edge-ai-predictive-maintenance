"""
imbalance.py

Advanced imbalance helpers for multi-task training.

Techniques implemented:
- Effective-number class weighting (class-balanced loss weighting)
- Multi-label sample weighting for rare faults
- Binary sample weighting for anomaly heads
- Weighted bootstrap resampling for minority enrichment
- Class-weighted focal binary cross-entropy
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def effective_number_weights(class_counts: np.ndarray, beta: float = 0.9995) -> np.ndarray:
    """
    Compute class-balanced weights from effective number of samples.

    Paper intuition: very large classes should not dominate gradients.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    counts = np.clip(counts, 1.0, None)

    beta = float(np.clip(beta, 0.0, 0.999999))
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.clip(effective_num, 1e-8, None)

    # Normalize to keep average weight ~1 for stable optimization.
    weights = weights / np.mean(weights)
    return weights.astype(np.float32)


def multilabel_sample_weights(
    y_multilabel: np.ndarray,
    beta: float = 0.9995,
    min_weight: float = 1.0,
    max_weight: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-class and per-sample weights for a multi-label target matrix.

    Returns:
        class_weights: (C,) weights for positive labels per class
        sample_weights: (N,) weight per sample
    """
    y = np.asarray(y_multilabel, dtype=np.float32)
    if y.ndim != 2:
        raise ValueError("y_multilabel must have shape (N, C)")

    class_counts = np.sum(y > 0.5, axis=0).astype(np.float64)
    class_counts = np.clip(class_counts, 1.0, None)
    class_weights = effective_number_weights(class_counts, beta=beta)

    # Samples carrying rare-positive classes get larger weight.
    weighted_positive_mass = np.sum(y * class_weights[None, :], axis=1)
    sample_weights = 1.0 + weighted_positive_mass

    sample_weights = np.clip(sample_weights, min_weight, max_weight).astype(np.float32)
    class_weights = class_weights.astype(np.float32)
    return class_weights, sample_weights


def binary_sample_weights(
    y_binary: np.ndarray,
    beta: float = 0.999,
    min_weight: float = 1.0,
    max_weight: float = 8.0,
) -> tuple[float, np.ndarray]:
    """
    Build positive-class weight and per-sample weights for binary labels.

    Returns:
        pos_weight: scalar weight for positive class
        sample_weights: (N,) sample weights
    """
    y = np.asarray(y_binary, dtype=np.float32).reshape(-1)
    pos_count = float(np.sum(y > 0.5))
    neg_count = float(len(y) - pos_count)

    pos_count = max(pos_count, 1.0)
    neg_count = max(neg_count, 1.0)

    class_weights = effective_number_weights(np.array([neg_count, pos_count]), beta=beta)
    pos_weight = float(class_weights[1])
    neg_weight = float(class_weights[0])

    sample_weights = np.where(y > 0.5, pos_weight, neg_weight).astype(np.float32)
    sample_weights = np.clip(sample_weights, min_weight, max_weight)
    return pos_weight, sample_weights


def weighted_bootstrap_indices(
    y_faults: np.ndarray,
    y_anomaly: np.ndarray,
    size_multiplier: float = 1.25,
    target_anomaly_rate: float = 0.65,
    seed: int = 42,
) -> np.ndarray:
    """
    Weighted bootstrap resampling index generator for multi-task data.

    Rare fault labels are sampled more frequently while anomaly prevalence
    is nudged toward target_anomaly_rate to avoid skew blow-up.
    """
    faults = np.asarray(y_faults, dtype=np.float32)
    anomaly = np.asarray(y_anomaly, dtype=np.float32).reshape(-1)

    if faults.ndim != 2:
        raise ValueError("y_faults must have shape (N, C)")
    if len(anomaly) != len(faults):
        raise ValueError("y_anomaly and y_faults must have same N")

    n = len(faults)
    if n == 0:
        return np.array([], dtype=np.int64)

    # Rarity score from inverse-positive-frequency per class.
    class_counts = np.sum(faults > 0.5, axis=0).astype(np.float64)
    inv_freq = 1.0 / np.clip(class_counts, 1.0, None)
    fault_score = np.sum(faults * inv_freq[None, :], axis=1)

    anomaly_binary = (anomaly > 0.5).astype(np.float32)
    pos_count = float(np.sum(anomaly_binary))
    neg_count = float(n - pos_count)

    if pos_count > 0.0 and neg_count > 0.0:
        observed_rate = pos_count / float(n)
        target_rate = float(np.clip(target_anomaly_rate, 0.05, 0.95))

        pos_factor = target_rate / max(observed_rate, 1e-6)
        neg_factor = (1.0 - target_rate) / max(1.0 - observed_rate, 1e-6)

        # Keep class-balancing factors bounded to avoid unstable resampling.
        pos_factor = float(np.clip(pos_factor, 0.25, 4.0))
        neg_factor = float(np.clip(neg_factor, 0.25, 4.0))

        anomaly_balance = np.where(anomaly_binary > 0.5, pos_factor, neg_factor)
    else:
        anomaly_balance = np.ones((n,), dtype=np.float32)

    score = (1.0 + fault_score) * anomaly_balance
    score = np.clip(score, 1e-6, None)
    probs = score / np.sum(score)

    rng = np.random.default_rng(seed)
    size = int(max(1, round(n * float(size_multiplier))))
    indices = rng.choice(np.arange(n), size=size, replace=True, p=probs)
    return indices.astype(np.int64)


def build_weighted_focal_bce(
    class_weights: np.ndarray,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
):
    """
    Create a class-weighted focal BCE loss for multi-label classification.
    """
    alpha = tf.constant(np.asarray(class_weights, dtype=np.float32), dtype=tf.float32)
    gamma = float(gamma)
    label_smoothing = float(np.clip(label_smoothing, 0.0, 0.2))

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if label_smoothing > 0:
            y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

        y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)

        # alpha_t: class-wise balance term (applied on positives)
        alpha_t = y_true * alpha + (1.0 - y_true)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_factor = tf.pow(1.0 - p_t, gamma)

        bce = -(
            y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )
        loss = alpha_t * focal_factor * bce
        return tf.reduce_mean(loss)

    return loss_fn


def build_weighted_binary_focal_loss(
    pos_weight: float,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
):
    """
    Binary focal loss with explicit positive-class weighting.
    """
    pos_weight = float(max(pos_weight, 1e-3))
    gamma = float(gamma)
    label_smoothing = float(np.clip(label_smoothing, 0.0, 0.2))

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if label_smoothing > 0:
            y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

        y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)

        alpha_t = y_true * pos_weight + (1.0 - y_true)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_factor = tf.pow(1.0 - p_t, gamma)

        bce = -(
            y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )
        loss = alpha_t * focal_factor * bce
        return tf.reduce_mean(loss)

    return loss_fn
