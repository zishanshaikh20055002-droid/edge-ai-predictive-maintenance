import numpy as np

from src.evaluation_metrics import (
    anomaly_metrics,
    compute_all_metrics,
    fault_metrics,
    rul_metrics,
    uncertainty_metrics,
)


def test_rul_metrics_perfect_prediction():
    y_true = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    y_pred = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    out = rul_metrics(y_true, y_pred)

    assert out["mae"] == 0.0
    assert out["rmse"] == 0.0
    assert out["mape"] == 0.0
    assert abs(out["r2"] - 1.0) < 1e-6


def test_fault_metrics_multilabel_basic():
    y_true = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=np.float32,
    )
    y_pred = np.array(
        [
            [0.9, 0.1, 0.1],
            [0.2, 0.8, 0.2],
            [0.2, 0.2, 0.9],
            [0.7, 0.3, 0.2],
        ],
        dtype=np.float32,
    )

    out = fault_metrics(y_true, y_pred, threshold=0.5)

    assert 0.0 <= out["accuracy"] <= 1.0
    assert 0.0 <= out["hamming_loss"] <= 1.0
    assert 0.0 <= out["macro_f1"] <= 1.0
    assert len(out["per_class"]) == 3


def test_anomaly_metrics_respects_threshold_override():
    y_true = np.array([0, 0, 1, 1], dtype=np.float32)
    y_pred = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)

    loose = anomaly_metrics(y_true, y_pred, threshold=0.5)
    strict = anomaly_metrics(y_true, y_pred, threshold=0.9)

    assert loose["best_f1_threshold"] == 0.5
    assert strict["best_f1_threshold"] == 0.9
    assert strict["best_f1"] <= loose["best_f1"]


def test_uncertainty_metrics_interval_width_and_coverage():
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_pred = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    unc = np.array([0.1, 0.1, 0.1], dtype=np.float32)

    out = uncertainty_metrics(y_true, y_pred, unc, percentile=0.95)

    assert out["target_coverage"] == 0.95
    assert out["coverage"] == 1.0
    # For 95% interval, width ~= 2 * 1.96 * 0.1
    assert abs(out["avg_interval_width"] - 0.392) < 0.03


def test_compute_all_metrics_has_all_heads():
    y_rul_true = np.array([5.0, 6.0], dtype=np.float32)
    y_rul_pred = np.array([5.5, 5.5], dtype=np.float32)

    y_faults_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
    y_faults_pred = np.array([[0.8, 0.1], [0.2, 0.9]], dtype=np.float32)

    y_anom_true = np.array([0, 1], dtype=np.float32)
    y_anom_pred = np.array([0.3, 0.7], dtype=np.float32)

    out = compute_all_metrics(
        y_rul_true,
        y_rul_pred,
        y_faults_true,
        y_faults_pred,
        y_anom_true,
        y_anom_pred,
        anomaly_threshold=0.6,
    )

    assert "rul" in out
    assert "faults" in out
    assert "anomaly" in out
    assert out["anomaly"]["best_f1_threshold"] == 0.6
