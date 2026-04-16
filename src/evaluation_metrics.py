"""
evaluation_metrics.py

Comprehensive evaluation metrics for multi-task learning models:
- RUL: MAE, RMSE, MAPE, R²
- Faults: Precision, Recall, F1 per class + macro/micro averages
- Anomaly: ROC-AUC, PR-AUC, F1@threshold
- Uncertainty: Coverage vs. accuracy of predictive intervals
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    accuracy_score,
)


def rul_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute RUL prediction metrics.
    
    Returns:
        Dict with MAE, RMSE, MAPE, R²
    """
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float32).flatten()
    
    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}
    
    # Mean Absolute Error
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    # Root Mean Squared Error
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # Mean Absolute Percentage Error (handle zero divisions)
    safe_true = np.where(np.abs(y_true) < 1e-6, 1e-6, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / safe_true))) * 100.0
    
    # R² Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8)) if ss_tot > 0 else 0.0
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
    }


def fault_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    compute_auc: bool = True,
) -> dict[str, float | dict[str, float]]:
    """
    Compute multi-label fault classification metrics.
    
    Args:
        y_true: (N, C) binary labels
        y_pred: (N, C) soft predictions [0, 1]
        threshold: classification threshold
        compute_auc: whether to compute ROC-AUC per class
    
    Returns:
        Dict with per-class and macro/micro averages
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    
    if y_true.ndim != 2:
        raise ValueError(f"Expected 2D labels, got shape {y_true.shape}")
    
    if len(y_true) == 0:
        return {"accuracy": 0.0, "hamming_loss": 0.0}
    
    y_pred_binary = (y_pred > threshold).astype(np.float32)
    n_classes = y_true.shape[1]
    
    # Per-class metrics
    per_class = {}
    for c in range(n_classes):
        y_c_true = y_true[:, c]
        y_c_pred = y_pred[:, c]
        y_c_pred_binary = y_pred_binary[:, c]
        
        # Skip if class is absent in both true and pred
        if np.sum(y_c_true) == 0 and np.sum(y_c_pred_binary) == 0:
            per_class[f"class_{c}"] = {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "support": 0,
            }
            if compute_auc:
                per_class[f"class_{c}"]["roc_auc"] = np.nan
            continue
        
        precision = float(precision_score(y_c_true, y_c_pred_binary, zero_division=0))
        recall = float(recall_score(y_c_true, y_c_pred_binary, zero_division=0))
        f1 = float(f1_score(y_c_true, y_c_pred_binary, zero_division=0))
        support = int(np.sum(y_c_true))
        
        per_class[f"class_{c}"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        
        if compute_auc and len(np.unique(y_c_true)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_c_true, y_c_pred))
                per_class[f"class_{c}"]["roc_auc"] = roc_auc
            except Exception:
                per_class[f"class_{c}"]["roc_auc"] = np.nan
    
    # Macro/Micro averages
    macro_f1 = float(np.nanmean([v["f1"] for v in per_class.values()]))
    accuracy = float(accuracy_score(y_true, y_pred_binary))
    hamming = float(hamming_loss(y_true, y_pred_binary))
    
    return {
        "accuracy": accuracy,
        "hamming_loss": hamming,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def anomaly_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float | None = None,
    compute_pr_curve: bool = False,
) -> dict[str, float | dict[str, np.ndarray]]:
    """
    Compute binary anomaly detection metrics.
    
    Args:
        y_true: (N,) binary labels {0, 1}
        y_pred: (N,) soft predictions [0, 1]
        compute_pr_curve: whether to compute precision-recall curve
    
    Returns:
        Dict with ROC-AUC, PR-AUC, F1@threshold, etc.
    """
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float32).flatten()
    
    if len(y_true) == 0:
        return {"roc_auc": np.nan, "pr_auc": np.nan}
    
    # ROC-AUC
    if len(np.unique(y_true)) < 2:
        roc_auc = np.nan
    else:
        try:
            roc_auc = float(roc_auc_score(y_true, y_pred))
        except Exception:
            roc_auc = np.nan
    
    # PR-AUC with F1@optimal threshold (unless threshold override is provided)
    pr_auc = np.nan
    best_threshold = 0.5 if threshold is None else float(np.clip(threshold, 0.0, 1.0))
    
    if len(np.unique(y_true)) > 1:
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            pr_auc = float(auc(recall, precision))
            
            # Find F1-optimal threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.nanargmax(f1_scores)
            if threshold is None:
                best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        except Exception:
            pass
    
    y_pred_binary = (y_pred > best_threshold).astype(np.float32)
    optimal_f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))
    
    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_f1_threshold": float(best_threshold),
        "best_f1": optimal_f1,
    }
    
    if compute_pr_curve:
        metrics["pr_curve"] = {
            "precision": precision.tolist() if 'precision' in locals() else [],
            "recall": recall.tolist() if 'recall' in locals() else [],
        }
    
    return metrics


def uncertainty_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    percentile: float = 0.95,
) -> dict[str, float]:
    """
    Compute predictive interval coverage and width metrics.
    
    Args:
        y_true: (N,) regression targets
        y_pred: (N,) point predictions (mean)
        uncertainty: (N,) prediction uncertainty (std dev)
        percentile: coverage percentile (e.g., 0.95 for 95% PI)
    
    Returns:
        Coverage, average interval width, calibration metrics
    """
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float32).flatten()
    uncertainty = np.asarray(uncertainty, dtype=np.float32).flatten()
    
    percentile = float(np.clip(percentile, 1e-6, 1.0 - 1e-6))

    # Two-sided normal interval, e.g. percentile=0.95 -> z=1.96
    try:
        from scipy.stats import norm

        z_score = float(norm.ppf(0.5 + percentile / 2.0))
    except Exception:
        fallback = {
            0.8: 1.2815515655,
            0.9: 1.6448536269,
            0.95: 1.9599639845,
            0.99: 2.5758293035,
        }
        nearest = min(fallback.keys(), key=lambda k: abs(k - percentile))
        z_score = fallback[nearest]
    
    lower = y_pred - z_score * uncertainty
    upper = y_pred + z_score * uncertainty
    
    # Coverage: fraction of points within predictive interval
    coverage = float(
        np.mean((y_true >= lower) & (y_true <= upper))
    )
    
    # Average interval width
    avg_width = float(np.mean(upper - lower))
    
    # Calibration: ideal coverage == percentile
    miscalibration = float(np.abs(coverage - percentile))
    
    # Sharpness: lower average uncertainty is sharper
    avg_uncertainty = float(np.mean(uncertainty))
    
    return {
        "coverage": coverage,
        "target_coverage": percentile,
        "miscalibration": miscalibration,
        "avg_interval_width": avg_width,
        "avg_uncertainty": avg_uncertainty,
    }


def compute_all_metrics(
    y_rul_true: np.ndarray,
    y_rul_pred: np.ndarray,
    y_faults_true: np.ndarray,
    y_faults_pred: np.ndarray,
    y_anomaly_true: np.ndarray,
    y_anomaly_pred: np.ndarray,
    y_rul_uncertainty: np.ndarray | None = None,
    fault_threshold: float = 0.5,
    anomaly_threshold: float = 0.5,
) -> dict[str, dict[str, float] | float]:
    """
    Compute all three task metrics in one call.
    
    Returns:
        Nested dict with "rul", "faults", "anomaly" keys
    """
    metrics = {
        "rul": rul_metrics(y_rul_true, y_rul_pred),
        "faults": fault_metrics(y_faults_true, y_faults_pred, threshold=fault_threshold),
        "anomaly": anomaly_metrics(y_anomaly_true, y_anomaly_pred, threshold=anomaly_threshold),
    }
    
    if y_rul_uncertainty is not None:
        metrics["rul_uncertainty"] = uncertainty_metrics(
            y_rul_true,
            y_rul_pred,
            y_rul_uncertainty,
            percentile=0.95,
        )
    
    return metrics
