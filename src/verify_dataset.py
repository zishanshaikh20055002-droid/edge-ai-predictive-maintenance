"""
verify_dataset.py

Command-line tool to validate and analyze dataset integrity before training.

Usage:
  python -m src.verify_dataset --npz data/multisource_train.npz
  python -m src.verify_dataset --source ai4i --csv data/ai4i2020.csv
  python -m src.verify_dataset --source cwru --dir data/external/CWRU
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def verify_npz_dataset(npz_path: str) -> dict[str, Any]:
    """
    Validate a fused multisource .npz dataset.
    
    Checks:
    - File exists and is readable
    - All expected keys present
    - Shape consistency
    - Data type correctness
    - Value ranges and NaN/Inf detection
    - Class balance statistics
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset not found: {npz_path}")
    
    try:
        data = np.load(npz_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load NPZ: {exc}")
    
    expected_keys = {
        "X_process", "X_vibration", "X_acoustic", "X_electrical", "X_thermal",
        "y_rul", "y_faults", "y_anomaly"
    }
    
    missing = expected_keys - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    
    report: dict[str, Any] = {
        "file": npz_path,
        "exists": True,
        "shapes": {},
        "dtypes": {},
        "issues": [],
        "stats": {},
    }
    
    n_samples = None
    for key in expected_keys:
        arr = data[key]
        report["shapes"][key] = tuple(arr.shape)
        report["dtypes"][key] = str(arr.dtype)
        
        # Check first dimension (samples) is consistent
        if n_samples is None:
            n_samples = len(arr)
        elif len(arr) != n_samples:
            report["issues"].append(
                f"Shape mismatch: {key} has {len(arr)} samples, expected {n_samples}"
            )
        
        # Check for NaN and Inf
        if np.issubdtype(arr.dtype, np.floating):
            nan_count = int(np.sum(np.isnan(arr)))
            inf_count = int(np.sum(np.isinf(arr)))
            
            if nan_count > 0:
                report["issues"].append(f"{key}: {nan_count} NaN values found")
            if inf_count > 0:
                report["issues"].append(f"{key}: {inf_count} Inf values found")
        
        # Statistics
        if np.issubdtype(arr.dtype, np.floating):
            report["stats"][key] = {
                "min": float(np.nanmin(arr)) if arr.size > 0 else None,
                "max": float(np.nanmax(arr)) if arr.size > 0 else None,
                "mean": float(np.nanmean(arr)) if arr.size > 0 else None,
                "std": float(np.nanstd(arr)) if arr.size > 0 else None,
            }
    
    # Class balance for binary targets
    if "y_anomaly" in data:
        y_anom = data["y_anomaly"]
        pos_count = int(np.sum(y_anom > 0.5))
        neg_count = int(len(y_anom) - pos_count)
        report["anomaly_balance"] = {
            "positive": pos_count,
            "negative": neg_count,
            "ratio": float(pos_count / (neg_count + 1e-6)),
        }
    
    # Fault balance
    if "y_faults" in data:
        y_faults = data["y_faults"]
        if y_faults.ndim == 2:
            per_class = {}
            for c in range(y_faults.shape[1]):
                pos = int(np.sum(y_faults[:, c] > 0.5))
                per_class[f"class_{c}"] = pos
            report["fault_balance"] = per_class
    
    report["summary"] = {
        "total_samples": int(n_samples),
        "memory_mb": float(sum(data[k].nbytes for k in data.keys()) / 1024 / 1024),
        "has_issues": len(report["issues"]) > 0,
    }
    
    return report


def verify_ai4i_csv(csv_path: str) -> dict[str, Any]:
    """Validate AI4I2020 CSV dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, nrows=10000)  # Limit to first 10k rows for speed
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV: {exc}")
    
    report: dict[str, Any] = {
        "file": csv_path,
        "rows": len(df),
        "columns": len(df.columns),
        "issues": [],
        "features": {},
        "targets": {},
    }
    
    expected_features = [
        "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ]
    
    for feat in expected_features:
        if feat in df.columns:
            col = df[feat]
            report["features"][feat] = {
                "dtype": str(col.dtype),
                "missing": int(col.isnull().sum()),
                "min": float(col.min()),
                "max": float(col.max()),
            }
        else:
            report["issues"].append(f"Missing feature: {feat}")
    
    # Check target columns
    for tgt in ["TWF", "HDF", "PWF", "OSF", "RNF", "Machine failure"]:
        if tgt in df.columns:
            col = df[tgt]
            report["targets"][tgt] = {
                "positives": int(col.sum() if col.dtype in [int, float] else (col == 1).sum()),
                "missing": int(col.isnull().sum()),
            }
    
    report["summary"] = {
        "total_features_present": len(report["features"]),
        "total_targets_present": len(report["targets"]),
        "has_issues": len(report["issues"]) > 0,
    }
    
    return report


def verify_cwru_dir(dir_path: str) -> dict[str, Any]:
    """Validate CWRU bearing dataset directory."""
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    mat_files = sorted(Path(dir_path).rglob("*.mat"))
    
    report: dict[str, Any] = {
        "directory": dir_path,
        "mat_files_found": len(mat_files),
        "issues": [],
        "sample_files": [],
    }
    
    if not mat_files:
        report["issues"].append("No .mat files found")
    else:
        try:
            from scipy.io import loadmat
            
            for mat_file in mat_files[:3]:  # Sample first 3 files
                try:
                    data = loadmat(str(mat_file), squeeze_me=True)
                    keys = [k for k in data.keys() if not k.startswith("__")]
                    report["sample_files"].append({
                        "name": mat_file.name,
                        "keys": keys[:5],  # First 5 keys
                    })
                except Exception as e:
                    report["issues"].append(f"Error reading {mat_file.name}: {e}")
        except ImportError:
            report["issues"].append("scipy.io.loadmat not available for validation")
    
    report["summary"] = {
        "has_issues": len(report["issues"]) > 0,
    }
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Verify and analyze datasets for multisource training"
    )
    parser.add_argument(
        "--npz",
        type=str,
        help="Path to fused .npz dataset to verify"
    )
    parser.add_argument(
        "--source",
        choices=["ai4i", "cwru", "mimii", "metropt3", "edgeiiot"],
        help="Individual source dataset type to verify"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file (for ai4i/metropt3/edgeiiot)"
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Path to directory (for cwru/mimii)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save report to JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed report"
    )
    
    args = parser.parse_args()
    
    if not (args.npz or args.source):
        parser.print_help()
        sys.exit(1)
    
    report = {}
    
    try:
        if args.npz:
            print(f"Verifying fused dataset: {args.npz}")
            report = verify_npz_dataset(args.npz)
            print("✅ NPZ dataset verification passed")
        
        elif args.source == "ai4i":
            if not args.csv:
                raise ValueError("--csv required for ai4i verification")
            print(f"Verifying AI4I dataset: {args.csv}")
            report = verify_ai4i_csv(args.csv)
            print("✅ AI4I CSV verification passed")
        
        elif args.source == "cwru":
            if not args.dir:
                raise ValueError("--dir required for cwru verification")
            print(f"Verifying CWRU dataset: {args.dir}")
            report = verify_cwru_dir(args.dir)
            print("✅ CWRU directory verification passed")
        
        else:
            print(f"Verification for {args.source} not yet implemented")
            sys.exit(1)
        
        if args.verbose:
            print("\n" + "=" * 60)
            print("DETAILED REPORT:")
            print("=" * 60)
            print(json.dumps(report, indent=2, default=str))
        
        if report.get("summary", {}).get("has_issues"):
            print("\n⚠️  Issues detected:")
            for issue in report.get("issues", []):
                print(f"  - {issue}")
        
        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nReport saved to {args.output}")
    
    except Exception as exc:
        print(f"❌ Verification failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
