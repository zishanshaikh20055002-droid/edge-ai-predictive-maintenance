"""
multisource_dataset.py

Builds a unified multimodal training dataset from multiple industrial sources:
- ai4i2020 (tabular process + fault labels)
- CWRU Bearing Dataset (.mat vibration signals)
- MIMII Dataset (.wav acoustic anomaly data)
- MetroPT-3 (tabular compressor telemetry)
- Edge-IIoTset (tabular cyber/IIoT traffic)

The output matches the expected keys used by train_multimodal_mtl.py.
"""

from __future__ import annotations

import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from scipy.io import loadmat, wavfile
except Exception:  # pragma: no cover
    loadmat = None
    wavfile = None


FAULT_CLASSES = ["twf", "hdf", "pwf", "osf", "rnf", "bearing_fault"]
NUM_FAULT_CLASSES = len(FAULT_CLASSES)


@dataclass
class MultiSourceConfig:
    ai4i_csv: str
    cwru_dir: str
    mimii_dir: str
    metropt3_path: str
    edgeiiot_path: str
    process_window: int = 30
    process_features: int = 14
    vibration_window: int = 256
    acoustic_window: int = 2048
    electrical_window: int = 64
    electrical_features: int = 4
    thermal_embedding_dim: int = 128
    random_seed: int = 42
    max_cwru_windows: int = 20000
    max_mimii_windows: int = 25000
    max_metro_windows: int = 30000
    max_edge_windows: int = 30000
    max_target_samples: int = 30000


def _empty(shape: tuple[int, ...], dtype=np.float32):
    return np.zeros(shape, dtype=dtype)


def _safe_float_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(np.float32).values


def _find_first_csv(path: str, keywords: list[str]) -> str | None:
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".csv":
        return str(p)
    if not p.exists():
        return None

    keyword_set = [k.lower() for k in keywords]
    candidates = []
    for file_path in p.rglob("*.csv"):
        lower = str(file_path).lower()
        score = sum(1 for kw in keyword_set if kw in lower)
        if score > 0:
            candidates.append((score, str(file_path)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


def _select_numeric_columns(df: pd.DataFrame, exclude_keywords: list[str], k: int) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        return []

    excludes = tuple(x.lower() for x in exclude_keywords)
    cols = [
        c for c in numeric.columns
        if not any(token in c.lower() for token in excludes)
    ]
    if not cols:
        cols = list(numeric.columns)

    variances = numeric[cols].var(numeric_only=True).sort_values(ascending=False)
    chosen = list(variances.head(k).index)
    return chosen


def _pad_features(x: np.ndarray, target_dim: int) -> np.ndarray:
    if x.shape[-1] == target_dim:
        return x.astype(np.float32)
    if x.shape[-1] > target_dim:
        return x[:, :target_dim].astype(np.float32)

    out = np.zeros((x.shape[0], target_dim), dtype=np.float32)
    out[:, : x.shape[-1]] = x.astype(np.float32)
    return out


def _sliding_windows(
    x: np.ndarray,
    window: int,
    stride: int = 1,
    max_windows: int | None = None,
    return_starts: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if len(x) < window:
        empty = _empty((0, window, x.shape[1]), dtype=np.float32)
        if return_starts:
            return empty, np.zeros((0,), dtype=np.int64)
        return empty

    total = 1 + (len(x) - window) // stride
    if max_windows is not None and max_windows > 0 and total > max_windows:
        sampled = np.linspace(0, total - 1, num=max_windows, dtype=np.int64)
        starts = np.unique(sampled * stride)
    else:
        starts = np.arange(0, len(x) - window + 1, stride, dtype=np.int64)

    windows = [x[s : s + window] for s in starts]
    out = np.asarray(windows, dtype=np.float32)
    if return_starts:
        return out, starts
    return out


def _target_countdown_from_binary_events(events: np.ndarray, cap: float = 200.0) -> np.ndarray:
    """
    Create a pseudo-RUL target from binary failure/anomaly events.
    """
    y = np.zeros(len(events), dtype=np.float32)
    current = cap
    for idx in reversed(range(len(events))):
        if events[idx] >= 0.5:
            current = 0.0
        else:
            current = min(cap, current + 1.0)
        y[idx] = current
    return y


def _thermal_embed_from_process(process_window_data: np.ndarray, dim: int) -> np.ndarray:
    """
    Statistical embedding to mimic thermal/context channels until physical
    thermal embeddings are available from real hardware.
    """
    mean = np.mean(process_window_data, axis=0)
    std = np.std(process_window_data, axis=0)
    vmin = np.min(process_window_data, axis=0)
    vmax = np.max(process_window_data, axis=0)
    stats = np.concatenate([mean, std, vmin, vmax], axis=0).astype(np.float32)

    if len(stats) >= dim:
        return stats[:dim]

    repeats = int(np.ceil(dim / len(stats)))
    tiled = np.tile(stats, repeats)
    return tiled[:dim].astype(np.float32)


def _parse_ai4i(config: MultiSourceConfig) -> dict[str, np.ndarray]:
    if not os.path.exists(config.ai4i_csv):
        return {
            "X_process": _empty((0, config.process_window, config.process_features)),
            "y_rul": _empty((0,)),
            "y_faults": _empty((0, NUM_FAULT_CLASSES)),
            "y_anomaly": _empty((0,)),
        }

    df = pd.read_csv(config.ai4i_csv)

    feature_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < 3:
        numeric_cols = _select_numeric_columns(
            df,
            exclude_keywords=["id", "machine", "failure", "label", "target", "rul"],
            k=min(5, len(df.columns)),
        )
        available = numeric_cols

    if not available:
        return {
            "X_process": _empty((0, config.process_window, config.process_features)),
            "y_rul": _empty((0,)),
            "y_faults": _empty((0, NUM_FAULT_CLASSES)),
            "y_anomaly": _empty((0,)),
        }

    x_raw = df[available].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    x_raw = _pad_features(x_raw, config.process_features)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw).astype(np.float32)

    # RUL from backwards countdown to machine failure events.
    failure_col = None
    for cand in ["Machine failure", "machine_failure", "failure", "label", "target"]:
        if cand in df.columns:
            failure_col = cand
            break

    if failure_col is not None:
        failures = _safe_float_array(df[failure_col])
    else:
        failures = np.zeros(len(df), dtype=np.float32)

    y_rul_step = _target_countdown_from_binary_events(failures, cap=150.0)

    twf = _safe_float_array(df["TWF"]) if "TWF" in df.columns else np.zeros(len(df), dtype=np.float32)
    hdf = _safe_float_array(df["HDF"]) if "HDF" in df.columns else np.zeros(len(df), dtype=np.float32)
    pwf = _safe_float_array(df["PWF"]) if "PWF" in df.columns else np.zeros(len(df), dtype=np.float32)
    osf = _safe_float_array(df["OSF"]) if "OSF" in df.columns else np.zeros(len(df), dtype=np.float32)
    rnf = _safe_float_array(df["RNF"]) if "RNF" in df.columns else np.zeros(len(df), dtype=np.float32)

    y_fault_step = np.stack([twf, hdf, pwf, osf, rnf, np.zeros(len(df), dtype=np.float32)], axis=1)
    y_anomaly_step = np.clip(np.max(y_fault_step[:, :5], axis=1) + failures, 0.0, 1.0)

    Xw = _sliding_windows(x_scaled, config.process_window)
    if len(Xw) == 0:
        return {
            "X_process": _empty((0, config.process_window, config.process_features)),
            "y_rul": _empty((0,)),
            "y_faults": _empty((0, NUM_FAULT_CLASSES)),
            "y_anomaly": _empty((0,)),
        }

    idx = np.arange(config.process_window - 1, config.process_window - 1 + len(Xw))

    return {
        "X_process": Xw.astype(np.float32),
        "y_rul": y_rul_step[idx].astype(np.float32),
        "y_faults": y_fault_step[idx].astype(np.float32),
        "y_anomaly": y_anomaly_step[idx].astype(np.float32),
    }


def _parse_metropt3(config: MultiSourceConfig) -> dict[str, np.ndarray]:
    csv_path = _find_first_csv(config.metropt3_path, ["metro", "pt", "compressor", "metropt-3"])
    if not csv_path:
        return {
            "X_process": _empty((0, config.process_window, config.process_features)),
            "y_rul": _empty((0,)),
            "y_faults": _empty((0, NUM_FAULT_CLASSES)),
            "y_anomaly": _empty((0,)),
        }

    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        return {
            "X_process": _empty((0, config.process_window, config.process_features)),
            "y_rul": _empty((0,)),
            "y_faults": _empty((0, NUM_FAULT_CLASSES)),
            "y_anomaly": _empty((0,)),
        }

    process_cols = _select_numeric_columns(
        df,
        exclude_keywords=["label", "target", "fault", "anomaly", "class", "rul"],
        k=config.process_features,
    )
    if not process_cols:
        return {
            "X_process": _empty((0, config.process_window, config.process_features)),
            "y_rul": _empty((0,)),
            "y_faults": _empty((0, NUM_FAULT_CLASSES)),
            "y_anomaly": _empty((0,)),
        }

    x = df[process_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    x = _pad_features(x, config.process_features)
    x = StandardScaler().fit_transform(x).astype(np.float32)

    label_col = None
    for cand in ["anomaly", "failure", "label", "target", "fault", "class"]:
        matches = [c for c in df.columns if cand in c.lower()]
        if matches:
            label_col = matches[0]
            break

    if label_col:
        raw = df[label_col]
        if raw.dtype == object:
            anomaly_step = (~raw.astype(str).str.lower().isin(["normal", "0", "false", "ok"])).astype(np.float32).values
        else:
            anomaly_step = pd.to_numeric(raw, errors="coerce").fillna(0.0).astype(np.float32).values
            anomaly_step = (anomaly_step > 0).astype(np.float32)
    else:
        anomaly_step = np.zeros(len(df), dtype=np.float32)

    rul_col = None
    for cand in ["rul", "remaining_useful_life"]:
        matches = [c for c in df.columns if cand in c.lower()]
        if matches:
            rul_col = matches[0]
            break

    if rul_col:
        y_rul_step = pd.to_numeric(df[rul_col], errors="coerce").ffill().fillna(0.0).astype(np.float32).values
    else:
        y_rul_step = _target_countdown_from_binary_events(anomaly_step, cap=220.0)

    Xw, starts = _sliding_windows(
        x,
        config.process_window,
        max_windows=config.max_metro_windows,
        return_starts=True,
    )
    if len(Xw) == 0:
        return {
            "X_process": _empty((0, config.process_window, config.process_features)),
            "y_rul": _empty((0,)),
            "y_faults": _empty((0, NUM_FAULT_CLASSES)),
            "y_anomaly": _empty((0,)),
        }

    idx = starts + (config.process_window - 1)
    y_rul_step_sel = y_rul_step[idx]
    anomaly_step_sel = anomaly_step[idx]

    y_faults = np.zeros((len(Xw), NUM_FAULT_CLASSES), dtype=np.float32)

    return {
        "X_process": Xw.astype(np.float32),
        "y_rul": y_rul_step_sel.astype(np.float32),
        "y_faults": y_faults,
        "y_anomaly": anomaly_step_sel.astype(np.float32),
    }


def _extract_cwru_channels(mat_data: dict[str, Any]) -> list[np.ndarray]:
    channels = []
    for key, value in mat_data.items():
        if key.startswith("__"):
            continue
        if not isinstance(value, np.ndarray):
            continue

        flat = value.reshape(-1)
        if flat.ndim != 1 or len(flat) < 32:
            continue

        k = key.lower()
        if any(token in k for token in ["de_time", "fe_time", "ba_time", "time"]):
            channels.append(flat.astype(np.float32))

    # fallback: any 1D numeric array if no known key exists
    if not channels:
        for key, value in mat_data.items():
            if isinstance(value, np.ndarray):
                flat = value.reshape(-1)
                if flat.ndim == 1 and len(flat) >= 32:
                    channels.append(flat.astype(np.float32))

    return channels[:3]


def _parse_cwru(config: MultiSourceConfig) -> dict[str, np.ndarray]:
    if loadmat is None:
        return {
            "X_vibration": _empty((0, config.vibration_window, 3)),
            "bearing_label": _empty((0,)),
            "y_anomaly": _empty((0,)),
        }

    root = Path(config.cwru_dir)
    if not root.exists():
        return {
            "X_vibration": _empty((0, config.vibration_window, 3)),
            "bearing_label": _empty((0,)),
            "y_anomaly": _empty((0,)),
        }

    mat_files = sorted(root.rglob("*.mat"))
    if not mat_files:
        return {
            "X_vibration": _empty((0, config.vibration_window, 3)),
            "bearing_label": _empty((0,)),
            "y_anomaly": _empty((0,)),
        }

    windows = []
    labels = []

    for mat_path in mat_files:
        if len(windows) >= config.max_cwru_windows:
            break

        try:
            mat = loadmat(str(mat_path))
            channels = _extract_cwru_channels(mat)
            if not channels:
                continue

            # Ensure 3 channels by repeating available channels.
            while len(channels) < 3:
                channels.append(channels[-1])

            # Align by minimum length.
            min_len = min(len(ch) for ch in channels)
            if min_len < config.vibration_window:
                continue

            aligned = np.stack([ch[:min_len] for ch in channels[:3]], axis=1)
            aligned = (aligned - np.mean(aligned, axis=0, keepdims=True)) / (
                np.std(aligned, axis=0, keepdims=True) + 1e-6
            )

            hop = max(1, config.vibration_window // 2)
            for i in range(0, min_len - config.vibration_window + 1, hop):
                windows.append(aligned[i : i + config.vibration_window].astype(np.float32))
                name = mat_path.stem.lower()
                is_normal = "normal" in name or "baseline" in name
                if (not is_normal) and name.isdigit():
                    # Common CWRU baseline files are 97-100.
                    is_normal = int(name) in {97, 98, 99, 100}
                labels.append(0.0 if is_normal else 1.0)

                if len(windows) >= config.max_cwru_windows:
                    break
        except Exception:
            continue

    if not windows:
        return {
            "X_vibration": _empty((0, config.vibration_window, 3)),
            "bearing_label": _empty((0,)),
            "y_anomaly": _empty((0,)),
        }

    x = np.asarray(windows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    return {
        "X_vibration": x,
        "bearing_label": y,
        "y_anomaly": y,
    }


def _read_wav_float(path: str) -> np.ndarray | None:
    # Prefer scipy for broad WAV codec support (including WAVE_FORMAT_EXTENSIBLE).
    if wavfile is not None:
        try:
            _, audio = wavfile.read(path)
            audio = np.asarray(audio)

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if audio.dtype.kind in ("i", "u"):
                info = np.iinfo(audio.dtype)
                audio = audio.astype(np.float32)
                if info.min == 0:
                    mid = info.max / 2.0
                    audio = (audio - mid) / max(mid, 1.0)
                else:
                    scale = float(max(abs(info.min), abs(info.max)))
                    audio = audio / max(scale, 1.0)
            elif audio.dtype.kind == "f":
                audio = audio.astype(np.float32)
            else:
                return None

            if len(audio) == 0:
                return None

            return audio.astype(np.float32)
        except Exception:
            pass

    # Fallback to stdlib for basic PCM WAV files.
    try:
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sampwidth == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            return None

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        if len(audio) == 0:
            return None

        return audio.astype(np.float32)
    except Exception:
        return None


def _parse_mimii(config: MultiSourceConfig) -> dict[str, np.ndarray]:
    root = Path(config.mimii_dir)
    if not root.exists():
        return {
            "X_acoustic": _empty((0, config.acoustic_window, 1)),
            "y_anomaly": _empty((0,)),
        }

    wav_files = sorted(root.rglob("*.wav"))
    if not wav_files:
        return {
            "X_acoustic": _empty((0, config.acoustic_window, 1)),
            "y_anomaly": _empty((0,)),
        }

    segments = []
    labels = []

    for wav_path in wav_files:
        if len(segments) >= config.max_mimii_windows:
            break

        audio = _read_wav_float(str(wav_path))
        if audio is None or len(audio) < config.acoustic_window:
            continue

        hop = config.acoustic_window
        lower = str(wav_path).lower()
        is_abnormal = "abnormal" in lower or "anomaly" in lower

        for i in range(0, len(audio) - config.acoustic_window + 1, hop):
            chunk = audio[i : i + config.acoustic_window]
            chunk = chunk / (np.std(chunk) + 1e-6)
            segments.append(chunk.reshape(-1, 1).astype(np.float32))
            labels.append(1.0 if is_abnormal else 0.0)

            if len(segments) >= config.max_mimii_windows:
                break

    if not segments:
        return {
            "X_acoustic": _empty((0, config.acoustic_window, 1)),
            "y_anomaly": _empty((0,)),
        }

    return {
        "X_acoustic": np.asarray(segments, dtype=np.float32),
        "y_anomaly": np.asarray(labels, dtype=np.float32),
    }


def _parse_edgeiiot(config: MultiSourceConfig) -> dict[str, np.ndarray]:
    csv_path = _find_first_csv(config.edgeiiot_path, ["edge", "iiot", "iot", "attack", "intrusion"])
    if not csv_path:
        return {
            "X_electrical": _empty((0, config.electrical_window, config.electrical_features)),
            "y_anomaly": _empty((0,)),
        }

    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        return {
            "X_electrical": _empty((0, config.electrical_window, config.electrical_features)),
            "y_anomaly": _empty((0,)),
        }

    candidate_cols = _select_numeric_columns(
        df,
        exclude_keywords=["label", "target", "class", "attack", "anomaly", "id"],
        k=max(config.electrical_features, 8),
    )
    if not candidate_cols:
        return {
            "X_electrical": _empty((0, config.electrical_window, config.electrical_features)),
            "y_anomaly": _empty((0,)),
        }

    x = df[candidate_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    x = StandardScaler().fit_transform(x).astype(np.float32)
    x = _pad_features(x, config.electrical_features)

    label_col = None
    for cand in ["attack", "label", "class", "target", "anomaly"]:
        matches = [c for c in df.columns if cand in c.lower()]
        if matches:
            label_col = matches[0]
            break

    if label_col:
        raw = df[label_col]
        if raw.dtype == object:
            y_anom = (~raw.astype(str).str.lower().isin(["normal", "benign", "0", "false", "ok"])).astype(np.float32).values
        else:
            y_anom = pd.to_numeric(raw, errors="coerce").fillna(0.0).astype(np.float32).values
            y_anom = (y_anom > 0).astype(np.float32)
    else:
        y_anom = np.zeros(len(df), dtype=np.float32)

    Xw, starts = _sliding_windows(
        x,
        config.electrical_window,
        max_windows=config.max_edge_windows,
        return_starts=True,
    )
    if len(Xw) == 0:
        return {
            "X_electrical": _empty((0, config.electrical_window, config.electrical_features)),
            "y_anomaly": _empty((0,)),
        }

    idx = starts + (config.electrical_window - 1)
    y_anom_sel = y_anom[idx]

    return {
        "X_electrical": Xw.astype(np.float32),
        "y_anomaly": y_anom_sel.astype(np.float32),
    }


def _rng_indices(rng: np.random.Generator, n_source: int, n_target: int) -> np.ndarray:
    if n_source <= 0 or n_target <= 0:
        return np.zeros((0,), dtype=np.int64)
    return rng.integers(0, n_source, size=n_target, endpoint=False).astype(np.int64)


def build_multisource_dataset(config: MultiSourceConfig) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rng = np.random.default_rng(config.random_seed)

    ai4i = _parse_ai4i(config)
    metro = _parse_metropt3(config)
    cwru = _parse_cwru(config)
    mimii = _parse_mimii(config)
    edge = _parse_edgeiiot(config)

    process_parts = [x for x in [ai4i["X_process"], metro["X_process"]] if len(x) > 0]
    rul_parts = [x for x in [ai4i["y_rul"], metro["y_rul"]] if len(x) > 0]
    fault_parts = [x for x in [ai4i["y_faults"], metro["y_faults"]] if len(x) > 0]
    process_anom_parts = [x for x in [ai4i["y_anomaly"], metro["y_anomaly"]] if len(x) > 0]

    if process_parts:
        X_process_pool = np.concatenate(process_parts, axis=0)
        y_rul_pool = np.concatenate(rul_parts, axis=0)
        y_fault_pool = np.concatenate(fault_parts, axis=0)
        y_process_anom_pool = np.concatenate(process_anom_parts, axis=0)
    else:
        X_process_pool = _empty((0, config.process_window, config.process_features))
        y_rul_pool = _empty((0,))
        y_fault_pool = _empty((0, NUM_FAULT_CLASSES))
        y_process_anom_pool = _empty((0,))

    counts = {
        "process": int(len(X_process_pool)),
        "vibration": int(len(cwru["X_vibration"])),
        "acoustic": int(len(mimii["X_acoustic"])),
        "electrical": int(len(edge["X_electrical"])),
    }

    n_target = max(counts.values()) if counts else 0
    if config.max_target_samples > 0:
        n_target = min(n_target, config.max_target_samples)
    if n_target == 0:
        raise ValueError(
            "No usable samples found. Provide at least one dataset file/folder "
            "(ai4i2020, CWRU, MIMII, MetroPT-3, Edge-IIoTset)."
        )

    X_process = _empty((n_target, config.process_window, config.process_features))
    X_vibration = _empty((n_target, config.vibration_window, 3))
    X_acoustic = _empty((n_target, config.acoustic_window, 1))
    X_electrical = _empty((n_target, config.electrical_window, config.electrical_features))
    X_thermal = _empty((n_target, config.thermal_embedding_dim))

    y_rul = np.full((n_target,), 120.0, dtype=np.float32)
    y_faults = _empty((n_target, NUM_FAULT_CLASSES))
    y_anomaly = _empty((n_target,))

    idx_process = _rng_indices(rng, len(X_process_pool), n_target)
    idx_vibration = _rng_indices(rng, len(cwru["X_vibration"]), n_target)
    idx_acoustic = _rng_indices(rng, len(mimii["X_acoustic"]), n_target)
    idx_electrical = _rng_indices(rng, len(edge["X_electrical"]), n_target)

    for i in range(n_target):
        anomaly_votes = []

        if len(idx_process) > 0:
            p = idx_process[i]
            X_process[i] = X_process_pool[p]
            y_rul[i] = y_rul_pool[p]
            y_faults[i] = y_fault_pool[p]
            anomaly_votes.append(float(y_process_anom_pool[p]))

        if len(idx_vibration) > 0:
            v = idx_vibration[i]
            X_vibration[i] = cwru["X_vibration"][v]
            bearing = float(cwru["bearing_label"][v])
            y_faults[i, 5] = max(y_faults[i, 5], bearing)
            anomaly_votes.append(float(cwru["y_anomaly"][v]))

        if len(idx_acoustic) > 0:
            a = idx_acoustic[i]
            X_acoustic[i] = mimii["X_acoustic"][a]
            anomaly_votes.append(float(mimii["y_anomaly"][a]))

        if len(idx_electrical) > 0:
            e = idx_electrical[i]
            X_electrical[i] = edge["X_electrical"][e]
            anomaly_votes.append(float(edge["y_anomaly"][e]))

        X_thermal[i] = _thermal_embed_from_process(X_process[i], config.thermal_embedding_dim)
        y_anomaly[i] = 1.0 if any(v >= 0.5 for v in anomaly_votes) else 0.0

    report = {
        "num_samples": int(n_target),
        "fault_classes": list(FAULT_CLASSES),
        "sources": {
            "ai4i2020": int(len(ai4i["X_process"])),
            "metropt3": int(len(metro["X_process"])),
            "cwru": int(len(cwru["X_vibration"])),
            "mimii": int(len(mimii["X_acoustic"])),
            "edgeiiotset": int(len(edge["X_electrical"])),
        },
        "max_target_samples": int(config.max_target_samples),
        "anomaly_rate": float(np.mean(y_anomaly > 0.5)),
        "fault_positive_rate": {
            FAULT_CLASSES[i]: float(np.mean(y_faults[:, i] > 0.5))
            for i in range(NUM_FAULT_CLASSES)
        },
    }

    dataset = {
        "X_process": X_process.astype(np.float32),
        "X_vibration": X_vibration.astype(np.float32),
        "X_acoustic": X_acoustic.astype(np.float32),
        "X_electrical": X_electrical.astype(np.float32),
        "X_thermal": X_thermal.astype(np.float32),
        "y_rul": y_rul.astype(np.float32),
        "y_faults": y_faults.astype(np.float32),
        "y_anomaly": y_anomaly.astype(np.float32),
    }

    return dataset, report


def save_multisource_dataset(output_npz: str, dataset: dict[str, np.ndarray], report: dict[str, Any]):
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez_compressed(output_npz, **dataset)

    report_path = output_npz.replace(".npz", "_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        import json

        json.dump(report, f, indent=2)

    return report_path
