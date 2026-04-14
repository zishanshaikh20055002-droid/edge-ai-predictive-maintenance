import numpy as np
import logging
import json
import os
from collections import defaultdict
from collections import deque
import time

from src.sensor_contract import CANONICAL_FEATURE_ALIASES

logger = logging.getLogger(__name__)

class HardwareAgnosticBuffer:
    def __init__(self, window_size=30, num_features=14):
        self.window_size = window_size
        self.num_features = num_features
        self.raw_streams = defaultdict(list)
        self.last_known_good = defaultdict(lambda: np.zeros(num_features))

    def process_payload(self, machine_id, step, features):
        try:
            parsed_features = [float(x) for x in features]
        except (TypeError, ValueError):
            parsed_features = []

        if not parsed_features or len(parsed_features) != self.num_features:
            clean_features = self.last_known_good[machine_id].copy()
        else:
            clean_features = np.clip(parsed_features, a_min=-1e6, a_max=1e6).copy()
            self.last_known_good[machine_id] = clean_features.copy()

        self.raw_streams[machine_id].append({
            'step': int(step), 
            'features': clean_features.copy()
        })
        
        if len(self.raw_streams[machine_id]) > self.window_size:
            self.raw_streams[machine_id].pop(0)

    def get_valid_window(self, machine_id):
        stream = self.raw_streams[machine_id]
        
        if len(stream) < self.window_size:
            return None
            
        ordered_packets = sorted(stream, key=lambda x: x['step'])
        
        window = np.array([item['features'] for item in ordered_packets], dtype=np.float32)
        window = np.nan_to_num(window, nan=0.0)
        
        return window.reshape(1, self.window_size, self.num_features)


class AsyncSensorFusionBuffer:
    """
    Buffers unsynchronized sensor topics and emits fixed-rate fused snapshots.

    This supports real hardware streams where each feature arrives at a different
    sampling frequency (for example vibration at kHz and temperature at Hz).
    """

    def __init__(
        self,
        feature_names,
        window_size=30,
        target_hz=1.0,
        max_buffer_seconds=120.0,
    ):
        self.feature_names = [str(name).strip().lower() for name in feature_names]
        self.window_size = int(window_size)
        self.target_hz = float(target_hz) if target_hz and target_hz > 0 else 1.0
        self.max_buffer_seconds = float(max_buffer_seconds)

        self.num_features = len(self.feature_names)
        self.feature_index = {name: i for i, name in enumerate(self.feature_names)}

        self.alias_map = {}
        for name in self.feature_names:
            if name.startswith("sensor_measurement_"):
                suffix = name.split("sensor_measurement_")[-1]
                self.alias_map[f"s{suffix}"] = name

        # Built-in aliases for future real sensors (rpm, torque_nm, etc.).
        for alias, target in CANONICAL_FEATURE_ALIASES.items():
            if target in self.feature_index:
                self.alias_map[str(alias).strip().lower()] = target

        # Optional runtime alias injection via JSON env var.
        # Example:
        # SENSOR_ALIAS_JSON={"process_temp_k":"sensor_measurement_2"}
        raw_aliases = os.getenv("SENSOR_ALIAS_JSON", "").strip()
        if raw_aliases:
            try:
                parsed = json.loads(raw_aliases)
                if isinstance(parsed, dict):
                    for alias, target in parsed.items():
                        norm_alias = str(alias).strip().lower().replace(" ", "_").replace("-", "_")
                        norm_target = str(target).strip().lower().replace(" ", "_").replace("-", "_")
                        if norm_target in self.feature_index:
                            self.alias_map[norm_alias] = norm_target
            except Exception:
                pass

        self.series = defaultdict(lambda: defaultdict(deque))
        self.snapshots = defaultdict(list)
        self.last_known_good = defaultdict(lambda: np.zeros(self.num_features, dtype=np.float32))
        self.next_emit_ts = {}
        self.step_counter = defaultdict(int)

    def _canonical_feature_name(self, feature_name):
        normalized = str(feature_name).strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in self.feature_index:
            return normalized

        mapped = self.alias_map.get(normalized)
        if mapped:
            return mapped

        # Allow modality prefixes like "thermal.process_temp_k".
        if "." in normalized:
            tail = normalized.split(".")[-1]
            if tail in self.feature_index:
                return tail
            return self.alias_map.get(tail)

        return None

    def process_feature(self, machine_id, feature_name, value, timestamp=None):
        canonical = self._canonical_feature_name(feature_name)
        if canonical is None:
            return False

        try:
            clean_value = float(value)
        except (TypeError, ValueError):
            return False

        if timestamp is None:
            ts = time.time()
        else:
            try:
                ts = float(timestamp)
            except (TypeError, ValueError):
                ts = time.time()

        clean_value = float(np.clip(clean_value, a_min=-1e6, a_max=1e6))

        feature_series = self.series[machine_id][canonical]
        feature_series.append((ts, clean_value))
        self._prune_old(machine_id, ts)
        self._emit_snapshots(machine_id, ts)
        return True

    def _prune_old(self, machine_id, now_ts):
        min_ts = now_ts - self.max_buffer_seconds
        for feature_name in self.feature_names:
            dq = self.series[machine_id][feature_name]
            while dq and dq[0][0] < min_ts:
                dq.popleft()

    def _value_at_time(self, machine_id, feature_name, ts):
        idx = self.feature_index[feature_name]
        last_good = float(self.last_known_good[machine_id][idx])
        dq = self.series[machine_id][feature_name]

        if not dq:
            return last_good

        chosen = None
        for sample_ts, sample_val in reversed(dq):
            if sample_ts <= ts:
                chosen = sample_val
                break

        if chosen is None:
            chosen = dq[0][1]

        return float(chosen)

    def _snapshot_at(self, machine_id, ts):
        vector = np.zeros(self.num_features, dtype=np.float32)
        for feature_name in self.feature_names:
            idx = self.feature_index[feature_name]
            vector[idx] = self._value_at_time(machine_id, feature_name, ts)

        vector = np.nan_to_num(vector, nan=0.0)
        self.last_known_good[machine_id] = vector.copy()
        return vector

    def _emit_snapshots(self, machine_id, up_to_ts):
        period = 1.0 / self.target_hz
        next_ts = self.next_emit_ts.get(machine_id)

        if next_ts is None:
            next_ts = up_to_ts
            self.next_emit_ts[machine_id] = next_ts

        while next_ts <= up_to_ts:
            vector = self._snapshot_at(machine_id, next_ts)
            self.step_counter[machine_id] += 1
            step = self.step_counter[machine_id]

            self.snapshots[machine_id].append({
                "step": step,
                "timestamp": next_ts,
                "features": vector,
            })

            if len(self.snapshots[machine_id]) > self.window_size:
                self.snapshots[machine_id].pop(0)

            next_ts += period

        self.next_emit_ts[machine_id] = next_ts

    def get_latest_step(self, machine_id):
        return int(self.step_counter[machine_id])

    def get_valid_window(self, machine_id):
        stream = self.snapshots[machine_id]
        if len(stream) < self.window_size:
            return None

        window = np.array([row["features"] for row in stream], dtype=np.float32)
        window = np.nan_to_num(window, nan=0.0)
        return window.reshape(1, self.window_size, self.num_features)