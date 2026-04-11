"""
sanitize.py — input sanitization helpers.

Validation (Pydantic) answers: "is this value acceptable?"
Sanitization (this file) answers: "how do we safely handle it?"

These run AFTER Pydantic validation passes.
"""

import re
import html


def sanitize_string(value: str, max_length: int = 255) -> str:
    """
    General-purpose string sanitizer:
      1. Strip leading/trailing whitespace
      2. Collapse internal whitespace runs to a single space
      3. HTML-escape special characters (<, >, &, ", ')
      4. Hard-truncate to max_length
    """
    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    value = html.escape(value)
    return value[:max_length]


def sanitize_machine_id(machine_id: str) -> str:
    """
    Normalise machine ID: uppercase, strip non-alphanumeric chars.
    'M 1 ' → 'M1', 'm01' → 'M01'
    """
    cleaned = re.sub(r"[^A-Za-z0-9]", "", machine_id)
    return cleaned.upper()


def sanitize_mode(mode: str) -> str:
    """
    Normalise mode string. Rejects anything not in the allowed set.
    Raises ValueError so FastAPI surfaces a clean 400.
    """
    allowed = {"normal", "degrade", "failure"}
    cleaned = mode.strip().lower()
    if cleaned not in allowed:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {allowed}")
    return cleaned


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a numeric value to [lo, hi]. Silently fixes out-of-range values."""
    return max(lo, min(hi, value))


def sanitize_sensor_dict(data: dict) -> dict:
    """
    Final sanitization pass on a sensor data dict before inserting into DB.
    Clamps numeric fields to safe ranges and sanitizes string fields.

    Use this as the last step before insert_data() to catch anything
    that slipped through validation (e.g. values injected by the model itself).
    """
    raw_stage_probs = data.get("stage_probs", [0.0, 0.0, 0.0])
    if not isinstance(raw_stage_probs, (list, tuple)):
        raw_stage_probs = [0.0, 0.0, 0.0]

    stage_probs = [float(clamp(p, 0.0, 1.0)) for p in list(raw_stage_probs)[:3]]
    while len(stage_probs) < 3:
        stage_probs.append(0.0)

    total = sum(stage_probs)
    if total > 0:
        stage_probs = [p / total for p in stage_probs]

    return {
        "machine_id":      sanitize_machine_id(str(data.get("machine_id", "M1"))),
        "step":            int(clamp(data.get("step", 0), 0, 1_000_000)),
        "RUL":             clamp(data.get("RUL", 0.0), 0.0, 10_000.0),
        "RUL_std":         clamp(data.get("RUL_std", 0.0), 0.0, 10_000.0),
        "stage_probs":     [round(p, 3) for p in stage_probs],
        "status":          sanitize_string(str(data.get("status", "UNKNOWN")), max_length=16),
        "temperature":     clamp(data.get("temperature", 0.0),     250.0, 400.0),
        "air_temperature": clamp(data.get("air_temperature", 0.0), 250.0, 400.0),
        "torque":          clamp(data.get("torque", 0.0),          0.0,   100.0),
        "tool_wear":       clamp(data.get("tool_wear", 0.0),       0.0,   300.0),
        "speed":           clamp(data.get("speed", 0.0),           0.0,   3000.0),
    }