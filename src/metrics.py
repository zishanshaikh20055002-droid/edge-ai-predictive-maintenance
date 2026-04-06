"""
metrics.py — custom Prometheus metrics for the predictive maintenance system.

prometheus-fastapi-instrumentator handles generic HTTP metrics automatically
(request count, latency, status codes). This file adds domain-specific metrics
that are actually useful for an industrial ML system.
"""

from prometheus_client import Gauge, Counter, Histogram, Info

# ── Machine health metrics ────────────────────────────────────

rul_gauge = Gauge(
    "machine_rul",
    "Current Remaining Useful Life prediction",
    ["machine_id"],
)

health_status_counter = Counter(
    "machine_health_status_total",
    "Total number of predictions per health status",
    ["machine_id", "status"],
)

# ── Inference metrics ─────────────────────────────────────────

inference_latency = Histogram(
    "ml_inference_latency_seconds",
    "Time taken for TFLite model inference",
    ["machine_id"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# ── WebSocket metrics ─────────────────────────────────────────

ws_active_connections = Gauge(
    "websocket_active_connections",
    "Number of currently active WebSocket connections",
)

ws_messages_sent = Counter(
    "websocket_messages_sent_total",
    "Total WebSocket messages sent to clients",
    ["machine_id"],
)

# ── Simulation mode ───────────────────────────────────────────

simulation_mode_info = Info(
    "simulation_mode",
    "Current simulation mode",
)

damage_level_gauge = Gauge(
    "simulation_damage_level",
    "Current simulated damage level (0–100)",
)

# ── Sensor gauges ─────────────────────────────────────────────

sensor_temperature = Gauge(
    "sensor_process_temperature_kelvin",
    "Process temperature in Kelvin",
    ["machine_id"],
)

sensor_torque = Gauge(
    "sensor_torque_nm",
    "Torque in Nm",
    ["machine_id"],
)

sensor_tool_wear = Gauge(
    "sensor_tool_wear_minutes",
    "Tool wear in minutes",
    ["machine_id"],
)

sensor_speed = Gauge(
    "sensor_rotational_speed_rpm",
    "Rotational speed in RPM",
    ["machine_id"],
)