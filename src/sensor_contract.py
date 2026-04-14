"""
sensor_contract.py

Canonical contract for future real-sensor integrations.

This module standardizes heterogeneous sensor packets from PLCs, gateways,
DAQ devices, and edge agents into canonical feature updates compatible with the
existing MQTT async topic format:

  sensors/{machine_id}/feature/{feature_name}
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# Common aliases expected from real deployments; values map to current CMAPSS
# channels used by the runtime inference buffer.
CANONICAL_FEATURE_ALIASES = {
    "temp_process_k": "sensor_measurement_2",
    "process_temperature": "sensor_measurement_2",
    "temperature_process": "sensor_measurement_2",
    "temp_air_k": "sensor_measurement_3",
    "air_temperature": "sensor_measurement_3",
    "torque_nm": "sensor_measurement_4",
    "torque": "sensor_measurement_4",
    "tool_wear_min": "sensor_measurement_7",
    "tool_wear": "sensor_measurement_7",
    "speed_rpm": "sensor_measurement_8",
    "rpm": "sensor_measurement_8",
    "vibration_rms": "sensor_measurement_15",
    "vibration": "sensor_measurement_15",
    "voltage_v": "sensor_measurement_13",
    "voltage": "sensor_measurement_13",
    "current_a": "sensor_measurement_11",
    "current": "sensor_measurement_11",
    "power_kw": "sensor_measurement_20",
    "acoustic_rms": "sensor_measurement_21",
}


class RealSensorPacket(BaseModel):
    """
    Device-agnostic packet for single-value or multi-value emission.

    One of the following payload styles must be provided:
    - Single feature: feature + value
    - Multi feature: values dict
    """

    machine_id: str = Field(min_length=1, max_length=32)
    timestamp: float | None = None
    modality: Literal[
        "process",
        "vibration",
        "acoustic",
        "electrical",
        "thermal",
        "network",
        "mixed",
    ] = "mixed"

    feature: str | None = None
    value: float | None = None
    values: dict[str, float] | None = None

    @field_validator("machine_id")
    @classmethod
    def sanitize_machine_id(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("machine_id is required")
        return cleaned

    @model_validator(mode="after")
    def validate_payload_shape(self) -> "RealSensorPacket":
        has_single = self.feature is not None and self.value is not None
        has_multi = bool(self.values)

        if not (has_single or has_multi):
            raise ValueError("Provide either feature+value or values dict")
        return self


def canonicalize_feature_name(name: str) -> str:
    normalized = str(name).strip().lower().replace(" ", "_")
    return CANONICAL_FEATURE_ALIASES.get(normalized, normalized)


def to_feature_updates(packet: RealSensorPacket) -> list[dict[str, float | str | None]]:
    """
    Convert a RealSensorPacket into canonical feature updates.

    Returns:
      [{"machine_id": ..., "feature": ..., "value": ..., "timestamp": ...}, ...]
    """
    updates: list[dict[str, float | str | None]] = []

    if packet.feature is not None and packet.value is not None:
        updates.append(
            {
                "machine_id": packet.machine_id,
                "feature": canonicalize_feature_name(packet.feature),
                "value": float(packet.value),
                "timestamp": packet.timestamp,
            }
        )

    if packet.values:
        for key, value in packet.values.items():
            updates.append(
                {
                    "machine_id": packet.machine_id,
                    "feature": canonicalize_feature_name(key),
                    "value": float(value),
                    "timestamp": packet.timestamp,
                }
            )

    return updates
