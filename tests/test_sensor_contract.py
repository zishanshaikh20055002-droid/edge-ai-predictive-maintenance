import pytest
from pydantic import ValidationError

from src.sensor_contract import (
    RealSensorPacket,
    canonicalize_feature_name,
    to_feature_updates,
)


def test_machine_id_is_sanitized_to_uppercase():
    packet = RealSensorPacket(machine_id="  m1  ", feature="torque", value=12.3)
    assert packet.machine_id == "M1"


def test_packet_requires_single_or_multi_payload():
    with pytest.raises(ValidationError):
        RealSensorPacket(machine_id="M1")


def test_canonicalize_feature_name_maps_aliases():
    assert canonicalize_feature_name("torque") == "sensor_measurement_4"
    assert canonicalize_feature_name("Air Temperature") == "sensor_measurement_3"


def test_to_feature_updates_from_single_feature():
    packet = RealSensorPacket(machine_id="M1", feature="torque", value=40.0, timestamp=1.23)
    updates = to_feature_updates(packet)

    assert len(updates) == 1
    assert updates[0]["machine_id"] == "M1"
    assert updates[0]["feature"] == "sensor_measurement_4"
    assert updates[0]["value"] == 40.0


def test_to_feature_updates_from_values_map():
    packet = RealSensorPacket(
        machine_id="M2",
        values={"torque": 50.0, "speed rpm": 1200.0},
        timestamp=2.0,
    )
    updates = to_feature_updates(packet)

    features = {u["feature"] for u in updates}
    assert len(updates) == 2
    assert "sensor_measurement_4" in features
    assert "sensor_measurement_8" in features
