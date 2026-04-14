"""
hardware_bridge.py

Bridges physical or emulated sensor signals into MQTT async feature topics:
  sensors/{machine_id}/feature/{feature_name}

Modes:
- emulate (default): emits feature streams at different sample rates
- serial-jsonl: reads line-delimited JSON packets from a serial device

Expected serial JSON format per line:
{
  "feature": "sensor_measurement_2",
  "value": 642.7,
  "timestamp": 1712824000.123,
  "machine_id": "M1"
}
"""

import asyncio
import json
import logging
import os
import random
import time

import paho.mqtt.client as mqtt

from src.sensor_contract import RealSensorPacket, to_feature_updates


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [HARDWARE-BRIDGE] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MACHINE_ID = os.getenv("MACHINE_ID", "M1").strip().upper() or "M1"
BRIDGE_MODE = os.getenv("HARDWARE_BRIDGE_MODE", "emulate").strip().lower()
SERIAL_PORT = os.getenv("SERIAL_PORT", "")
SERIAL_BAUD = int(os.getenv("SERIAL_BAUD", "115200"))

# Example heterogeneous rates (Hz): fast, medium, slow sensors.
FEATURE_RATES_HZ = {
    "sensor_measurement_2": 1.0,
    "sensor_measurement_3": 1.0,
    "sensor_measurement_4": 5.0,
    "sensor_measurement_7": 20.0,
    "sensor_measurement_8": 20.0,
    "sensor_measurement_9": 10.0,
    "sensor_measurement_11": 50.0,
    "sensor_measurement_12": 50.0,
    "sensor_measurement_13": 50.0,
    "sensor_measurement_14": 10.0,
    "sensor_measurement_15": 10.0,
    "sensor_measurement_17": 10.0,
    "sensor_measurement_20": 5.0,
    "sensor_measurement_21": 5.0,
}

FEATURE_BOUNDS = {
    "sensor_measurement_2": (641.2, 644.6),
    "sensor_measurement_3": (1571.0, 1617.0),
    "sensor_measurement_4": (1382.0, 1442.0),
    "sensor_measurement_7": (549.0, 557.0),
    "sensor_measurement_8": (2387.7, 2388.8),
    "sensor_measurement_9": (9020.0, 9250.0),
    "sensor_measurement_11": (45.0, 50.0),
    "sensor_measurement_12": (518.0, 523.0),
    "sensor_measurement_13": (2385.0, 2395.0),
    "sensor_measurement_14": (8100.0, 8260.0),
    "sensor_measurement_15": (8.2, 8.7),
    "sensor_measurement_17": (389.0, 401.0),
    "sensor_measurement_20": (38.0, 40.0),
    "sensor_measurement_21": (23.0, 24.0),
}


class HardwareBridge:
    def __init__(self):
        self.client = mqtt.Client(client_id=f"hardware-bridge-{MACHINE_ID}")
        self.current_values = {
            name: random.uniform(bounds[0], bounds[1])
            for name, bounds in FEATURE_BOUNDS.items()
        }

    def connect(self):
        while True:
            try:
                self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
                self.client.loop_start()
                logger.info(f"Connected to {MQTT_BROKER}:{MQTT_PORT}")
                return
            except Exception as exc:
                logger.error(f"MQTT connect failed ({exc}); retrying in 3s")
                time.sleep(3)

    def publish_feature(self, feature_name, value, timestamp=None, machine_id=MACHINE_ID):
        payload = {
            "machine_id": machine_id,
            "feature": feature_name,
            "value": float(value),
            "timestamp": float(timestamp if timestamp is not None else time.time()),
        }
        topic = f"sensors/{machine_id}/feature/{feature_name}"
        self.client.publish(topic, json.dumps(payload), qos=1)

    def _next_emulated_value(self, feature_name):
        lo, hi = FEATURE_BOUNDS[feature_name]
        current = self.current_values[feature_name]
        span = hi - lo
        step = random.uniform(-0.01 * span, 0.01 * span)
        next_value = max(lo, min(hi, current + step))
        self.current_values[feature_name] = next_value
        return next_value

    async def run_emulator(self):
        async def feature_task(feature_name, hz):
            period = 1.0 / max(hz, 0.1)
            while True:
                value = self._next_emulated_value(feature_name)
                self.publish_feature(feature_name, value)
                await asyncio.sleep(period)

        tasks = [
            asyncio.create_task(feature_task(name, hz))
            for name, hz in FEATURE_RATES_HZ.items()
        ]
        logger.info("Running async hardware emulator (heterogeneous sample rates)")
        await asyncio.gather(*tasks)

    async def run_serial_jsonl(self):
        if not SERIAL_PORT:
            raise RuntimeError("SERIAL_PORT is required for serial-jsonl mode")

        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for serial-jsonl mode. Install with: pip install pyserial"
            ) from exc

        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        logger.info(f"Reading serial stream from {SERIAL_PORT} @ {SERIAL_BAUD}")

        while True:
            line = await asyncio.to_thread(ser.readline)
            if not line:
                await asyncio.sleep(0.01)
                continue

            try:
                packet = json.loads(line.decode(errors="ignore").strip())
                canonical_packet = RealSensorPacket(**packet)
                updates = to_feature_updates(canonical_packet)

                for update in updates:
                    feature = str(update["feature"]).strip().lower()
                    value = float(update["value"])
                    timestamp = update.get("timestamp")
                    machine_id = str(update["machine_id"]).strip().upper() or MACHINE_ID

                    if feature not in FEATURE_RATES_HZ:
                        continue

                    self.publish_feature(feature, value, timestamp=timestamp, machine_id=machine_id)
            except Exception:
                continue


async def main():
    bridge = HardwareBridge()
    bridge.connect()

    if BRIDGE_MODE == "serial-jsonl":
        await bridge.run_serial_jsonl()
    else:
        await bridge.run_emulator()


if __name__ == "__main__":
    asyncio.run(main())
