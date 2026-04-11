import time
import json
import os
import logging
import pandas as pd
import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PUBLISHER] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MACHINE_ID = os.getenv("MACHINE_ID", "M1").strip().upper() or "M1"

# Using the complete run-to-failure training file so we have plenty of data
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "train_FD001.txt")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


PUBLISH_RATE = max(0.05, _env_float("PUBLISH_RATE", 1.0))
PUBLISH_START_STEP = max(0, _env_int("PUBLISH_START_STEP", 100))


def _machine_id_to_unit(machine_id: str) -> int:
    digits = "".join(ch for ch in machine_id if ch.isdigit())
    return int(digits) if digits else 1

def start_publishing():
    unit_number = _machine_id_to_unit(MACHINE_ID)
    topic = f"sensors/{MACHINE_ID}/data"
    client = mqtt.Client(client_id=f"publisher-{MACHINE_ID}")
    
    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            logger.info(f"Connected to {MQTT_BROKER}:{MQTT_PORT}")
            break
        except Exception:
            logger.error(f"Connection failed. Retrying in 3s...")
            time.sleep(3)

    client.loop_start()
    logger.info("Loading NASA CMAPSS dataset...")
    
    columns = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
              [f'sensor_measurement_{i}' for i in range(1, 22)]
    df = pd.read_csv(DATA_PATH, sep=r'\s+', names=columns)
    
    # Filter to selected machine/unit
    df_m1 = df[df['unit_number'] == unit_number].copy()
    if df_m1.empty:
        logger.warning(f"Unit {unit_number} not found in dataset. Falling back to unit 1.")
        df_m1 = df[df['unit_number'] == 1].copy()
        unit_number = 1
    
    # 14 features used in the CMAPSS model
    features = [f'sensor_measurement_{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
    
    # Fast forward for faster warm-up to degradation phase
    start_index = min(PUBLISH_START_STEP, max(0, len(df_m1) - 1))
    logger.info(f"Fast-forwarding to step {start_index} to observe active degradation...")
    
    # Convert to a Python list of dictionaries first to completely bypass Pandas index issues
    records = df_m1[features].to_dict('records')
    
    # Slice the clean Python list
    sliced_records = records[start_index:]
    
    step = start_index
    for row in sliced_records:
        # Extract the values from the dictionary in the correct order
        feature_values = [row[feat] for feat in features]
        
        payload = {
            "machine_id": f"M{unit_number}",
            "step": step,
            "features": feature_values
        }
        
        client.publish(topic, json.dumps(payload), qos=1)
        logger.info(f"[REPLAY] step={step} sent 14 sensor features (CMAPSS)")
        
        step += 1
        time.sleep(PUBLISH_RATE)

if __name__ == "__main__":
    start_publishing()