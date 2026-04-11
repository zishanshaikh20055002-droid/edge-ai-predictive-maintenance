import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "machine_data.db")
DB_TIMEOUT_SECONDS = 10


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT_SECONDS)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _connect() as conn:
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            machine_id TEXT,
            RUL REAL,
            status TEXT,
            temperature REAL,
            air_temperature REAL,
            torque REAL,
            tool_wear REAL,
            speed REAL
        )
        """)

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_machine_ts ON predictions(machine_id, timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(timestamp)"
        )

        conn.commit()

def insert_data(data):
    with _connect() as conn:
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO predictions (
            machine_id, RUL, status,
            temperature, air_temperature,
            torque, tool_wear, speed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("machine_id", "M1"),
            data.get("RUL", 0.0),
            data.get("status", "UNKNOWN"),
            data.get("temperature", 0.0),
            data.get("air_temperature", 0.0),
            data.get("torque", 0.0),
            data.get("tool_wear", 0.0),
            data.get("speed", 0.0),
        ))

        conn.commit()