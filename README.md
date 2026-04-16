Edge AI Predictive Maintenance System

Documentation Hub

- Defense master guide: docs/PROJECT_DEFENSE_MASTER.md
- Idea pitch scripts: docs/IDEA_PITCH_SCRIPTS.md
- Teacher Q and A bank: docs/TEACHER_QA_BANK.md
- Timeline and major changes: docs/PROJECT_TIMELINE_AND_CHANGES.md
- Evaluation and accuracy guide: docs/EVALUATION_AND_ACCURACY.md
- System diagrams: docs/SYSTEM_DIAGRAMS.md
- Slide-ready diagrams: docs/SLIDE_DIAGRAMS.md
- Training deep dive: TRAINING_GUIDE.md

1. Introduction

This project presents a real-time predictive maintenance system designed to monitor machine health, estimate Remaining Useful Life (RUL), and provide early warnings of potential failures.

The system integrates machine learning, real-time data streaming, secure backend services, and observability tools to simulate and support industrial-grade monitoring environments.

It is architected as a modular and extensible platform, enabling seamless transition from simulated datasets to real-world sensor inputs and industrial deployment scenarios.


---

2. Problem Statement

Industrial machines such as motors and rotating equipment are prone to unexpected failures, leading to:

Production downtime

Increased maintenance costs

Equipment damage

Safety risks


Traditional monitoring systems (e.g., PLC-based systems) are typically reactive, triggering alerts only after thresholds are exceeded.

This project aims to provide a predictive approach, identifying degradation patterns before critical failure occurs.


---

3. Objectives

Develop a real-time system for machine health monitoring

Predict Remaining Useful Life (RUL) using machine learning

Enable early detection of abnormal behavior

Provide secure and controlled system access

Ensure system reliability through monitoring and observability

Support both simulated and real sensor data



---

4. Key Features

4.1 Real-Time Data Streaming

Continuous data flow using WebSocket communication

Live updates without repeated HTTP requests

Low-latency system behavior


4.2 Predictive Analytics
# PulseGuard Edge AI

Predict failures early. Act with confidence.

PulseGuard Edge AI is a real-time predictive maintenance platform that ingests machine telemetry, estimates remaining useful life, localizes likely fault components, assigns alarm priority, and exposes the results through secure APIs, live dashboards, and observability tooling.

It is built to work in two modes:
- simulation mode for repeatable demos and evaluation
- hardware mode for PLC/gateway-connected industrial data

## Why this project matters

Industrial machines usually fail after degradation has already started. Traditional monitoring reacts late, often after a threshold is crossed. PulseGuard Edge AI moves the workflow earlier by combining machine learning, rule-based policy, and operator-facing diagnostics in one system.

The result is not just a prediction model. It is a complete decision-support platform for maintenance planning.

## What we built

- real-time telemetry streaming with MQTT and WebSocket updates
- RUL prediction and health stage classification
- fault localization with component-level diagnosis
- failure probability, time-to-failure, and maintenance priority outputs
- alarm policy with explainable reasons and recommended action windows
- secure FastAPI backend with JWT authentication, RBAC, and rate limiting
- SQLite persistence for diagnostics and feedback loops
- Prometheus metrics and Grafana dashboards for observability
- multisource training pipeline with imbalance handling and domain adaptation hooks
- PLC bridge design for Modbus, OPC UA, MQTT, and file-based polling

## Core features

### 1. Real-time monitoring

The backend streams machine state in real time instead of requiring repeated polling. This makes the system suitable for live operations and low-latency dashboards.

### 2. Predictive analytics

The ML layer produces:
- remaining useful life
- healthy / warning / critical stage
- failure probability
- probable fault component
- fault type and severity
- confidence and maintenance window

### 3. Rich diagnosis output

The system now streams a full diagnostic record for each machine, including:
- electrical telemetry: voltage, current, power
- mechanical telemetry: speed, torque, vibration, tool wear
- thermal telemetry: process and air temperatures
- health index and time-to-failure
- probable causes and recommended actions

### 4. Fault localization model

When a supervised fault-localizer artifact is available, the runtime uses ML-based component probabilities. If not, it falls back to rule-based localization so the system remains usable.

### 5. Alarm policy

Alarm level is not the same as health stage. Health stage describes condition, while alarm level reflects urgency. The policy uses failure probability, severity, confidence, and time-to-failure to assign:
- INFO
- ADVISORY
- ALERT
- EMERGENCY

### 6. Fleet and feedback loops

The platform supports fleet-level monitoring, human relabel feedback, and scheduled retraining with drift detection. That makes the project more realistic than a one-shot ML demo.

## System architecture

```text
Sensors / Simulation / PLC Bridge
        |
        v
MQTT Broker
        |
        v
FastAPI Subscriber + Preprocessing
        |
        v
ML Inference and Diagnostics
        |
        +--> SQLite persistence
        +--> WebSocket live updates
        +--> REST diagnosis APIs
        +--> Alarm policy engine
        |
        v
Dashboard + Prometheus + Grafana
```

## Technology stack

- Backend: Python, FastAPI, TensorFlow Lite, WebSockets
- Frontend: HTML, CSS, JavaScript
- Data storage: SQLite
- Messaging: MQTT
- Security: JWT, RBAC, rate limiting
- Monitoring: Prometheus, Grafana
- Deployment: Docker, Docker Compose

## Machine learning pipeline

### Training data

The system supports multisource training and fusion from:
- AI4I 2020
- CWRU bearing data
- MIMII acoustic data
- MetroPT-3 compressor telemetry
- Edge-IIoTset network data

### Training capabilities

- multisource dataset builder
- class imbalance handling with effective-number weighting
- focal-loss training for rare classes
- weighted bootstrap resampling
- multimodal fusion across process, vibration, acoustic, electrical, and thermal inputs
- Monte Carlo Dropout support for uncertainty estimation
- dataset verification CLI before training
- evaluation metrics for RUL, faults, anomaly, and uncertainty

### Evaluation approach

There is no single accuracy number for the whole project. We evaluate task-wise:
- RUL: MAE, RMSE, MAPE, R²
- faults: precision, recall, F1, ROC-AUC, hamming loss
- anomaly: ROC-AUC, PR-AUC, thresholded F1
- uncertainty: coverage, interval width, miscalibration

That is the correct way to evaluate a multi-task maintenance system.

## Project documentation

- Defense master guide: [docs/PROJECT_DEFENSE_MASTER.md](docs/PROJECT_DEFENSE_MASTER.md)
- Idea pitch scripts: [docs/IDEA_PITCH_SCRIPTS.md](docs/IDEA_PITCH_SCRIPTS.md)
- Teacher Q and A bank: [docs/TEACHER_QA_BANK.md](docs/TEACHER_QA_BANK.md)
- Timeline and major changes: [docs/PROJECT_TIMELINE_AND_CHANGES.md](docs/PROJECT_TIMELINE_AND_CHANGES.md)
- Evaluation and accuracy guide: [docs/EVALUATION_AND_ACCURACY.md](docs/EVALUATION_AND_ACCURACY.md)
- Training guide: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

## Repository structure

```text
app.py                     FastAPI application and routes
dashboard.html             Live web dashboard
src/                       Runtime, ML, retraining, and integration code
data/                      Training datasets and generated arrays
models/                    Saved models and metadata
grafana/                   Dashboard provisioning
prometheus.yml             Metrics scraping config
docker-compose.yml         Local stack orchestration
```

## How to run locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the stack

```bash
docker compose up --build
```

### 3. Open the system

- API: `http://localhost:8000`
- Dashboard: open `dashboard.html`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Useful commands

### Verify datasets

```bash
python -m src.verify_dataset --npz data/multisource_train.npz --verbose
```

### Train the multimodal model

```bash
python -m src.train_multimodal_mtl --rebuild-dataset --epochs 40 --batch-size 128
```

### Evaluate metrics

```python
from src.evaluation_metrics import compute_all_metrics
```

### Load the PLC bridge

```python
from src.plc_bridge import PLCConfig, create_plc_connection, PLCStreamingBuffer
```

## What changed in recent work

- hardened TFLite runtime scaling and replay defaults
- improved dashboard diagnostics and score visualizations
- added multisource dataset fusion and imbalance utilities
- added evaluation metrics and dataset verification tooling
- added domain adaptation and PLC bridge modules
- created defense-ready documentation for teachers and viva prep

## Honest limitations

This is a strong end-to-end platform, but the last step toward real deployment still requires plant-specific validation:
- map the PLC registers or OPC UA nodes to the actual machine signals
- calibrate thresholds on real site data
- verify cross-domain adaptation on the target environment
- confirm latency, reliability, and alarm policy against plant requirements

## Closing summary

PulseGuard Edge AI is a full predictive maintenance platform, not just a model. It combines streaming ingestion, intelligent diagnosis, operator-friendly dashboards, secure APIs, observability, and a path to real hardware integration. That is the strongest way to present the project on GitHub and in front of teachers.

11. Data Handling

11.1 Simulation Mode

Uses dataset (X.npy)

Ideal for testing and development


11.2 Hardware Mode (Planned)

Accepts real sensor input

Enables real-world deployment



---

12. Database Design

The system stores:

Machine ID

Timestamp

RUL prediction

Status classification

Sensor values


Purpose:

Historical analysis

Debugging

Trend tracking



---

13. Security Considerations

JWT-based authentication ensures stateless security

Rate limiting prevents abuse

Role-based access restricts critical operations

Input validation prevents malformed requests



---

14. Deployment

14.1 Docker

Containerized backend

Simplified deployment across environments


14.2 Execution Steps

git clone <repository>
cd project

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

uvicorn app:app --reload

Frontend:

python -m http.server 5500


14.3 Runtime Tuning (Optional)

MC_PASSES: Number of Monte Carlo inference passes in fallback mode.

MC_NOISE_STD: Noise scale for stochastic inference in fallback mode.

PUBLISH_RATE: Delay (seconds) between MQTT replay messages.

PUBLISH_START_STEP: Initial replay index for CMAPSS stream.

MACHINE_ID: Publisher machine/topic identifier (e.g., M1, M2).


14.4 Real Hardware Sensor Ingestion

The backend now accepts asynchronous per-sensor streams in addition to array payloads.

Legacy topic (synchronized batch):

sensors/{machine_id}/data

Payload:

{"machine_id":"M1","step":123,"features":[...14 values...]}

Asynchronous topic (real hardware style):

sensors/{machine_id}/feature/{feature_name}

Payload:

{"value":642.7,"timestamp":1712824000.123}

Notes:

The subscriber resamples asynchronous streams to a fixed model cadence before inference.

Supported feature names are CMAPSS feature channels:

sensor_measurement_2, sensor_measurement_3, sensor_measurement_4, sensor_measurement_7,
sensor_measurement_8, sensor_measurement_9, sensor_measurement_11, sensor_measurement_12,
sensor_measurement_13, sensor_measurement_14, sensor_measurement_15, sensor_measurement_17,
sensor_measurement_20, sensor_measurement_21

Resampling controls:

ASYNC_RESAMPLE_HZ: fused output cadence for model windows.

ASYNC_MAX_BUFFER_SECONDS: max sensor history retained per machine.


14.5 Hardware Bridge Service

An optional hardware bridge is available at src/hardware_bridge.py.

Modes:

emulate: heterogeneous asynchronous sensor rates for rapid testing.

serial-jsonl: line-delimited JSON from real microcontrollers/PLCs over serial.

Run with Docker profile:

docker compose --profile hardware up -d --build


14.6 Multimodal Multi-Task Model

A new multimodal MTL architecture is provided for whole-machine health:

src/model_multimodal_mtl.py

Heads:

head_rul: system-level RUL regression.

head_faults: multi-label fault diagnostics.

head_anomaly_score: unknown-behavior anomaly score.

Training scaffold:

src/train_multimodal_mtl.py


14.7 Advanced Multi-Dataset Training (Imbalance-Optimized)

The multimodal trainer now supports fused training across:

ai4i2020

CWRU Bearing Dataset

MIMII Dataset

MetroPT-3 Dataset

Edge-IIoTset

Implemented optimization techniques:

Class-balanced weighting using effective number of samples

Class-weighted focal losses for fault/anomaly heads

Minority-aware sample weighting per output head

Weighted bootstrap resampling for long-tail classes

Real-sensor-ready alias mapping for asynchronous feature streams

Expected dataset layout:

data/

        ai4i2020.csv

        external/

                CWRU/

                MIMII/

                MetroPT-3/

                Edge-IIoTset/

Build fused dataset and train:

python -m src.train_multimodal_mtl --rebuild-dataset --epochs 40 --batch-size 32

Reuse existing fused dataset without rebuilding:

python -m src.train_multimodal_mtl --epochs 40 --batch-size 32

Disable bootstrap (ablation/debug):

python -m src.train_multimodal_mtl --no-bootstrap

One-command GPU training launcher from Windows PowerShell (runs in WSL):

.\scripts\start-gpu-training.ps1 -ProbeOnly

.\scripts\start-gpu-training.ps1

.\scripts\start-gpu-training.ps1 -RebuildDataset -Epochs 40 -BatchSize 32

Artifacts produced:

data/multisource_train.npz (fused training tensor pack)

data/multisource_train_report.json (source mix statistics)

models/best_multimodal_mtl.keras (best checkpoint)

models/multisource_training_report.json (imbalance + training metadata)


14.8 Portable Multimodal Runtime Artifact (Windows + WSL)

Primary portable path (recommended): export TensorFlow SavedModel from WSL:

powershell -ExecutionPolicy Bypass -File scripts/export-multimodal-savedmodel.ps1

Default output directory:

models/multimodal_savedmodel

Runtime configuration for full multimodal inference:

RUNTIME_MODEL_MODE=multimodal_savedmodel

RUNTIME_MODEL_PATH=models/multimodal_savedmodel

Optional path: export multimodal TFLite artifact:

python -m src.convert_tflite_multimodal --quantization float16

Default TFLite output:

models/model_multimodal_portable.tflite

Note:

Current multimodal BiLSTM graph may require Select TF Ops (Flex) for TFLite runtime.
If your interpreter does not include Flex delegate support, use SavedModel mode.

Runtime configuration for full multimodal inference:

RUNTIME_MODEL_MODE=multimodal_savedmodel or multimodal_tflite

RUNTIME_MODEL_PATH=models/multimodal_savedmodel or models/model_multimodal_portable.tflite

Auto-runtime candidate order (when RUNTIME_MODEL_PATH is unset):

models/best_multimodal_mtl.keras (if loadable)

models/multimodal_savedmodel

models/model_multimodal_portable.tflite

models/model_cmapss_int8.tflite (single-input fallback)


14.9 One-Command Verification Flow

Windows quick verification:

powershell -ExecutionPolicy Bypass -File scripts/verify-runtime.ps1

Windows quick verification (force SavedModel runtime mode):

powershell -ExecutionPolicy Bypass -File scripts/verify-runtime.ps1 -ForceSavedModelMode

Windows + WSL multimodal evaluation verification:

powershell -ExecutionPolicy Bypass -File scripts/verify-runtime.ps1 -RunWslMultimodalEval

WSL helper script used by the verification flow:

scripts/evaluate_multimodal_mtl_wsl.sh

SavedModel export helpers:

scripts/export-multimodal-savedmodel.ps1

scripts/export_savedmodel_multimodal_wsl.sh


---

15. Use Cases

Industrial motor monitoring

Predictive maintenance systems

IoT-based health monitoring

Edge AI deployment prototypes



---

16. Limitations

Model trained on simulated data

Requires calibration for real-world conditions

Hardware integration not fully implemented

Limited to single-machine simulation



---

17. Future Enhancements

Integration with real industrial datasets

Multi-machine support

MQTT-based architecture

Time-series database (InfluxDB / TimescaleDB)

PLC integration (Modbus / OPC UA)

Edge deployment on embedded systems



---

18. Conclusion

This project demonstrates a complete end-to-end predictive maintenance system, combining machine learning, backend engineering, real-time communication, and system monitoring.

It is designed not merely as a prototype but as a foundation for scalable industrial solutions, with clear pathways toward real-world deployment and integration.


---

