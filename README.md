Edge AI Predictive Maintenance System

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

Machine learning model deployed using TensorFlow Lite

Predicts Remaining Useful Life (RUL)

Classifies machine condition:

Healthy

Warning

Critical



4.3 Dual Data Mode Support

Simulation Mode using dataset inputs

Hardware Mode for real-time sensor integration


4.4 Secure Authentication

JWT-based authentication

Stateless session management

Token-based authorization


4.5 Role-Based Access Control (RBAC)

Admin: Full system control

Operator: Monitoring access only


4.6 Rate Limiting

Prevents brute-force attacks

Protects critical endpoints

Ensures system stability under load


4.7 Control System

Allows simulation of machine conditions:

Normal

Degradation

Failure


Dynamically affects prediction outputs


4.8 Data Persistence

Stores predictions and sensor values

Enables historical analysis


4.9 Observability and Monitoring

Metrics collection using Prometheus

Visualization using Grafana

Tracks system performance and health


4.10 Containerization

Docker-based deployment

Environment consistency across systems

Simplified setup and scaling


4.11 Complete Machine Diagnosis (Real-Time)

In addition to RUL and health stage, the system now streams full diagnosis fields per machine:

Electrical telemetry: voltage, current, power

Mechanical telemetry: speed, torque, vibration, tool wear

Thermal telemetry: process and air temperatures

Machine KPIs: health index, failure probability, time-to-failure (hours)

Fault localization: likely component (stator/rotor/bearing/cooling/power_supply/lubrication),
fault type, severity, confidence, probable causes, and recommended actions

New diagnosis APIs:

GET /diagnosis/latest/{machine_id}

GET /diagnosis/recent/{machine_id}?limit=100


4.12 Supervised Fault Localization (Phase 2)

The system now supports a trainable component-localization model that can replace
rule-only fault part ranking when a trained artifact is available.

Runtime behavior:

If models/fault_localizer.pkl exists, live diagnosis uses ML component probabilities.

If missing or invalid, the system automatically falls back to rule-based localization.

Model endpoints:

GET /diagnosis/model/fault-localizer

POST /diagnosis/model/fault-localizer/reload (admin)

Training workflow:

1) Create labeled dataset CSV from your plant logs using data/fault_localization_template.csv schema.

2) Optionally bootstrap from historical diagnosis DB:

python -m src.export_fault_training_data --min-confidence 0.6

3) Train model:

python -m src.train_fault_localization --input data/fault_localization_labeled.csv

4) Reload model in running API:

POST /diagnosis/model/fault-localizer/reload


4.13 Alarm Policy and Maintenance Priority

Alarm levels are generated in real time from failure probability, fault severity,
confidence, and time-to-failure windows.

Outputs now include:

alarm_level (INFO, ADVISORY, ALERT, EMERGENCY)

maintenance_priority (P4 to P1)

alarm_reasons (list)

recommended_window_hours

Tuning env vars:

ALARM_FAILURE_WARN

ALARM_FAILURE_CRIT

ALARM_TTF_WARN_HOURS

ALARM_TTF_CRIT_HOURS


4.14 Industrial Hardening (Fleet + Feedback + Auto-Retraining)

The platform now includes three production hardening loops:

1) Multi-machine fleet operations:

GET /fleet/overview

GET /fleet/machines

The dashboard can monitor and rank all active machines by alert pressure,
while still allowing deep drill-down on a selected machine.

2) Human feedback relabel loop:

POST /feedback/relabel

GET /feedback/relabels?resolved=false&limit=20

POST /feedback/relabels/{feedback_id}/resolve (admin)

Operators can submit corrected fault-part labels for any diagnosis record.
Resolved feedback is retained for supervised retraining and traceability.

3) Scheduled auto-retraining with drift detection:

GET /retraining/status

POST /retraining/run-now (admin)

The scheduler periodically evaluates feature drift against the baseline profile
stored in models/fault_localizer_meta.json (feature statistics exported at training time).
Retraining can be triggered by drift and/or enough resolved relabel samples,
with cooldown and minimum-data gates.

Scheduler and drift tuning environment variables:

RETRAIN_ENABLED

RETRAIN_CHECK_MINUTES

RETRAIN_MIN_COOLDOWN_MINUTES

RETRAIN_MIN_ROWS

RETRAIN_MIN_FEEDBACK

RETRAIN_MIN_CONFIDENCE

RETRAIN_REQUIRE_DRIFT

DRIFT_ZSCORE_THRESHOLD

DRIFT_WINDOW_ROWS


Training export note:

python -m src.export_fault_training_data now merges resolved human feedback labels
into the supervised target (fault_component) before retraining.



---

5. System Architecture

Data Source (Simulation / Sensors)
        │
        ▼
Preprocessing Layer (Scaling / Formatting)
        │
        ▼
Machine Learning Model (TFLite)
        │
        ▼
FastAPI Backend
   ├── REST APIs
   ├── WebSocket Streaming
   ├── Authentication (JWT)
   ├── Rate Limiting
   └── Control Logic
        │
        ▼
Database (SQLite)
        │
        ▼
Frontend Dashboard (Real-Time UI)
        │
        ▼
Prometheus (Metrics Collection)
        │
        ▼
Grafana (Visualization)


---

6. Technology Stack

Backend

Python

FastAPI (asynchronous API framework)

TensorFlow Lite (optimized ML inference)

WebSockets (real-time communication)


Frontend

HTML, CSS, JavaScript

Real-time dashboard updates


Database

SQLite (lightweight relational database)


Security

JWT Authentication

Role-Based Access Control


Monitoring

Prometheus (metrics collection)

Grafana (metrics visualization)


DevOps

Docker (containerization)

Git (version control)



---

7. Machine Learning Component

7.1 Model Overview

The system uses a machine learning model to predict Remaining Useful Life (RUL) based on sensor data.

7.2 Deployment

Converted to TensorFlow Lite format

Optimized for low-latency inference


7.3 Output Interpretation

RUL > threshold → Healthy

Moderate RUL → Warning

Low RUL → Critical


7.4 Current Status

Trained on processed/simulated dataset

Designed to integrate real industrial datasets



---

8. Backend Design

8.1 FastAPI

Used as the core backend framework due to:

High performance

Asynchronous request handling

Native WebSocket support


8.2 API Design

REST endpoints for control and authentication

WebSocket endpoint for real-time streaming


8.3 Authentication Flow

User login → token generation

Token used for protected endpoints

Role-based authorization enforced



---

9. Real-Time Communication

WebSockets are used for:

Continuous streaming of predictions

Eliminating polling overhead

Achieving near real-time updates



---

10. Monitoring and Observability

10.1 Prometheus

Collects system metrics

Tracks:

request count

response time

error rates

system usage



10.2 Grafana

Visualizes metrics in dashboards

Enables performance analysis

Detects anomalies in system behavior



---

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

