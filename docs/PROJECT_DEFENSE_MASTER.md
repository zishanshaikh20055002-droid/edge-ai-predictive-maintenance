# Project Defense Master Guide

## 1) One-line Project Definition
Edge AI Predictive Maintenance is a real-time industrial monitoring platform that ingests machine telemetry, predicts Remaining Useful Life (RUL), estimates failure risk, localizes probable fault components, assigns alarm priority, and exposes all results through APIs, WebSocket streaming, dashboarding, and observability tooling.

## 2) What Problem It Solves
Industrial maintenance is often reactive. Teams act after thresholds are crossed or after failure. This causes:
- unplanned downtime
- expensive emergency maintenance
- spare-parts waste
- safety risk

This system shifts maintenance from reactive to predictive by continuously answering:
- What is the machine condition now?
- How likely is near-term failure?
- What component is most likely responsible?
- How urgent is intervention?

## 3) Core Value Proposition
- Real-time intelligence: low-latency updates over MQTT + WebSocket
- Multi-task diagnosis: RUL, stage, fault localization, alarm policy in one stream
- Practical deployment: Dockerized stack with Prometheus + Grafana
- Extendable pipeline: simulation today, hardware PLC/gateway integration path ready
- Human-in-the-loop learning: feedback relabel + retraining hooks

## 4) System Architecture (Teacher-ready)
### Data and Inference Path
1. Telemetry producer publishes sensor events (simulated or hardware bridge).
2. MQTT broker carries events to subscriber.
3. Backend pre-processes windows and runs model inference.
4. Diagnosis layer computes fault context, alarm policy, and maintenance priority.
5. Results are persisted and broadcast over WebSocket.
6. Dashboard renders machine and fleet status in real time.
7. Metrics are scraped by Prometheus and visualized in Grafana.

### Main Components
- API and orchestration: app.py
- Ingestion and subscriber: src/ingestion.py, src/mqtt_subscriber.py
- Publisher: src/mqtt_publisher.py
- Diagnostics and policy: src/diagnostics.py, src/fault_localization.py, src/alarm_policy.py
- Data and retraining loops: src/retraining.py, src/export_fault_training_data.py
- Security and controls: src/auth.py, src/limiter.py
- Observability: src/metrics.py, prometheus.yml, grafana provisioning

## 5) ML and Analytics Stack
### Single/Primary Runtime Outputs
- RUL (regression)
- Health stage (healthy/warning/critical)
- Failure probability
- Fault component and type
- Fault confidence and severity
- Alarm level and maintenance priority

### Multi-task and Multimodal Work
- Dataset fusion from multiple sources for broader domain coverage
- Class imbalance utilities (effective number weighting, focal losses, weighted bootstrap)
- Multimodal training pipeline with process, vibration, acoustic, electrical, thermal channels
- Evaluation utilities for task-wise metrics
- Dataset verification CLI for pre-training integrity checks

### Domain Adaptation Layer (advanced extension)
- MMD loss for domain-invariant embeddings
- Domain-adversarial path with gradient reversal
- Progressive unfreezing callback for transfer stability

## 6) Security and Reliability Decisions
- JWT auth for protected endpoints
- Role-based access (admin/operator)
- Rate limiting for abuse resistance
- Defensive fallbacks when artifacts are missing or invalid
- Runtime defaults tuned for stable demo behavior

## 7) Observability and MLOps Decisions
- Prometheus metrics endpoint for model/system telemetry
- Grafana dashboards for operations visibility
- Diagnosis APIs for latest/recent per machine
- Feedback APIs for relabel and future retraining

## 8) Major Implementation Updates Completed
### Backend and Runtime
- End-to-end stack bring-up verified with Docker Compose
- TFLite output dequantization path hardened to avoid scaling distortions
- Safer runtime defaults for more realistic replay behavior
- Improved alarm/health interpretation consistency in live streams

### Dashboard
- Added richer diagnosis cards (alarm component, diagnosis version, timestamp)
- Added score visualizations for stage probabilities, component risk, component health
- Improved model source/version display for operator transparency

### Training and Data
- Added evaluation metrics module for RUL/fault/anomaly reporting
- Added dataset verification CLI for NPZ/source integrity checks
- Added domain adaptation module and PLC bridge module for next-phase deployment readiness

## 9) Known Constraints and Honest Limitations
- PLC bridge and domain adaptation are implemented but need plant-specific integration and validation before claiming production readiness.
- Ground-truth label quality strongly affects fault localization performance.
- Cross-domain generalization still depends on data coverage and adaptation tuning.
- Real-time latency and throughput differ by hardware profile and deployment topology.

## 10) How To Explain Why Your Design Is Strong
- It is modular, so each layer can evolve independently.
- It supports both synthetic and real-world data modes.
- It combines predictive modeling with operational UX and observability.
- It includes practical controls: auth, limits, alarms, and audit-style feedback loops.

## 11) 10-Minute Viva Walkthrough Structure
1. Problem and stakes (1 min)
2. System architecture and data flow (2 min)
3. ML pipeline and diagnosis logic (2 min)
4. Security, reliability, and observability (2 min)
5. Demo flow and results interpretation (2 min)
6. Limitations and future work (1 min)

## 12) Demo Checklist (Fast)
- Start stack with Docker Compose
- Verify /health and /metrics
- Open dashboard and confirm live machine updates
- Show diagnosis latest endpoint for one machine
- Show Grafana panels and Prometheus target status
- Trigger a degraded condition and explain stage vs alarm differences

## 13) Teacher-facing Closing Statement
This project is not only a model training exercise. It is a full predictive maintenance platform that integrates streaming ingestion, real-time inference, diagnosis policy, secure APIs, observability, and operator-facing decision support in one coherent system.
