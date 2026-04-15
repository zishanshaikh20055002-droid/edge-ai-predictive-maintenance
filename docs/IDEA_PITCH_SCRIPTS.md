# Idea Pitch Scripts

## 1) 60-Second Elevator Pitch
Our project is an Edge AI Predictive Maintenance platform for industrial machines. Instead of waiting for breakdowns, we stream sensor data in real time, estimate remaining useful life, detect warning and critical stages, localize likely fault components, and generate maintenance priorities. The system is fully deployable with Docker, uses MQTT and WebSocket for low-latency updates, and includes Prometheus-Grafana monitoring. We built both ML intelligence and production-style operations, so it is practical, measurable, and extendable to real PLC hardware.

## 2) 3-Minute Pitch (Class Demo)
### Opening
Industrial failures are expensive and often detected too late. We designed a system that predicts degradation early and explains what is failing.

### What We Built
- Real-time ingestion pipeline with MQTT
- ML inference for RUL, stage, failure probability
- Fault localization and alarm policy layer
- Secure API + role controls
- Live dashboard with component-level insights
- Observability stack with Prometheus and Grafana

### Why It Is Strong
This is not a notebook-only prototype. It is an integrated platform:
- data flows continuously
- predictions are persisted and broadcast
- operators can act on alarms and maintenance windows
- engineers can inspect metrics and drift/retraining signals

### Closing
We moved from reactive monitoring to predictive decision support with a deployable architecture.

## 3) 7-Minute Pitch (Viva/Panel)
### Slide 1: Problem
Traditional threshold alerts are reactive. We need early, explainable, machine-specific risk detection.

### Slide 2: Architecture
Publisher/PLC -> MQTT -> Subscriber -> Model -> Diagnosis -> API/WebSocket -> Dashboard -> Prometheus/Grafana.

### Slide 3: ML Outputs
- RUL regression
- health stage classification
- failure probability
- fault component probabilities
- alarm level and maintenance priority

### Slide 4: Operational Hardening
- JWT auth, RBAC, rate limiter
- fault-localizer fallback rules
- robust runtime defaults
- improved quantized model handling

### Slide 5: Dashboard and UX
We added richer diagnosis cards plus score bars for stage probabilities, component risk, and health percentages. This helps non-ML users interpret model state quickly.

### Slide 6: Training and Expansion
- multisource dataset builder
- imbalance-aware losses and bootstrap
- metrics and verification tooling
- domain adaptation module and PLC bridge for next phase

### Slide 7: Impact and Future
Current: real-time predictive monitoring.
Next: plant-specific domain adaptation calibration and full PLC integration validation.

## 4) Slide-by-Slide Voice Cues
- Keep each slide to one message.
- Explain one design tradeoff per slide.
- Use numbers (latency, rate, thresholds) when possible.
- Always connect model output to maintenance action.

## 5) Common Speaking Mistakes To Avoid
- Do not claim production at scale without qualification.
- Do not over-focus on model architecture and ignore operations.
- Do not confuse status stage with alarm urgency.
- Do not skip limitations.

## 6) Strong Closing Lines
- We built an end-to-end predictive maintenance platform, not just a model.
- The design is modular, observable, secure, and ready for hardware-grounded expansion.
- The project demonstrates both AI capability and real engineering execution.
