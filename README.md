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

