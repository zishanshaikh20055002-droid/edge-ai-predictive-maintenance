# Teacher Q and A Bank

## A) Foundation Questions
### Q1. What exactly is your project?
It is a real-time predictive maintenance platform that ingests telemetry, predicts RUL and health stage, localizes probable faults, and prioritizes maintenance actions.

### Q2. What is novel in your implementation?
We combined streaming, inference, diagnosis policy, security, observability, and operator UX into one integrated system instead of a model-only prototype.

### Q3. Why edge AI?
Edge-side inference reduces latency, improves resilience in intermittent networks, and enables near-source operational decisions.

## B) Data and Modeling Questions
### Q4. Which outputs are predicted?
RUL, health stage, failure probability, likely fault component/type, and policy-level alarm/priority.

### Q5. How do you handle class imbalance?
Through effective-number class weighting, focal-style objectives, and weighted bootstrap sampling.

### Q6. Why multimodal training?
Different fault signatures appear in different modalities. Fusion improves robustness over single-channel learning.

### Q7. How do you validate datasets?
Using a verification CLI that checks shape consistency, missing keys, NaN/Inf, and class distribution.

### Q8. What is domain adaptation and why needed?
It reduces mismatch between source and target domains, improving generalization when real plant data differs from training data.

## C) Runtime and System Questions
### Q9. Why MQTT plus WebSocket?
MQTT handles machine telemetry messaging efficiently. WebSocket pushes processed diagnosis updates to UI clients with low latency.

### Q10. How do you ensure API security?
JWT authentication, role-based authorization, and request rate limiting.

### Q11. What happens if a model artifact is missing?
The system falls back to rule-based paths where designed, so diagnosis remains available.

### Q12. How do you observe health of your platform?
Prometheus metrics plus Grafana dashboards, along with API health checks.

## D) Diagnostic Interpretation Questions
### Q13. Difference between health stage and alarm level?
Health stage reflects model-estimated machine condition. Alarm level is operational urgency derived from policy thresholds and risk windows.

### Q14. Why can warning stage still trigger high alarm?
Because alarm policy also considers factors like failure probability trend and time-to-failure urgency.

### Q15. How should an operator use your dashboard?
Read stage and fault indicators first, then use probability and component risk bars to prioritize checks and maintenance windows.

## E) Engineering Tradeoff Questions
### Q16. Why not only use deep learning end-to-end?
Operational systems need explainability, policy control, and fallback behavior. Pure black-box output is hard to operationalize safely.

### Q17. Why Docker?
Reproducibility, fast setup, dependency isolation, and easier deployment across development machines.

### Q18. Why keep simulation mode?
It accelerates testing, demos, and regression checks before hardware integration.

## F) PLC and Integration Questions
### Q19. Is PLC integration complete?
A bridge module is implemented with protocol handlers and buffering. Final plant-grade deployment still needs site-specific mapping, protocol credentials, and acceptance testing.

### Q20. Which protocols are supported in your bridge design?
Modbus TCP, OPC UA, MQTT gateway mode, and file polling for controlled test scenarios.

### Q21. How do you normalize different sensor names?
Through a canonical schema contract that maps heterogeneous source names to standardized feature keys.

## G) Evaluation and Results Questions
### Q22. Which evaluation metrics are used?
RUL: MAE/RMSE/MAPE/R2, fault tasks: precision/recall/F1 and AUC, anomaly tasks: ROC-AUC and PR-AUC.

### Q23. How do you avoid misleading metrics?
By reporting per-class and macro metrics, inspecting class support, and not relying on accuracy alone under imbalance.

## H) Risk, Limitations, and Future Work
### Q24. What are your current limitations?
Cross-domain performance still needs plant-specific adaptation tuning and more real sensor ground truth.

### Q25. What are your top next steps?
- calibrate domain adaptation on real plant streams
- complete PLC-site mapping and hardened deployment runbooks
- automate retraining quality gates

## I) Short High-Confidence Answers You Can Memorize
- This is an end-to-end predictive maintenance platform, not only a model.
- We engineered data, inference, policy, security, observability, and UX together.
- Our key strength is actionable diagnosis with real-time operational context.
- We are transparent about what is production-ready now versus integration-ready next.
