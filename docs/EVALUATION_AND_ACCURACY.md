# Evaluation and Accuracy of the System

## 1) Short Answer
There is no single accuracy number for this project because it is a multi-task predictive maintenance system. Instead, we evaluate each output separately:
- Remaining Useful Life (RUL): regression metrics
- Fault localization: multi-label classification metrics
- Anomaly detection: binary classification metrics
- Uncertainty quality: coverage and calibration metrics

That is the most correct way to explain the system to teachers.

## 2) What the System Predicts
The system produces these outputs in the current architecture:
- RUL estimate
- health stage: healthy, warning, critical
- fault component / fault type
- failure probability
- alarm level and maintenance priority
- probable causes and recommended actions

Because each output is different, the evaluation metric also changes.

## 3) Accuracy Is Not One Number Here
If a teacher asks, "What is your accuracy?" the strongest answer is:
- For regression, we do not use classification accuracy.
- For multi-label faults, we use precision, recall, F1, and AUC.
- For anomaly detection, we use ROC-AUC and PR-AUC.
- For the full system, we judge operational usefulness, not a single score.

So the project is evaluated as a multi-task system, not a single-label classifier.

## 4) Metrics Used in the Codebase
The implementation is in [src/evaluation_metrics.py](../src/evaluation_metrics.py).

### RUL Metrics
Used for the RUL head:
- MAE: mean absolute error
- RMSE: root mean squared error
- MAPE: mean absolute percentage error
- R²: variance explained

These show how close the predicted useful life is to the true value.

### Fault Localization Metrics
Used for the multi-label fault head:
- precision
- recall
- F1 score
- per-class ROC-AUC
- macro F1
- hamming loss
- subset-style accuracy on thresholded outputs

These tell us whether the model identifies the right fault components.

### Anomaly Metrics
Used for the anomaly head:
- ROC-AUC
- PR-AUC
- best F1 threshold
- F1 at the selected threshold

These show how well the model separates normal from abnormal behavior.

### Uncertainty Metrics
Used when Monte Carlo Dropout or predictive intervals are used:
- coverage
- average interval width
- miscalibration
- average uncertainty

These are important because a reliable system should know when it is unsure.

## 5) How Training Reports Accuracy
The training pipeline in [src/train_multimodal_mtl.py](../src/train_multimodal_mtl.py) focuses on:
- validation loss
- per-head validation metrics
- best epoch
- imbalance statistics
- training device and runtime settings

The model is not judged by one accuracy percentage. It is judged by whether all three tasks improve together.

## 6) What To Tell Teachers About Accuracy
If asked directly, say:

"This project does not use a single accuracy score because it is a multi-task system. We evaluate RUL with MAE and RMSE, fault localization with precision, recall, F1, and AUC, and anomaly detection with ROC-AUC and PR-AUC. That gives a more honest view of system quality than a single number."

If they ask why not just report accuracy:
- accuracy can be misleading on imbalanced fault data
- a model can look good on accuracy and still miss rare faults
- maintenance systems care more about missed failures than raw class accuracy

## 7) How To Compare Results Fairly
When comparing versions of the system, compare:
- lower MAE and RMSE for RUL
- higher F1 and AUC for faults
- higher ROC-AUC and PR-AUC for anomaly detection
- better calibration and coverage for uncertainty
- lower false negatives on critical faults

Do not compare only training accuracy.

## 8) Best Explanation of "Good Performance"
A good model in this project is one that:
- predicts degradation early enough to be useful
- identifies the likely failing component
- avoids too many false alarms
- handles class imbalance well
- stays calibrated and explainable
- works consistently on both simulated and hardware-streamed data

## 9) If You Need a One-Line Viva Answer
"Our system is evaluated task-wise: RUL uses regression errors, fault localization uses precision/recall/F1/AUC, and anomaly detection uses ROC-AUC and PR-AUC, because a single accuracy score would not properly represent this multi-task maintenance system."
