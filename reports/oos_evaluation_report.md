# Chronological out-of-sample evaluation of the saved XGBoost model

## Evaluation design

- Model: existing `models/xgboost_model.pkl`; loaded for prediction only. No fitting, threshold changes, or model writes were performed.
- Data: `data/processed/training_dataset.csv` (15,263 labeled rows; 8 symbols).
- Training period represented in saved metadata: 2019-01-01 to 2024-05-23 (10,640 rows).
- Validation period used for model selection/refit: 2024-05-31 to 2025-07-16 (2,248 rows). The final saved artifact was fit on training + validation, not on test.
- Holdout: 2025-07-24 to 2026-09-15 (2,295 rows), the metadata-defined test set.
- Five trading dates purged immediately before test: 2025-07-17, 2025-07-18, 2025-07-21, 2025-07-22, 2025-07-23. Latest model-fit date precedes this gap; no test rows were used in fitting.
- Input check: 41 model features; data columns, saved metadata order, and model feature order match exactly. No metadata, target, or future-derived input columns were used.
- Target: existing five-session forward-return label; BUY > +3%, SELL < -3%, otherwise HOLD. Thresholds unchanged.

## Class distribution

| Split | BUY | HOLD | SELL |
|---|---:|---:|---:|
| Training | 2,195 | 6,864 | 1,581 |
| Validation | 341 | 1,592 | 315 |
| Holdout actual | 281 | 1,666 | 348 |
| Holdout predicted | 304 | 1,644 | 347 |

## Holdout classification results

- Accuracy: **0.5913** (59.13%).
- Macro F1: **0.3658**.
- Always-HOLD baseline macro F1: **0.2804**; model difference: **+0.0854**.
- Always-HOLD baseline accuracy: 0.7259 (72.59%).

### Confusion matrix (rows = actual, columns = predicted)

| Actual \ Predicted | BUY | HOLD | SELL |
|---|---:|---:|---:|
| BUY | 55 | 168 | 58 |
| HOLD | 183 | 1,248 | 235 |
| SELL | 66 | 228 | 54 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| BUY | 0.1809 | 0.1957 | 0.1880 | 281 |
| HOLD | 0.7591 | 0.7491 | 0.7541 | 1,666 |
| SELL | 0.1556 | 0.1552 | 0.1554 | 348 |

## Predicted probability distribution

Per-class probabilities over the holdout rows. Actual rate is included to aid calibration review.

| Class | Mean probability | Median | 10th pct | 90th pct | Actual rate | OVR Brier | 10-bin ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| BUY | 0.2759 | 0.2594 | 0.1923 | 0.3894 | 0.1224 | 0.1323 | 0.1543 |
| HOLD | 0.4270 | 0.4286 | 0.2961 | 0.5578 | 0.7259 | 0.2823 | 0.2989 |
| SELL | 0.2971 | 0.2964 | 0.1905 | 0.4025 | 0.1516 | 0.1546 | 0.1454 |

- Multiclass log loss: **0.9626**.
- Multiclass Brier score (mean sum of squared class-probability errors): **0.5691**.
- Calibration interpretation: this holdout has 2,295 rows (BUY 281, HOLD 1,666, SELL 348), sufficient for a coarse aggregate/10-bin diagnostic, but not to certify calibration across future market regimes. ECE is bin-dependent and should be treated as descriptive.

## Findings and limitations

- HOLD is the majority outcome (72.6%); the model predicted HOLD for 71.6% of holdout rows.
- BUY recall is 19.6% and SELL recall is 15.5%. These are weak for actionable class detection: each misses roughly four out of five actual events.
- Macro F1 is 0.3658 versus 0.2804 for always-HOLD, so the model beats that baseline on balanced class F1 despite lower accuracy (59.1% vs 72.6%).
- This is an out-of-sample evaluation for the existing saved artifact because the artifact was refit only through the validation period and the test window is later and date-disjoint with a five-date purge. It evaluates this historical holdout, not current/future performance.
- No model, training data, thresholds, or class weights were changed.
