# Model B experiment and final chronological holdout results

## Split and leakage checks

- Training rows: 10,640 (2019-01-01 through 2024-05-23).
- Five-date purge before validation: 2024-05-24, 2024-05-27, 2024-05-28, 2024-05-29, 2024-05-30.
- Validation rows: 2,248 (2024-05-31 through 2025-07-16); used for candidate selection.
- Five-date purge before final test: 2025-07-17 through 2025-07-23.
- Final test rows: 2,295 (2025-07-24 through 2026-09-15). During this Model B experiment, test features and labels were not passed to candidate fitting/scoring or used for selection; only split dates were inspected until final evaluation. The full CSV was loaded to construct the chronological partitions. This same period had previously been evaluated for Model A, so it is not a never-inspected holdout across project history; Model B selection did not use its prior metrics.
- For every symbol, the latest training label endpoint is 2024-05-30 and the latest validation label endpoint is 2025-07-23. No five-session target window crosses into the next split.
- Feature availability was determined from training rows only. Both models use the same 41 feature names and order; no metadata, target, or future-derived fields are inputs. Conversion: numeric coercion, infinities to missing, float32; no scaler is fit. Class weights were calculated from training counts only.
- Hyperparameter selection used validation macro F1. Test rows/outcomes were not used for fitting, weighting, feature selection, calibration, or selection.

## Imbalance and experiment

- Training counts: BUY 2,195, HOLD 6,864, SELL 1,581. Validation: BUY 341, HOLD 1,592, SELL 315. Test actual: BUY 281, HOLD 1,666, SELL 348.
- Search: 72 configurations over none/balanced/square-root-balanced weights; 200/400 trees; depth 2/3/4; learning rate 0.03/0.05; `min_child_weight` 1/5. Other settings follow the existing XGBoost setup.
- Minority-class macro F1 is (BUY F1 + SELL F1) / 2. Selection used validation macro F1; minority macro F1 was the tie-breaker.
- Selected parameters: `{"class_weight": "balanced", "learning_rate": 0.05, "max_depth": 2, "min_child_weight": 1, "n_estimators": 400}`. Validation macro F1 0.3472; validation minority macro F1 0.1557.
- Final Model B fit used training + validation rows, with weights derived from training counts only. No probability calibrator was fit.

## Final test class distribution

| Class | Actual | Model A predicted | Model B predicted | Always-HOLD predicted |
|---|---:|---:|---:|---:|
| BUY | 281 | 304 | 287 | 0 |
| HOLD | 1,666 | 1,644 | 1,637 | 2,295 |
| SELL | 348 | 347 | 371 | 0 |

## Aggregate comparison

| Model | Accuracy | Macro F1 | BUY/SELL mean F1 |
|---|---:|---:|---:|
| Model A | 0.5913 | 0.3658 | 0.1717 |
| Model B | 0.5874 | 0.3630 | 0.1691 |
| Always-HOLD | 0.7259 | 0.2804 | 0.0000 |

- Model B minus Model A: macro F1 -0.0028; BUY recall -0.0142; SELL recall +0.0086.

## Confusion matrices

Rows are actual; columns are predicted.

### Model A

| Actual \ Predicted | BUY | HOLD | SELL |
|---|---:|---:|---:|
| BUY | 55 | 168 | 58 |
| HOLD | 183 | 1,248 | 235 |
| SELL | 66 | 228 | 54 |

### Model B

| Actual \ Predicted | BUY | HOLD | SELL |
|---|---:|---:|---:|
| BUY | 51 | 172 | 58 |
| HOLD | 170 | 1,240 | 256 |
| SELL | 66 | 225 | 57 |

## Per-class test metrics

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| Model A | BUY | 0.1809 | 0.1957 | 0.1880 | 281 |
| Model A | HOLD | 0.7591 | 0.7491 | 0.7541 | 1,666 |
| Model A | SELL | 0.1556 | 0.1552 | 0.1554 | 348 |
| Model B | BUY | 0.1777 | 0.1815 | 0.1796 | 281 |
| Model B | HOLD | 0.7575 | 0.7443 | 0.7508 | 1,666 |
| Model B | SELL | 0.1536 | 0.1638 | 0.1586 | 348 |
| Always-HOLD | BUY | 0.0000 | 0.0000 | 0.0000 | 281 |
| Always-HOLD | HOLD | 0.7259 | 1.0000 | 0.8412 | 1,666 |
| Always-HOLD | SELL | 0.0000 | 0.0000 | 0.0000 | 348 |

## Probability summaries and calibration

Diagnostics use final test predictions only; they were not used to tune parameters or fit a calibrator. ECE uses 10 equal-width bins.

| Model | Class | Mean prob. | Median | 10th pct | 90th pct | Actual rate | OVR Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Model A | BUY | 0.2759 | 0.2594 | 0.1923 | 0.3894 | 0.1224 | 0.1323 | 0.1543 |
| Model A | HOLD | 0.4270 | 0.4286 | 0.2961 | 0.5578 | 0.7259 | 0.2823 | 0.2989 |
| Model A | SELL | 0.2971 | 0.2964 | 0.1905 | 0.4025 | 0.1516 | 0.1546 | 0.1454 |
| Model B | BUY | 0.2644 | 0.2446 | 0.1747 | 0.3904 | 0.1224 | 0.1302 | 0.1420 |
| Model B | HOLD | 0.4398 | 0.4418 | 0.2958 | 0.5850 | 0.7259 | 0.2760 | 0.2861 |
| Model B | SELL | 0.2958 | 0.2943 | 0.1779 | 0.4115 | 0.1516 | 0.1561 | 0.1441 |

| Model | Log loss | Multiclass Brier |
|---|---:|---:|
| Model A | 0.9626 | 0.5691 |
| Model B | 0.9522 | 0.5623 |

The holdout has 2,295 rows (281 BUY, 1,666 HOLD, 348 SELL), adequate for coarse aggregate diagnostics but not guarantees across market regimes. ECE is descriptive and bin-dependent. Always-HOLD has no meaningful learned-probability calibration score.

## Feature importance and interpretation

Model B top gain features:

| Rank | Feature | Gain importance |
|---:|---|---:|
| 1 | India_VIX_Close | 0.0467 |
| 2 | Volatility_30D | 0.0371 |
| 3 | Low | 0.0360 |
| 4 | SMA_50 | 0.0321 |
| 5 | High_Low_Pct | 0.0316 |
| 6 | Open | 0.0313 |
| 7 | NIFTY_Return_20D | 0.0303 |
| 8 | SMA_200 | 0.0289 |
| 9 | SMA_5 | 0.0280 |
| 10 | SMA_100 | 0.0280 |

- BUY recall: Model A 19.6%, Model B 18.1%. SELL recall: Model A 15.5%, Model B 16.4%. Assess these as measured; low recall means many true events are missed.
- Global gain importance is not class-specific and does not establish that a feature distinguishes BUY from SELL.
- Model B is experimental and was not deployed. This final test was evaluated once after selection; do not use it for further tuning.
- Model A model and metadata files were not modified.
