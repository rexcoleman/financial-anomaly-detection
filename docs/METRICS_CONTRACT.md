# METRICS CONTRACT

<!-- version: 1.0 -->
<!-- created: 2026-03-16 -->

## 1) Primary Metric

| Property | Value |
|----------|-------|
| Metric | AUC-ROC |
| Direction | Maximize |
| Computation | `sklearn.metrics.roc_auc_score(y_true, y_score)` |
| Reporting | Mean +/- std across 5 seeds |

## 2) Secondary Metrics

| Metric | Computation | Purpose |
|--------|-------------|---------|
| F1 (binary) | `sklearn.metrics.f1_score(y_true, y_pred)` | Threshold-dependent performance |
| Precision | `sklearn.metrics.precision_score(y_true, y_pred)` | False positive cost |
| Recall | `sklearn.metrics.recall_score(y_true, y_pred)` | Fraud catch rate |
| AUC-PR | `sklearn.metrics.average_precision_score(y_true, y_score)` | Imbalanced-class performance |

## 3) Reporting Format

### Per-Seed Output

```json
{
  "model": "xgboost",
  "seed": 42,
  "auc_roc": 0.987,
  "f1": 0.85,
  "precision": 0.82,
  "recall": 0.88,
  "wall_clock_s": 12.3
}
```

### Aggregated Summary

```
Model       | AUC-ROC (mean +/- std) | F1 (mean +/- std)
XGBoost     | 0.987 +/- 0.000        | 0.85 +/- 0.01
LightGBM    | 0.987 +/- 0.000        | 0.84 +/- 0.01
LR          | 0.977 +/- 0.001        | 0.79 +/- 0.02
RF          | 0.974 +/- 0.001        | 0.81 +/- 0.01
SVM-RBF     | 0.951 +/- 0.002        | 0.73 +/- 0.02
```

## 4) Threshold Selection

- Optimal threshold selected on validation set (NOT test)
- Method: maximize F1 on validation predictions
- Same threshold applied to test set for final reporting

## 5) Multi-Seed Stability

All metrics reported as mean +/- std across seeds [42, 123, 456, 789, 1024]. Single-seed results are never used for claims.

## 6) Change Control

Changes to primary metric, threshold selection method, or seed list require a `CONTRACT_CHANGE` commit.
