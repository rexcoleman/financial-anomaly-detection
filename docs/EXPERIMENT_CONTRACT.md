# EXPERIMENT CONTRACT

<!-- version: 1.0 -->
<!-- created: 2026-03-16 -->

## 1) Scope

Defines the experimental protocol for **Financial Anomaly Detection (FP-04)**.

## 2) Experiment Matrix

| Part | Description | Models | Data |
|------|-------------|--------|------|
| 1 | EDA + Hypothesis Registration | Correlation, class distribution | PaySim 100K |
| 2 | Baseline Training | LR, RF, SVM-RBF, XGBoost, LightGBM | PaySim 100K, temporal split |
| 3 | Diagnostics | Complexity curves, learning curves | PaySim 100K |
| 4 | Explainability | SHAP, controllability analysis | Trained models |
| 5 | Anomaly Detection | Isolation Forest | PaySim 100K |

## 3) Model Configuration

| Model | Key Hyperparameters | Seeds |
|-------|-------------------|-------|
| Logistic Regression | C=1.0, max_iter=1000 | [42, 123, 456, 789, 1024] |
| Random Forest | n_estimators=100, max_depth=10 | [42, 123, 456, 789, 1024] |
| SVM-RBF | C=1.0, kernel=rbf, gamma=scale | [42, 123, 456, 789, 1024] |
| XGBoost | n_estimators=100, max_depth=6, lr=0.1 | [42, 123, 456, 789, 1024] |
| LightGBM | n_estimators=100, max_depth=6, lr=0.1 | [42, 123, 456, 789, 1024] |

## 4) Cross-Part Constraints

| Constraint | Rule |
|-----------|------|
| Dataset lock | All parts use PaySim synthetic data only |
| Seed lock | All parts share [42, 123, 456, 789, 1024] |
| Split lock | Temporal split: train (first 70%), val (next 15%), test (last 15%) |
| Model lock | Parts 3-5 use models trained in Part 2 |

## 5) Primary Metric

- **AUC-ROC** (primary), reported as mean +/- std across 5 seeds
- Secondary: F1, precision, recall at optimal threshold

## 6) Budget

Wall-clock constrained (Azure B2ms). No GPU required. Each model trains in <2 min per seed.

## 7) Exit Gates

- [x] All 5 models trained across 5 seeds
- [x] AUC-ROC > 0.90 for top model (XGBoost: 0.987)
- [x] Complexity curves generated for all models
- [x] SHAP analysis complete
- [x] Hypotheses H-1 through H-6 resolved

## 8) Change Control

Changes to model list, seed list, or metric require a `CONTRACT_CHANGE` commit.
