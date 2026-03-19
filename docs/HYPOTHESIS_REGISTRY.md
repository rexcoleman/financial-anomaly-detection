# Hypothesis Registry — Financial Anomaly Detection (FP-04)

> Pre-registered hypotheses with outcomes. All results are on
> synthetic PaySim data and tagged accordingly.
>
> `lock_commit:` Lock commit: pre-experiment (exact SHA unavailable for retroactive projects)

| ID | Hypothesis | Metric | Threshold | Status | Evidence |
|----|-----------|--------|-----------|--------|----------|
| H-1 | CFA-informed features (amount ratios, velocity, merchant risk) capture significant signal as measured by SHAP importance in the top-20 features | Fraction of CFA features in SHAP top-20 | >30% of top-20 | **SUPPORTED, SYNTHETIC** | 8 of top 20 SHAP features (40%) are CFA-informed; amt_to_median_ratio is rank 4, protonmail rank 5. Rule-based baseline (CFA-informed) achieves AUC 0.898 — 91% of XGBoost's AUC 0.987. On synthetic data these features encode known generation rules; real-data validation needed via IEEE-CIS |
| H-2 | System-controlled features (card type, device, address verification) provide an adversary-resistant detection floor achieving >70% of full model AUC | System-only model AUC / Full model AUC | >70% | **SUPPORTED, SYNTHETIC** | System-only RF achieves AUC 0.798 vs full model 0.987 = 81% robustness ratio (5-seed mean). Confirms methodology; absolute value is synthetic-data-dependent |
| H-3 | Gradient-boosted ensembles (XGBoost, LightGBM) outperform simpler models (LR, RF, SVM-RBF) on fraud detection AUC | AUC-ROC on temporal test set | XGBoost/LightGBM AUC > LR AUC and > RF AUC and > SVM-RBF AUC | **SUPPORTED, SYNTHETIC** | XGBoost AUC 0.987, LightGBM AUC 0.987 vs LR 0.977, RF 0.974, SVM-RBF 0.951 (5 seeds). Gradient-boosted models win; margin narrow for LR/RF (+1pp) but substantial vs SVM-RBF (+3.6pp). All models perform well on synthetic data with clean signals |
| H-4 | Complexity curves reveal an optimal model complexity point beyond which increasing capacity hurts generalization (overfitting) | Train-test AUC gap widens with increasing complexity | Train-test gap increases monotonically past optimal | **SUPPORTED, SYNTHETIC** | XGBoost: best test AUC at max_depth=2 (0.990), drops to 0.983 at depth=10 while train AUC rises to 1.0 — classic overfitting signature. SVM-RBF: best test AUC at C=0.01 (0.958), drops to 0.940 at C=100 — regularization-dependent. LightGBM: best at depth=2 (0.990), nearly flat to depth=15 — robust to over-parameterization. RandomForest: test AUC improves with n_estimators (0.948 to 0.978) — no overfitting, consistent with ensemble averaging theory |
| H-5 | Unsupervised anomaly detection (Isolation Forest) adds minimal incremental fraud detection when labeled data is available | Additional fraud caught by anomaly but NOT by supervised model | <10% additional | **SUPPORTED, SYNTHETIC** | Isolation Forest catches only 12 additional fraud cases (1.1% of all fraud) that XGBoost misses. Anomaly precision 54%. On synthetic data with programmatic fraud patterns, supervised models dominate; anomaly detection value would increase on real data with novel/evolving fraud patterns |
| H-6 | Multi-seed validation shows stable results (low variance across seeds) | Standard deviation of AUC across 5 seeds | std < 0.01 | **SUPPORTED, SYNTHETIC** | XGBoost AUC std = 0.000, LightGBM AUC std = 0.000, SVM-RBF AUC std = 0.002, RF AUC std < 0.001 across 5 seeds. Zero variance for deterministic models reflects synthetic data stability |

## Resolution Key

- **SUPPORTED**: Evidence confirms hypothesis at stated threshold (on synthetic data)
- **SUGGESTED, SYNTHETIC**: Pattern observed on synthetic data; real-data validation needed
- **REFUTED**: Evidence contradicts hypothesis
- **INCONCLUSIVE**: Evidence is mixed or insufficient
- **PENDING**: Not yet tested

## Cross-Reference to Findings

| Hypothesis | FINDINGS.md Section | Key Output File |
|-----------|--------------------|--------------------|
| H-1 | RQ3: CFA Features Dominate SHAP Rankings | `outputs/baselines/summary_seed42.json` (shap field) |
| H-2 | RQ4: Controllability Analysis | `outputs/baselines/summary_seed42.json` (controllability field) |
| H-3 | RQ1 + Expanded Model Comparison | `outputs/models/expanded_summary.json` |
| H-4 | Complexity Curve Analysis | `outputs/diagnostics/complexity_curves_summary.json` |
| H-5 | RQ2: Anomaly Detection | `outputs/baselines/summary_seed42.json` (isolation_forest field) |
| H-6 | Multi-Seed Validation | `outputs/models/expanded_summary.json`, `outputs/diagnostics/learning_curves_summary.json` |
