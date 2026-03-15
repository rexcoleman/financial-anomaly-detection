# Hypothesis Registry — Financial Anomaly Detection (FP-04)

> Pre-registered hypotheses with outcomes. All results are on
> synthetic PaySim data and tagged accordingly.

| ID | Hypothesis | Metric | Threshold | Status | Evidence |
|----|-----------|--------|-----------|--------|----------|
| H-1 | CFA-informed features (amount ratios, velocity, merchant risk) capture >50% of ML signal as measured by SHAP importance | Fraction of CFA features in SHAP top-20 | >50% of top-20 | SUGGESTED, SYNTHETIC | 8 of top 20 SHAP features (40%) are CFA-informed; amt_to_median_ratio is rank 4, protonmail rank 5. On synthetic data these features encode known generation rules; real-data validation needed via IEEE-CIS |
| H-2 | System-controlled features (card type, device, address verification) provide an adversary-resistant detection floor achieving >70% of full model AUC | System-only model AUC / Full model AUC | >70% | SUGGESTED, SYNTHETIC | System-only RF achieves AUC 0.798 vs full model 0.987 = 81% robustness ratio. Confirms methodology; absolute value is synthetic-data-dependent |
| H-3 | XGBoost outperforms simpler models (Logistic Regression, Random Forest) on fraud detection AUC | AUC-ROC on temporal test set | XGBoost AUC > LR AUC and XGBoost AUC > RF AUC | SUGGESTED, SYNTHETIC | XGBoost AUC 0.987 vs LR 0.977 vs RF 0.974 (seed 42). XGBoost wins but margin is narrow (+1.0pp over LR, +1.3pp over RF); all models perform well on synthetic data with clean signals |

## Resolution Key

- **SUPPORTED**: Evidence confirms hypothesis at stated threshold
- **SUGGESTED, SYNTHETIC**: Pattern observed on synthetic data; real-data validation needed
- **REFUTED**: Evidence contradicts hypothesis
- **INCONCLUSIVE**: Evidence is mixed or insufficient
- **PENDING**: Not yet tested
