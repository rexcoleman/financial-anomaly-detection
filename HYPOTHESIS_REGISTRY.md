# Hypothesis Registry — Financial Anomaly Detection (FP-04)

> **govML v2.5 format** | **Profile:** blog-track
> **Date:** 2026-03-16 | **Author:** Rex Coleman (CFA)
> **Data:** PaySim synthetic (100K transactions, 3.5% fraud rate)
> **Seeds:** 42, 123, 456, 789, 1024

---

## H-1: ML Outperforms Rule-Based Scoring

| Field | Value |
|-------|-------|
| **Hypothesis** | ML models outperform CFA-informed rule-based scoring by at least 5 percentage points AUC-ROC on synthetic fraud detection data. |
| **Metric** | AUC-ROC difference: best ML model minus rule-based baseline. |
| **Threshold** | >= 5 percentage points AUC improvement. |
| **Status** | **SUPPORTED** |
| **Evidence** | XGBoost achieves AUC 0.987 vs rule-based AUC 0.898, a gap of +8.9pp. LogisticRegression (+7.9pp) and RandomForest (+7.6pp) also exceed the 5pp threshold. All results on synthetic PaySim data across 5 seeds with zero cross-seed variance for XGBoost. The rule-based baseline is strong because it encodes CFA domain knowledge (high amounts, address mismatches, merchant risk, timing patterns), confirming that domain expertise sets a floor, not a ceiling. |
| **Limitations** | Synthetic data: fraud signals are cleaner than real-world transactions. Rule-based baseline may benefit from matching the data generation process. Real IEEE-CIS data would provide a more rigorous test. |

---

## H-2: CFA-Informed Features Rank in Top-10 SHAP Importance

| Field | Value |
|-------|-------|
| **Hypothesis** | CFA-informed engineered features (amt_to_median_ratio, suspicious_time, protonmail, high_risk_country, high_risk_merchant, amt_zscore, high_amount, amt_log) rank among the top 10 features by mean absolute SHAP value. |
| **Metric** | Count of CFA-informed features in the top 20 SHAP-ranked features. |
| **Threshold** | At least 1 CFA feature in top 10. |
| **Status** | **SUPPORTED** |
| **Evidence** | 8 of the top 20 SHAP features are CFA-informed (40%). Notably, amt_to_median_ratio ranks 4th (SHAP 1.275) and protonmail ranks 5th (SHAP 0.770) -- both in the top 10. high_risk_country ranks 10th (SHAP 0.220). A pure ML practitioner would use raw TransactionAmt; the CFA perspective engineers ratios relative to merchant/card norms, capturing domain-specific signal that raw features miss. |
| **Limitations** | On synthetic data, CFA features encode known generation rules, inflating their apparent importance. Real data with adversarial dynamics may show different SHAP rankings. |

---

## H-3: System-Controlled Features Achieve >= 70% of Full Model AUC

| Field | Value |
|-------|-------|
| **Hypothesis** | A model trained only on system-controlled features (card_type, DeviceType, DeviceInfo, merchant_risk_score, high_risk_merchant, addr_mismatch) achieves at least 70% of the full model's AUC-ROC, establishing an adversary-resistant detection floor. |
| **Metric** | Robustness ratio: system-only model AUC / full model AUC. |
| **Threshold** | >= 70% robustness ratio. |
| **Status** | **SUPPORTED** |
| **Evidence** | System-only RandomForest achieves AUC 0.798 vs full model AUC 0.987, yielding an 81% robustness ratio (5-seed mean). This is the 5th domain where controllability analysis has been applied (after IDS, CVE, Agents, Crypto), validating the methodology's cross-domain generalizability. 6 system-controlled features vs 12 fraudster-controlled features. |
| **Limitations** | The 81% floor is synthetic-data-dependent. On real transaction data with adversarial dynamics (where fraudsters actively manipulate controllable features), the robustness ratio would likely differ. The conceptual finding -- that system-controlled features provide an adversary-resistant floor -- is the transferable insight. |

---

## H-4: Unsupervised Anomaly Detection Adds >= 10% Fraud Recall Over Supervised

| Field | Value |
|-------|-------|
| **Hypothesis** | Unsupervised anomaly detection (Isolation Forest) catches at least 10% additional fraud cases that the best supervised model misses, demonstrating value as a complementary detection layer. |
| **Metric** | Percentage of total fraud caught by Isolation Forest but NOT by the best supervised model. |
| **Threshold** | >= 10% additional fraud recall. |
| **Status** | **REFUTED** |
| **Evidence** | Isolation Forest catches only 12 additional fraud cases that XGBoost misses, representing 1.1% of all fraud -- far below the 10% threshold. Isolation Forest has decent standalone precision (54% of flagged anomalies are fraud, 602 of 1,118) but adds negligible incremental value when labeled data is available. |
| **Limitations** | Synthetic data has programmatic fraud patterns with no novel/evolving attack types. Anomaly detection's real-world value is in detecting novel fraud patterns absent from training data -- a scenario the synthetic dataset cannot test by design. This refutation is specific to the synthetic-data regime; real data with concept drift may yield different results. |

---

## Resolution Key

| Tag | Meaning |
|-----|---------|
| **SUPPORTED** | Evidence confirms hypothesis at stated threshold (on synthetic data) |
| **REFUTED** | Evidence contradicts hypothesis at stated threshold |
| **INCONCLUSIVE** | Evidence is mixed or insufficient |
| **PENDING** | Not yet tested |

## Cross-Reference

| Hypothesis | FINDINGS.md Section | Key Output File |
|-----------|--------------------|--------------------|
| H-1 | RQ1: ML vs Rule-Based | `outputs/baselines/summary_seed42.json` |
| H-2 | RQ3: CFA Features Dominate SHAP Rankings | `outputs/baselines/summary_seed42.json` (shap field) |
| H-3 | RQ4: Controllability Analysis | `outputs/baselines/summary_seed42.json` (controllability field) |
| H-4 | RQ2: Anomaly Detection | `outputs/baselines/summary_seed42.json` (isolation_forest field) |
