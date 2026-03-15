# Conference Abstract — BSides / FS-ISAC

## Title
The CFA × ML Fraud Detector: Domain Expertise as Feature Engineering and the 81% Adversary-Resistant Floor

## Abstract (250 words)

We present an ML fraud detection pipeline that combines CFA (Chartered Financial Analyst) domain expertise with gradient boosting to achieve 0.987 AUC-ROC on 100K financial transactions. Our most significant finding: CFA-informed rule-based scoring alone achieves 0.898 AUC — ML adds 8.9 percentage points, but domain expertise embedded in rules captures 91% of the signal.

SHAP explainability analysis reveals that **8 of the top 20 predictive features are CFA-informed**: amount-to-median ratios (transaction size relative to merchant norms), merchant risk tiers, privacy-focused email detection, and suspicious timing patterns. These domain-engineered features outperform raw statistical equivalents — the CFA-informed `amt_to_median_ratio` (SHAP rank 4) is more predictive than raw `TransactionAmt` (rank 7).

We apply **adversarial controllability analysis** — classifying fraud detection features by who controls them (fraudster vs. system) — and demonstrate an **81% adversary-resistant detection floor**. Even if a sophisticated fraudster perfectly manipulates all controllable features (amount, timing, email, billing country), system-controlled features (device fingerprint, card BIN, merchant risk, address verification) retain 81% of model performance.

This is the fifth domain validation of controllability analysis as a general security architecture methodology, following network IDS, vulnerability prediction, AI agent red-teaming, and post-quantum cryptography migration. The principle holds: classifying inputs by who controls them determines what defense is possible.

All code and results are open source.

## Keywords
fraud detection, CFA, financial anomaly detection, SHAP explainability, adversarial controllability, feature engineering, XGBoost

## Bio
Rex Coleman, CFA, is an MS Computer Science student (Machine Learning) at Georgia Tech. Previously 15 years in cybersecurity (FireEye/Mandiant — analytics, enterprise sales, cross-functional leadership). Creator of govML (open-source ML governance framework).
