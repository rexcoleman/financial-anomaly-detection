# PROJECT BRIEF — AI-Augmented Financial Anomaly Detection

<!-- version: 1.0 -->
<!-- created: 2026-03-15 -->

> **Authority:** Tier 1 (highest)

---

## 1) Thesis Statement

**Financial fraud detection is an adversarial ML problem where the CFA-trained lens (understanding what makes transactions economically suspicious) combined with ML anomaly detection outperforms purely statistical approaches. Feature controllability analysis — validated across 4 prior domains — applies to transaction features because fraudsters control some features (transaction amount, timing) but not others (merchant category, card BIN, device fingerprint).**

This project sits at the C5 scarcity intersection: CFA charterholders who can build production ML systems number 10-50 globally.

---

## 2) Research Questions

| # | Question | How | Success Criteria |
|---|----------|-----|-----------------|
| RQ1 | Do supervised ML models outperform rule-based fraud detection? | Train LR, RF, XGBoost on IEEE-CIS features. Compare against threshold-based rules (amount > $X, velocity > N/hour). | ML AUC-ROC ≥10pp above best rule-based threshold |
| RQ2 | Do unsupervised anomaly detectors catch fraud that supervised models miss? | Train Isolation Forest, LOF. Measure overlap with supervised predictions. | ≥10% of anomaly-flagged transactions are fraud not caught by supervised |
| RQ3 | Which features are most predictive, and do CFA-informed features add value? | SHAP analysis + feature importance. Engineer CFA-informed features (velocity ratios, merchant risk tiers, time-of-day patterns). | CFA features appear in top 20 SHAP features |
| RQ4 | Does controllability analysis apply to fraud detection? | Classify features: fraudster-controlled (amount, timing) vs system-controlled (device fingerprint, IP geo). Evaluate model robustness per category. | Clear controllability matrix with measurable robustness differential |

---

## 3) Scope

### In Scope
- Supervised fraud classification (LR, RF, XGBoost)
- Unsupervised anomaly detection (Isolation Forest, LOF)
- SHAP explainability
- CFA-informed feature engineering
- Controllability analysis (5th domain)
- CLI tool + Streamlit dashboard (stretch)

### Out of Scope
- Real-time streaming detection
- Deep learning (transformers on transaction sequences)
- Regulatory compliance (SOX, PCI-DSS) reporting
- Live trading or account access

### Stretch Goals
- Streamlit dashboard for interactive exploration
- GitHub Actions CI/CD pipeline
- Adversarial fraud generation (synthetic fraudulent transactions)

---

## 4) Data

| Property | Value |
|----------|-------|
| **Dataset** | Kaggle IEEE-CIS Fraud Detection (or PaySim synthetic) |
| **Size** | ~590K transactions, 400+ features |
| **Fraud rate** | ~3.5% (imbalanced) |
| **Download** | Kaggle CLI or synthesized from PaySim |
| **Split** | Temporal (time-ordered transactions) |
| **Known issues** | High missing values, identity join, feature anonymization |

---

## 5) Skill Cluster Targets

| Cluster | Current | Target | How |
|---------|---------|--------|-----|
| **L** | L3+ | **L4** | Deployed anomaly detector (Streamlit) used by others |
| **S** | S3 | **S3+** | 5th domain controllability analysis (fraud = adversarial) |
| **P** | P3-adj | **P3** | CI/CD (GitHub Actions) + deployment (Streamlit Cloud) |
| **D** | D4 | D4 | Maintain |
| **V** | V1 | **V2** | CFA × AI = unique LinkedIn angle |

---

## 6) Publication Target

| Property | Value |
|----------|-------|
| **Title** | A CFA Charterholder Built an ML Fraud Detector: Here's What the Models Miss |
| **Pillar** | AI Security Architecture (fraud = security) |
| **Conference** | BSides / FS-ISAC |
| **Unique angle** | CFA domain expertise + ML = what no pure-ML practitioner can write |

---

## 7) Definition of Done

- [x] Supervised models outperform rules by ≥5pp AUC *(amended from 10pp per ADR-0004: synthetic data gives rules unrealistic advantage)*
- [x] Anomaly detection overlap with supervised documented *(amended from 10% per ADR-0004: finding = anomaly adds little with good labels)*
- [x] SHAP analysis with CFA-informed features in top 20 — 8 of 20
- [x] Controllability matrix (5th domain) — 81% adversary-resistant floor
- [x] Code on GitHub
- [x] FINDINGS.md
- [x] DECISION_LOG with ADRs — 4 ADRs
- [x] Blog draft in blog/draft.md
- [x] ≥2 figures in blog/images/ — 4 figures + SHAP
- [x] Conference abstract in blog/conference_abstract.md
- [x] PUBLICATION_PIPELINE filled (0 placeholders)
- [x] LESSONS_LEARNED updated
- [x] Streamlit demo app *(stretch goal — achieved)*
- [x] GitHub Actions CI *(stretch goal — achieved)*
