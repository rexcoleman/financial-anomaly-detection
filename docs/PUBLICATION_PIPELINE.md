# PUBLICATION PIPELINE — AI-Augmented Financial Anomaly Detection

<!-- version: 2.0 -->
<!-- created: 2026-03-15 -->

> **Authority:** Subordinate to PROJECT_BRIEF (Tier 1)

---

## 1) Target Venue
- [x] Blog (Hugo canonical)
- [x] Conference CFP: BSides / FS-ISAC
- [x] LinkedIn (CFA community — unique angle)

## 2) Content Identity

| Property | Value |
|----------|-------|
| **Working title** | A CFA Charterholder Built an ML Fraud Detector: Here's What the Models Miss |
| **Content pillar** | AI Security Architecture (fraud = adversarial security) |
| **Target audience** | P1: CFA charterholders curious about AI. P2: Security engineers building fraud detection. P3: AI hiring managers (CFA × ML signal). |
| **One-line thesis** | CFA-informed feature engineering captures 91% of ML's fraud detection signal; controllability analysis proves an 81% adversary-resistant floor from system features alone. |
| **What was shipped** | github.com/rexcoleman/financial-anomaly-detection |

### Voice Check
| Test | Pass? |
|------|-------|
| References something you built | [x] Pipeline, SHAP analysis, controllability matrix |
| Shows work (code, data, architecture) | [x] 100K transactions, XGBoost, SHAP top 20 |
| Avoids pundit framing | [x] "Here's What the Models Miss" = showing work |
| Architecture diagram | [x] Cross-domain ACA chart (5 domains) |
| Links to repo | [x] |

## 4) Evidence Inventory

| Claim | Evidence | Source |
|-------|---------|-------|
| XGBoost AUC 0.987 | Model output | `outputs/baselines/summary_seed42.json` |
| Rule-based AUC 0.898 | Same | Same |
| +8.9pp improvement | Same | Same |
| 8 CFA features in top 20 SHAP | SHAP analysis | Same |
| 81% adversary-resistant floor | Controllability | Same |
| 5th domain ACA validation | Cross-project | FINDINGS.md |

## 5) Distribution Checklist
### 5.1 Pre-Publication
- [x] Builder voice passes
- [x] Figures finalized (3 charts + SHAP)
- [x] All claims traceable
- [ ] No anti-claims (grep check)
### 5.2-5.4
- [ ] Hugo, Substack, dev.to, Hashnode, LinkedIn, HN (pending brand infra)
