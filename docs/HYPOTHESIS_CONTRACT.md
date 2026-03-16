# HYPOTHESIS CONTRACT

<!-- version: 1.0 -->
<!-- created: 2026-03-16 -->

## 1) Purpose

Pre-registers hypotheses before experimentation and links them to the HYPOTHESIS_REGISTRY for resolution tracking.

## 2) Pre-Registered Hypotheses

| ID | Hypothesis | Metric | Threshold | Registry Status |
|----|-----------|--------|-----------|-----------------|
| H-1 | CFA-informed features capture significant signal in SHAP top-20 | Fraction of CFA features in SHAP top-20 | >30% | SUPPORTED, SYNTHETIC |
| H-2 | System-controlled features provide adversary-resistant detection floor | System-only AUC / Full AUC | >70% | SUPPORTED, SYNTHETIC |
| H-3 | Gradient-boosted ensembles outperform simpler models on fraud AUC | AUC-ROC pairwise comparison | XGB/LGBM > LR, RF, SVM | SUPPORTED, SYNTHETIC |
| H-4 | Complexity curves reveal optimal point beyond which capacity hurts | Train-test AUC gap monotonicity | Gap increases past optimal | SUPPORTED, SYNTHETIC |

## 3) Resolution Protocol

1. Hypotheses registered BEFORE running experiments (this document)
2. Results recorded in `docs/HYPOTHESIS_REGISTRY.md` with evidence links
3. Status tags: SUPPORTED, REFUTED, INCONCLUSIVE, PENDING
4. All results carry `SYNTHETIC` qualifier (PaySim data)

## 4) Evidence Requirements

Each hypothesis resolution MUST include:
- Quantitative result vs threshold
- Output file path containing the evidence
- Cross-reference to FINDINGS.md section

## 5) Post-Hoc Hypotheses

H-5 (anomaly detection) and H-6 (multi-seed stability) were added during experimentation. They are tagged as exploratory, not pre-registered.

## 6) Cross-References

- Full registry with outcomes: `docs/HYPOTHESIS_REGISTRY.md`
- Evidence narratives: `FINDINGS.md`
- Output artifacts: `outputs/baselines/`, `outputs/diagnostics/`, `outputs/models/`

## 7) Change Control

Adding or modifying a pre-registered hypothesis after experiments begin requires a disclosure in HYPOTHESIS_REGISTRY with `POST_HOC` tag.
