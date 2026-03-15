# AI-Augmented Financial Anomaly Detection — Claude Code Context

> **govML v2.5** | Profile: blog-track (blog-track)

## Project Purpose

A CFA Charterholder Built an ML Fraud Detector: Here's What the Models Miss

- **Context:** Self-directed research (AI-Augmented Financial Anomaly Detection)
- **Profile:** blog-track
- **Python:** 3.11 | **Env:** fin-anomaly
- **Brand pillar:** AI Security Architecture
- **Workload type:** cpu_bound

## Authority Hierarchy

| Tier | Source | Path |
|------|--------|------|
| 1 (highest) | Project Brief | `docs/PROJECT_BRIEF.md` |
| 2 | — | No external FAQ |
| 3 | Advisory methodology | `docs/ADVERSARIAL_EVALUATION.md` |
| Contracts | Governance docs | `docs/*.md` |

## Current Phase

**Phase:** 0 — Environment & Setup

### Phase Progression

| Phase | Name | Status |
|-------|------|--------|
| 0 | Phase 0 — Environment & Data | **CURRENT** |
| 1 | Phase 1 — EDA & Feature Engineering | Not started |
| 2 | Phase 2 — Models & Explainability | Not started |
| 3 | Phase 3 — Adversarial Eval & Findings | Not started |
| 4 | Phase 4 — Publication Artifacts | Not started |

## Experiment Summary

Seeds: [42, 123, 456]

- **baselines:** logistic_regression, random_forest, xgboost
- **anomaly_detection:** isolation_forest, local_outlier_factor, autoencoder
- **explainability:** shap, feature_importance
- **adversarial_eval:** feature_controllability

## Key Files

| File | Purpose |
|------|---------|
| `docs/PROJECT_BRIEF.md` | **READ FIRST** — thesis, RQs, scope |
| `docs/PUBLICATION_PIPELINE.md` | Blog post governance + distribution |
| `docs/DECISION_LOG.md` | All tradeoff decisions (mandatory at every phase gate) |
| `config/base.yaml` | Experiment configuration |

## AI Division of Labor

### Permitted
- **Claude Code:** Coding copilot, feature engineering, figure generation
  - Prohibited: Must not interpret financial risk implications (CFA judgment). Must not generate trading signals.

### Prohibited (all projects)
- Modifying PROJECT_BRIEF thesis or research questions
- Writing interpretation/analysis prose (human insight)

## Conventions

- **Seeds:** [42, 123, 456]
- **Smoke test first:** `--sample-frac 0.01` or `--dry-run` before full runs
- **Decisions:** Log in DECISION_LOG at every phase gate (mandatory per v2.5)
- **Commit early:** Phase 0a scaffold → commit → Phase 0b research → commit
