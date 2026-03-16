# FIGURES & TABLES CONTRACT

<!-- version: 1.0 -->
<!-- created: 2026-03-16 -->

## 1) Purpose

Defines all report-ready figures and tables, their data sources, and generation scripts.

## 2) Required Figures

| ID | Title | Source Data | Script |
|----|-------|-------------|--------|
| F-1 | Model Comparison (AUC bar chart) | `outputs/models/expanded_summary.json` | `scripts/make_report_figures.py` |
| F-2 | Complexity Curves (train vs test AUC) | `outputs/diagnostics/complexity_curves_summary.json` | `scripts/make_report_figures.py` |
| F-3 | SHAP Feature Importance (top-20 beeswarm) | `outputs/baselines/summary_seed42.json` | `scripts/make_report_figures.py` |
| F-4 | Controllability Analysis (system vs full AUC) | `outputs/baselines/summary_seed42.json` | `scripts/make_report_figures.py` |

## 3) Required Tables

| ID | Title | Source Data |
|----|-------|-------------|
| T-1 | Model performance summary (AUC, F1, precision, recall) | `outputs/models/expanded_summary.json` |
| T-2 | Complexity curve optimal points | `outputs/diagnostics/complexity_curves_summary.json` |
| T-3 | Hypothesis resolution summary | `docs/HYPOTHESIS_REGISTRY.md` |

## 4) Data Flow

All figures and tables are generated from JSON output files. No figure reads raw data directly.

```
outputs/*.json → scripts/make_report_figures.py → outputs/figures/*.png
                                                → outputs/tables/*.csv
```

## 5) Determinism

- Figures use `matplotlib` with fixed seeds and no random jitter
- All numeric values formatted to consistent precision (3 decimal places)
- Re-running `make_report_figures.py` produces byte-identical output

## 6) Output Paths

```
outputs/figures/
├── model_comparison.png
├── complexity_curves.png
├── shap_importance.png
└── controllability_analysis.png

outputs/tables/
├── model_summary.csv
├── complexity_optimal.csv
└── hypothesis_summary.csv
```

## 7) Change Control

Adding or removing a figure/table requires a `CONTRACT_CHANGE` commit.
