# AI-Augmented Financial Anomaly Detection

**CFA × ML: Domain expertise meets machine learning for fraud detection. Built by a CFA charterholder who writes Python.**

## Key Results

| Metric | Value |
|--------|-------|
| XGBoost AUC | 0.987 (+8.9pp over CFA rule-based baseline) |
| CFA Rule-Based AUC | 0.898 (surprisingly strong baseline) |
| CFA features in top 20 SHAP | 8 of 20 |
| Adversary-resistant floor | 81% (system-controlled features only) |
| Data | 100K synthetic financial transactions |

**Core insight:** Domain expertise encoded as rules is a floor, not a ceiling. ML adds non-linear interaction detection that rules miss, but the CFA-informed features dominate SHAP importance rankings.

## Quick Start

```bash
git clone https://github.com/rexcoleman/financial-anomaly-detection.git
cd financial-anomaly-detection
conda env create -f environment.yml
conda activate fin-anomaly

# Run full pipeline
python scripts/run_pipeline.py --seed 42

# Launch interactive dashboard
streamlit run app.py
```

## Architecture

```
src/
  detection/          # Fraud detection models (XGBoost, Isolation Forest)
  features/           # CFA-informed feature engineering
  explainability/     # SHAP analysis
  core/               # Types and utilities
scripts/
  run_pipeline.py               # Full pipeline: data -> features -> models -> SHAP -> ACA
  generate_synthetic_data.py    # PaySim-style synthetic transactions
  generate_figures.py           # Publication-ready charts
app.py                          # Streamlit interactive dashboard
```

## Methodology

This project validates the **adversarial controllability analysis** methodology (5th domain). Transaction features are classified by who controls them:
- **System-controlled:** account history, transaction frequency, institutional flags — 81% detection floor
- **Attacker-controlled:** transaction amount, timing, merchant category — manipulable by adversaries

See [FINDINGS.md](FINDINGS.md) for detailed results.

## Governed by [govML](https://github.com/rexcoleman/govML)

Built with reproducibility and decision traceability enforced across the entire pipeline.

## License

[MIT](LICENSE)
