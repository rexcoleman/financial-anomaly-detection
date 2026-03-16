#!/usr/bin/env bash
# reproduce.sh — Full reproduction pipeline for FP-04 Financial Anomaly Detection
#
# Prerequisites:
#   conda env create -f environment.yml   # creates 'fin-anomaly' environment
#   pip install lightgbm                  # optional: for LightGBM models
#
# Usage:
#   bash reproduce.sh          # full pipeline (all seeds, ~2 hours with SVM)
#   bash reproduce.sh --quick  # smoke test (seed 42 only, ~20 min)
#
# Output:
#   outputs/baselines/summary_seed{42,123}.json
#   outputs/models/expanded_seed{42,...,1024}.json
#   outputs/models/expanded_summary.json
#   outputs/diagnostics/sanity_baselines_seed{42,...,1024}.json
#   outputs/diagnostics/learning_curves_seed{42,...,1024}.json
#   outputs/diagnostics/learning_curves_summary.json
#   outputs/diagnostics/complexity_curves_seed{42,...,1024}.json
#   outputs/diagnostics/complexity_curves_summary.json
#   outputs/explainability/shap_summary.png
#   outputs/figures/model_comparison.png
#   outputs/figures/controllability.png

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate fin-anomaly

echo "============================================================"
echo "FP-04: Financial Anomaly Detection — Reproduction Pipeline"
echo "============================================================"
echo "Started: $(date)"
echo ""

# Parse args
QUICK=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK=true
    echo "Mode: QUICK (seed 42 only)"
else
    echo "Mode: FULL (5 seeds)"
fi
echo ""

# Step 0: Generate synthetic data (if not present)
echo "--- Step 0: Generate synthetic data ---"
if [[ ! -f data/raw/transactions.csv ]]; then
    python scripts/generate_synthetic_data.py \
        --n-transactions 100000 \
        --fraud-rate 0.035 \
        --seed 42 \
        --output data/raw/transactions.csv
else
    echo "  data/raw/transactions.csv already exists, skipping."
fi
echo ""

# Step 1: Run main pipeline (5 seeds)
echo "--- Step 1: Main pipeline (RQ1-RQ4) ---"
if [[ "$QUICK" == true ]]; then
    python scripts/run_pipeline.py --seed 42
else
    for seed in 42 123 456 789 1024; do
        python scripts/run_pipeline.py --seed "$seed"
    done
fi
echo ""

# Step 2: Sanity baselines
echo "--- Step 2: Sanity baselines ---"
if [[ "$QUICK" == true ]]; then
    python scripts/run_sanity_baselines.py --seeds 42
else
    python scripts/run_sanity_baselines.py
fi
echo ""

# Step 3: Learning curves
echo "--- Step 3: Learning curves ---"
if [[ "$QUICK" == true ]]; then
    python scripts/run_learning_curves.py --seeds 42
else
    python scripts/run_learning_curves.py
fi
echo ""

# Step 4: Expanded models (SVM-RBF + LightGBM)
echo "--- Step 4: Expanded models (SVM-RBF + LightGBM) ---"
if [[ "$QUICK" == true ]]; then
    python scripts/train_expanded_models.py --seeds 42
else
    python scripts/train_expanded_models.py
fi
echo ""

# Step 5: Complexity curves
echo "--- Step 5: Complexity curves ---"
if [[ "$QUICK" == true ]]; then
    python scripts/run_complexity_curves.py --seeds 42
else
    python scripts/run_complexity_curves.py
fi
echo ""

# Step 6: Generate figures
echo "--- Step 6: Generate figures ---"
if [[ -f scripts/generate_figures.py ]]; then
    python scripts/generate_figures.py
fi
echo ""

# Step 7: Run tests
echo "--- Step 7: Run test suite ---"
python -m pytest tests/ -v --tb=short
echo ""

echo "============================================================"
echo "Reproduction complete: $(date)"
echo "============================================================"
