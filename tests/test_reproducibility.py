"""Reproducibility tests for FP-04 financial anomaly detection.

Validates that multi-seed experiments produce consistent results and
that output files for all seeds exist with matching schemas.
"""
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
BASELINES_DIR = PROJECT_ROOT / "outputs" / "baselines"
DIAGNOSTICS_DIR = PROJECT_ROOT / "outputs" / "diagnostics"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

SEEDS = [42, 123, 456, 789, 1024]


# ---------------------------------------------------------------------------
# Tests: output files exist for all seeds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_sanity_baselines_exist(seed):
    """Sanity baseline file must exist for each seed."""
    path = DIAGNOSTICS_DIR / f"sanity_baselines_seed{seed}.json"
    assert path.exists(), f"Missing: {path}"


@pytest.mark.parametrize("seed", SEEDS)
def test_learning_curves_exist(seed):
    """Learning curves file must exist for each seed."""
    path = DIAGNOSTICS_DIR / f"learning_curves_seed{seed}.json"
    assert path.exists(), f"Missing: {path}"


@pytest.mark.parametrize("seed", SEEDS)
def test_expanded_model_results_exist(seed):
    """Expanded model results must exist for each seed."""
    path = MODELS_DIR / f"expanded_seed{seed}.json"
    assert path.exists(), f"Missing: {path}"


@pytest.mark.parametrize("seed", SEEDS)
def test_complexity_curves_exist(seed):
    """Complexity curves file must exist for each seed."""
    path = DIAGNOSTICS_DIR / f"complexity_curves_seed{seed}.json"
    assert path.exists(), f"Missing: {path}"


# ---------------------------------------------------------------------------
# Tests: cross-seed consistency (low variance)
# ---------------------------------------------------------------------------

def test_expanded_svm_rbf_low_variance():
    """SVM-RBF AUC variance across 5 seeds should be < 0.01."""
    path = MODELS_DIR / "expanded_summary.json"
    if not path.exists():
        pytest.skip("expanded_summary.json not found")
    with open(path) as f:
        data = json.load(f)
    std = data["SVM-RBF_auc_std"]
    assert std < 0.01, f"SVM-RBF AUC std {std} >= 0.01"


def test_expanded_lightgbm_low_variance():
    """LightGBM AUC variance across 5 seeds should be < 0.01."""
    path = MODELS_DIR / "expanded_summary.json"
    if not path.exists():
        pytest.skip("expanded_summary.json not found")
    with open(path) as f:
        data = json.load(f)
    std = data["LightGBM_auc_std"]
    assert std < 0.01, f"LightGBM AUC std {std} >= 0.01"


# ---------------------------------------------------------------------------
# Tests: deterministic data generation
# ---------------------------------------------------------------------------

def test_generate_transactions_deterministic():
    """Same seed must produce identical dataframe."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.generate_synthetic_data import generate_transactions

    df1 = generate_transactions(500, 0.035, 42)
    df2 = generate_transactions(500, 0.035, 42)
    assert df1.shape == df2.shape
    # Compare numeric columns
    for col in df1.select_dtypes(include=["float64", "int64"]).columns:
        assert (df1[col] == df2[col]).all(), f"Column {col} differs between runs"


def test_different_seeds_produce_different_data():
    """Different seeds must produce different dataframes."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.generate_synthetic_data import generate_transactions

    df1 = generate_transactions(500, 0.035, 42)
    df2 = generate_transactions(500, 0.035, 99)
    # At least the amounts should differ
    assert not (df1["TransactionAmt"] == df2["TransactionAmt"]).all(), \
        "Different seeds produced identical TransactionAmt"


# ---------------------------------------------------------------------------
# Tests: summary file schemas match across seeds
# ---------------------------------------------------------------------------

def test_sanity_baseline_schema_consistent():
    """All sanity baseline files should have the same top-level keys."""
    keys_by_seed = {}
    for seed in SEEDS:
        path = DIAGNOSTICS_DIR / f"sanity_baselines_seed{seed}.json"
        if not path.exists():
            pytest.skip(f"Missing {path}")
        with open(path) as f:
            data = json.load(f)
        keys_by_seed[seed] = set(data.keys())
    # All seeds should have the same keys
    reference = keys_by_seed[SEEDS[0]]
    for seed in SEEDS[1:]:
        assert keys_by_seed[seed] == reference, \
            f"Seed {seed} keys {keys_by_seed[seed]} differ from seed {SEEDS[0]} keys {reference}"


def test_expanded_per_seed_result_count():
    """Each expanded seed file should have results for both SVM-RBF and LightGBM."""
    for seed in SEEDS:
        path = MODELS_DIR / f"expanded_seed{seed}.json"
        if not path.exists():
            pytest.skip(f"Missing {path}")
        with open(path) as f:
            data = json.load(f)
        model_names = [r["model"] for r in data["results"]]
        assert "SVM-RBF" in model_names, f"Seed {seed}: SVM-RBF missing"
        assert "LightGBM" in model_names, f"Seed {seed}: LightGBM missing"
