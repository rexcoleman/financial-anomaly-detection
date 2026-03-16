"""Sanity tests for FP-04 financial anomaly detection.

Validates that models genuinely learn signal: they must beat dummy
classifiers, beat shuffled-label controls, produce AUC > 0.5, and
produce non-constant predictions. Tests read from pre-computed output
files to avoid re-training.
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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sanity_seed42():
    """Load sanity baselines for seed 42."""
    path = DIAGNOSTICS_DIR / "sanity_baselines_seed42.json"
    if not path.exists():
        pytest.skip("sanity_baselines_seed42.json not found")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def summary_seed42():
    """Load pipeline summary for seed 42."""
    path = BASELINES_DIR / "summary_seed42.json"
    if not path.exists():
        pytest.skip("summary_seed42.json not found")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def expanded_summary():
    """Load expanded models summary."""
    path = MODELS_DIR / "expanded_summary.json"
    if not path.exists():
        pytest.skip("expanded_summary.json not found")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def learning_curves():
    """Load learning curves summary."""
    path = DIAGNOSTICS_DIR / "learning_curves_summary.json"
    if not path.exists():
        pytest.skip("learning_curves_summary.json not found")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def complexity_curves():
    """Load complexity curves summary."""
    path = DIAGNOSTICS_DIR / "complexity_curves_summary.json"
    if not path.exists():
        pytest.skip("complexity_curves_summary.json not found")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tests: beats dummy baselines
# ---------------------------------------------------------------------------

def test_real_model_beats_dummy_most_frequent(sanity_seed42):
    """Real RF F1 must exceed dummy most_frequent F1."""
    rf_f1 = sanity_seed42["baselines"]["rf_real"]["f1"]
    dummy_f1 = sanity_seed42["baselines"]["dummy_most_frequent"]["f1"]
    assert rf_f1 > dummy_f1, \
        f"RF F1 {rf_f1} does not beat dummy {dummy_f1}"


def test_real_model_beats_dummy_stratified(sanity_seed42):
    """Real RF AUC must exceed stratified dummy AUC by > 0.1."""
    rf_auc = sanity_seed42["baselines"]["rf_real"]["auc"]
    dummy_auc = sanity_seed42["baselines"]["dummy_stratified"]["auc"]
    gap = rf_auc - dummy_auc
    assert gap > 0.1, \
        f"RF AUC {rf_auc} only {gap:.4f} above dummy {dummy_auc}"


def test_shuffled_label_model_near_chance(sanity_seed42):
    """RF trained on shuffled labels should have AUC near 0.5."""
    shuf_auc = sanity_seed42["baselines"]["rf_shuffled"]["auc"]
    assert abs(shuf_auc - 0.5) < 0.05, \
        f"Shuffled-label AUC {shuf_auc} too far from chance (0.5)"


def test_sanity_gap_passes(sanity_seed42):
    """The pre-computed sanity gap should pass (>0.1)."""
    assert sanity_seed42["sanity_pass"] is True, \
        f"Sanity gap {sanity_seed42['sanity_gap_auc']} did not pass"


# ---------------------------------------------------------------------------
# Tests: AUC above chance for all models
# ---------------------------------------------------------------------------

def test_xgboost_auc_above_chance(summary_seed42):
    """XGBoost AUC must be > 0.5."""
    auc = summary_seed42["results"]["XGBoost"]["auc"]
    assert auc > 0.5, f"XGBoost AUC {auc} not above chance"


def test_logistic_regression_auc_above_chance(summary_seed42):
    """LogisticRegression AUC must be > 0.5."""
    auc = summary_seed42["results"]["LogisticRegression"]["auc"]
    assert auc > 0.5, f"LR AUC {auc} not above chance"


def test_random_forest_auc_above_chance(summary_seed42):
    """RandomForest AUC must be > 0.5."""
    auc = summary_seed42["results"]["RandomForest"]["auc"]
    assert auc > 0.5, f"RF AUC {auc} not above chance"


def test_svm_rbf_auc_above_chance(expanded_summary):
    """SVM-RBF AUC must be > 0.5 (from expanded models)."""
    auc_mean = expanded_summary["SVM-RBF_auc_mean"]
    assert auc_mean > 0.5, f"SVM-RBF mean AUC {auc_mean} not above chance"


def test_lightgbm_auc_above_chance(expanded_summary):
    """LightGBM AUC must be > 0.5 (from expanded models)."""
    auc_mean = expanded_summary["LightGBM_auc_mean"]
    assert auc_mean > 0.5, f"LightGBM mean AUC {auc_mean} not above chance"


# ---------------------------------------------------------------------------
# Tests: model ranking
# ---------------------------------------------------------------------------

def test_xgboost_beats_rule_based(summary_seed42):
    """XGBoost must beat rule-based baseline."""
    xgb_auc = summary_seed42["results"]["XGBoost"]["auc"]
    rule_auc = summary_seed42["results"]["rule_based"]["auc"]
    assert xgb_auc > rule_auc, \
        f"XGBoost {xgb_auc} does not beat rules {rule_auc}"


# ---------------------------------------------------------------------------
# Tests: complexity curves overfitting detection
# ---------------------------------------------------------------------------

def test_xgboost_overfits_at_high_depth(complexity_curves):
    """XGBoost test AUC should decrease as max_depth increases."""
    test_means = complexity_curves["XGBoost_test_auc_mean"]
    # Best at lowest depth, worst at highest depth
    assert test_means[0] > test_means[-1], \
        f"XGBoost test AUC does not decrease: {test_means}"


def test_svm_overfits_at_high_c(complexity_curves):
    """SVM-RBF test AUC should decrease as C increases past optimal."""
    test_means = complexity_curves["SVM-RBF_test_auc_mean"]
    assert test_means[0] > test_means[-1], \
        f"SVM test AUC does not decrease with C: {test_means}"


# ---------------------------------------------------------------------------
# Tests: learning curve monotonicity
# ---------------------------------------------------------------------------

def test_xgboost_learning_curve_monotonic(learning_curves):
    """XGBoost test AUC should generally increase with more data."""
    test_means = learning_curves["XGBoost_test_auc_mean"]
    # First value should be less than last (more data = better)
    assert test_means[0] <= test_means[-1], \
        f"XGBoost learning curve not improving: {test_means}"
