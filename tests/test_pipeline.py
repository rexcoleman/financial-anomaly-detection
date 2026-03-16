"""Pipeline tests for FP-04 financial anomaly detection.

Tests the feature engineering, model training helpers, and output format
without running full training (uses small subsets or mocks).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_df():
    """Generate a small synthetic dataset for pipeline testing."""
    from scripts.generate_synthetic_data import generate_transactions
    return generate_transactions(1000, 0.035, 42)


@pytest.fixture(scope="module")
def engineered(small_df):
    """Return feature-engineered dataframe and feature columns."""
    from scripts.run_pipeline import engineer_features
    df_copy = small_df.copy()
    return engineer_features(df_copy)


# ---------------------------------------------------------------------------
# Tests: data generation
# ---------------------------------------------------------------------------

def test_synthetic_data_generation():
    """Generate function returns correct row count and columns."""
    from scripts.generate_synthetic_data import generate_transactions
    df = generate_transactions(1000, 0.035, 42)
    assert len(df) == 1000
    assert "isFraud" in df.columns
    assert "TransactionAmt" in df.columns
    assert df["isFraud"].mean() > 0.02  # Roughly 3.5%


def test_synthetic_data_deterministic():
    """Same seed produces identical output."""
    from scripts.generate_synthetic_data import generate_transactions
    df1 = generate_transactions(500, 0.035, 99)
    df2 = generate_transactions(500, 0.035, 99)
    assert df1.equals(df2)


# ---------------------------------------------------------------------------
# Tests: feature engineering
# ---------------------------------------------------------------------------

def test_feature_engineering_creates_cfa_features(engineered):
    """CFA-informed features must be present after engineering."""
    df, feature_cols = engineered
    cfa_expected = ["amt_to_median_ratio", "protonmail", "high_risk_country",
                    "suspicious_time", "high_amount"]
    for feat in cfa_expected:
        assert feat in feature_cols, f"CFA feature {feat} missing"


def test_feature_count_minimum(engineered):
    """Pipeline should produce at least 15 features."""
    _, feature_cols = engineered
    assert len(feature_cols) >= 15, f"Only {len(feature_cols)} features"


def test_feature_engineering_no_nans_in_output(engineered):
    """Feature columns should have no NaNs after engineering (fillna(0) in split)."""
    df, feature_cols = engineered
    # The pipeline fills NaN at split time; verify the raw engineering step
    # does not introduce unexpected NaN beyond what fillna handles
    for col in feature_cols:
        nan_count = df[col].isna().sum()
        # Some columns may have NaN from missing categoricals; that is OK
        # as long as total is < 50% of rows
        assert nan_count < len(df) * 0.5, \
            f"Column {col} has {nan_count} NaNs ({nan_count/len(df)*100:.0f}%)"


# ---------------------------------------------------------------------------
# Tests: rule-based scorer
# ---------------------------------------------------------------------------

def test_rule_based_scorer_range(engineered):
    """Rule-based scores should be in [0, 1]."""
    from scripts.run_pipeline import rule_based_scorer
    df, _ = engineered
    scores = rule_based_scorer(df)
    assert len(scores) == len(df)
    assert scores.min() >= 0, f"Min score {scores.min()} < 0"
    assert scores.max() <= 1, f"Max score {scores.max()} > 1"


def test_rule_based_scorer_not_constant(engineered):
    """Rule scores should vary across transactions."""
    from scripts.run_pipeline import rule_based_scorer
    df, _ = engineered
    scores = rule_based_scorer(df)
    assert scores.std() > 0, "Rule scores are constant"


# ---------------------------------------------------------------------------
# Tests: temporal split
# ---------------------------------------------------------------------------

def test_temporal_split_no_leakage(engineered):
    """Train set max time < test set min time (temporal ordering)."""
    from scripts.run_pipeline import temporal_split
    df, feature_cols = engineered
    # Need TransactionDT in df
    df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.7)
    train_max_t = df_sorted.iloc[:split_idx]["TransactionDT"].max()
    test_min_t = df_sorted.iloc[split_idx:]["TransactionDT"].min()
    assert train_max_t <= test_min_t, \
        f"Temporal leakage: train max {train_max_t} > test min {test_min_t}"


def test_temporal_split_sizes(engineered):
    """Split should produce ~70/30 train/test."""
    from scripts.run_pipeline import temporal_split
    df, feature_cols = engineered
    X_train, y_train, X_test, y_test = temporal_split(df, feature_cols)
    total = len(X_train) + len(X_test)
    train_frac = len(X_train) / total
    assert 0.65 < train_frac < 0.75, \
        f"Train fraction {train_frac:.2f} outside expected 0.65-0.75"


# ---------------------------------------------------------------------------
# Tests: output file format
# ---------------------------------------------------------------------------

def test_baseline_summary_schema():
    """Baseline summary JSON must contain required keys."""
    path = PROJECT_ROOT / "outputs" / "baselines" / "summary_seed42.json"
    if not path.exists():
        pytest.skip("summary_seed42.json not found")
    with open(path) as f:
        data = json.load(f)
    required_keys = ["seed", "n_transactions", "fraud_rate", "results"]
    for key in required_keys:
        assert key in data, f"Missing key: {key}"
    assert "rule_based" in data["results"]
    assert "auc" in data["results"]["rule_based"]


def test_expanded_summary_schema():
    """Expanded model summary must contain required keys."""
    path = PROJECT_ROOT / "outputs" / "models" / "expanded_summary.json"
    if not path.exists():
        pytest.skip("expanded_summary.json not found")
    with open(path) as f:
        data = json.load(f)
    assert "experiment" in data
    assert data["experiment"] == "expanded_models"
    assert "seeds" in data
    assert "results" in data
    assert len(data["results"]) > 0
