"""Data integrity tests for FP-04 financial anomaly detection.

Validates that the synthetic dataset and output artifacts conform to
expected schemas, distributions, and constraints.
"""
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
BASELINES_DIR = PROJECT_ROOT / "outputs" / "baselines"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def transactions_df():
    """Load the raw transactions CSV."""
    import pandas as pd
    assert DATA_PATH.exists(), f"Data file not found: {DATA_PATH}"
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def summary_seed42():
    """Load the seed-42 baseline summary."""
    path = BASELINES_DIR / "summary_seed42.json"
    assert path.exists(), f"Summary not found: {path}"
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tests: schema and column presence
# ---------------------------------------------------------------------------

def test_required_columns_present(transactions_df):
    """Verify all columns needed by the pipeline exist."""
    required = [
        "TransactionID", "isFraud", "TransactionAmt", "TransactionDT",
        "card_type", "DeviceType", "DeviceInfo", "email_domain",
        "billing_country", "merchant_risk_score", "addr_mismatch",
        "hour_of_day", "day_of_week", "is_weekend", "is_night",
        "amt_to_median_ratio", "amt_log",
    ]
    missing = [c for c in required if c not in transactions_df.columns]
    assert missing == [], f"Missing columns: {missing}"


def test_row_count(transactions_df):
    """Dataset should have 100K transactions."""
    assert len(transactions_df) == 100_000


def test_label_is_binary(transactions_df):
    """isFraud must be binary (0 or 1)."""
    unique_vals = set(transactions_df["isFraud"].unique())
    assert unique_vals.issubset({0, 1}), f"Unexpected label values: {unique_vals}"


def test_fraud_rate_in_range(transactions_df):
    """Fraud rate should be approximately 3.5% (within 1-6%)."""
    fraud_rate = transactions_df["isFraud"].mean()
    assert 0.01 < fraud_rate < 0.06, f"Fraud rate {fraud_rate:.4f} out of expected range"


def test_transaction_amount_positive(transactions_df):
    """All transaction amounts must be positive."""
    assert (transactions_df["TransactionAmt"] > 0).all(), \
        "Found non-positive transaction amounts"


def test_temporal_ordering_possible(transactions_df):
    """TransactionDT column must exist and be numeric for temporal split."""
    assert transactions_df["TransactionDT"].dtype in ["float64", "int64"], \
        f"TransactionDT has wrong dtype: {transactions_df['TransactionDT'].dtype}"
    assert transactions_df["TransactionDT"].nunique() > 1, \
        "TransactionDT has no variation — temporal split would fail"


def test_feature_types_numeric(transactions_df):
    """Numeric feature columns must have numeric dtypes."""
    numeric_cols = [
        "TransactionAmt", "merchant_risk_score", "addr_mismatch",
        "hour_of_day", "day_of_week", "is_weekend", "is_night",
        "amt_to_median_ratio", "amt_log",
    ]
    for col in numeric_cols:
        if col in transactions_df.columns:
            assert transactions_df[col].dtype.kind in ("f", "i"), \
                f"Column {col} has non-numeric dtype: {transactions_df[col].dtype}"


def test_no_duplicate_transaction_ids(transactions_df):
    """TransactionID should be unique."""
    assert transactions_df["TransactionID"].is_unique, \
        "Duplicate TransactionIDs found"
