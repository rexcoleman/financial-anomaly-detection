"""Controllability tests for FP-04 financial anomaly detection.

Validates the adversarial control analysis (ACA) framework:
feature controllability labels, system-only vs fraudster-only model
performance, category counts, and pipeline execution correctness.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASELINES_DIR = PROJECT_ROOT / "outputs" / "baselines"


# ---------------------------------------------------------------------------
# Constants: canonical controllability labels (from run_pipeline.py)
# ---------------------------------------------------------------------------

FRAUDSTER_CONTROLLED = [
    "TransactionAmt", "amt_log", "amt_to_median_ratio", "amt_zscore", "high_amount",
    "hour_of_day", "is_night", "suspicious_time",
    "email_domain_enc", "protonmail",
    "billing_country_enc", "high_risk_country",
]

SYSTEM_CONTROLLED = [
    "card_type_enc", "DeviceType_enc", "DeviceInfo_enc",
    "merchant_risk_score", "high_risk_merchant",
    "addr_mismatch",
]

TEMPORAL = [
    "day_of_week", "is_weekend",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_df():
    """Generate a small synthetic dataset for controllability testing."""
    from scripts.generate_synthetic_data import generate_transactions
    return generate_transactions(1000, 0.035, 42)


@pytest.fixture(scope="module")
def engineered(small_df):
    """Return feature-engineered dataframe and feature columns."""
    from scripts.run_pipeline import engineer_features
    df_copy = small_df.copy()
    return engineer_features(df_copy)


@pytest.fixture(scope="module")
def summary_seed42():
    """Load pipeline summary for seed 42 (if available)."""
    path = BASELINES_DIR / "summary_seed42.json"
    if not path.exists():
        pytest.skip("summary_seed42.json not found")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tests: controllability labels are defined and correct
# ---------------------------------------------------------------------------

def test_controllability_labels_defined():
    """All three controllability categories must be non-empty."""
    assert len(FRAUDSTER_CONTROLLED) > 0, "Fraudster-controlled list is empty"
    assert len(SYSTEM_CONTROLLED) > 0, "System-controlled list is empty"
    assert len(TEMPORAL) > 0, "Temporal list is empty"


def test_controllability_labels_no_overlap():
    """No feature should appear in multiple controllability categories."""
    fc_set = set(FRAUDSTER_CONTROLLED)
    sc_set = set(SYSTEM_CONTROLLED)
    tp_set = set(TEMPORAL)
    assert fc_set.isdisjoint(sc_set), \
        f"Overlap FC/SC: {fc_set & sc_set}"
    assert fc_set.isdisjoint(tp_set), \
        f"Overlap FC/TP: {fc_set & tp_set}"
    assert sc_set.isdisjoint(tp_set), \
        f"Overlap SC/TP: {sc_set & tp_set}"


def test_feature_count_per_category():
    """Feature counts must match FINDINGS.md documentation."""
    # FINDINGS.md: "Fraudster-controlled: 12 features", "System-controlled: 6 features"
    assert len(FRAUDSTER_CONTROLLED) == 12, \
        f"Expected 12 fraudster-controlled, got {len(FRAUDSTER_CONTROLLED)}"
    assert len(SYSTEM_CONTROLLED) == 6, \
        f"Expected 6 system-controlled, got {len(SYSTEM_CONTROLLED)}"


def test_all_controllability_features_in_pipeline(engineered):
    """Every labeled feature must appear in the pipeline's feature list."""
    _, feature_cols = engineered
    all_labeled = set(FRAUDSTER_CONTROLLED + SYSTEM_CONTROLLED + TEMPORAL)
    missing = all_labeled - set(feature_cols)
    assert missing == set(), f"Labeled features missing from pipeline: {missing}"


def test_all_pipeline_features_labeled(engineered):
    """Every pipeline feature must have a controllability label."""
    _, feature_cols = engineered
    all_labeled = set(FRAUDSTER_CONTROLLED + SYSTEM_CONTROLLED + TEMPORAL)
    unlabeled = set(feature_cols) - all_labeled
    assert unlabeled == set(), f"Unlabeled features in pipeline: {unlabeled}"


# ---------------------------------------------------------------------------
# Tests: system-controlled features produce valid AUC
# ---------------------------------------------------------------------------

def test_system_controlled_features_produce_valid_auc(engineered):
    """A model using only system-controlled features should produce AUC > 0.5."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from scripts.run_pipeline import temporal_split

    df, feature_cols = engineered
    system_feats = [f for f in SYSTEM_CONTROLLED if f in feature_cols]
    assert len(system_feats) >= 3, "Too few system features available"

    X_train, y_train, X_test, y_test = temporal_split(df, system_feats)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
    rf.fit(X_train_s, y_train)
    y_prob = rf.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    assert auc > 0.5, f"System-only AUC {auc:.4f} not above chance"


def test_fraudster_controlled_features_produce_valid_auc(engineered):
    """A model using only fraudster-controlled features should produce AUC > 0.5."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from scripts.run_pipeline import temporal_split

    df, feature_cols = engineered
    fraud_feats = [f for f in FRAUDSTER_CONTROLLED if f in feature_cols]
    assert len(fraud_feats) >= 5, "Too few fraudster features available"

    X_train, y_train, X_test, y_test = temporal_split(df, fraud_feats)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
    rf.fit(X_train_s, y_train)
    y_prob = rf.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    assert auc > 0.5, f"Fraudster-only AUC {auc:.4f} not above chance"


# ---------------------------------------------------------------------------
# Tests: controllability analysis pipeline output
# ---------------------------------------------------------------------------

def test_controllability_in_summary_output(summary_seed42):
    """Pipeline summary must contain controllability results."""
    assert "controllability" in summary_seed42["results"], \
        "controllability key missing from summary"
    ctrl = summary_seed42["results"]["controllability"]
    required_keys = ["fraudster_controlled", "system_controlled",
                     "full_auc", "system_only_auc", "robustness_ratio"]
    for key in required_keys:
        assert key in ctrl, f"Missing controllability key: {key}"


def test_robustness_ratio_above_threshold(summary_seed42):
    """System-only model should achieve >= 70% of full model AUC (H-3 threshold)."""
    ctrl = summary_seed42["results"]["controllability"]
    ratio = ctrl["robustness_ratio"]
    assert ratio >= 0.70, \
        f"Robustness ratio {ratio:.3f} below 70% threshold"


def test_system_only_auc_below_full(summary_seed42):
    """System-only AUC must be strictly less than full model AUC."""
    ctrl = summary_seed42["results"]["controllability"]
    assert ctrl["system_only_auc"] < ctrl["full_auc"], \
        f"System-only AUC {ctrl['system_only_auc']} >= full {ctrl['full_auc']}"
