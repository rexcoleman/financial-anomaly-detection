"""Basic tests for FP-04 pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_synthetic_data_generation():
    from scripts.generate_synthetic_data import generate_transactions
    df = generate_transactions(1000, 0.035, 42)
    assert len(df) == 1000
    assert "isFraud" in df.columns
    assert "TransactionAmt" in df.columns
    assert df["isFraud"].mean() > 0.02  # Roughly 3.5%


def test_feature_engineering():
    from scripts.generate_synthetic_data import generate_transactions
    from scripts.run_pipeline import engineer_features
    df = generate_transactions(500, 0.035, 42)
    df, feature_cols = engineer_features(df)
    assert len(feature_cols) >= 15
    assert "amt_to_median_ratio" in feature_cols  # CFA feature
    assert "protonmail" in feature_cols  # CFA feature


def test_rule_based_scorer():
    from scripts.generate_synthetic_data import generate_transactions
    from scripts.run_pipeline import engineer_features, rule_based_scorer
    df = generate_transactions(500, 0.035, 42)
    df, _ = engineer_features(df)
    scores = rule_based_scorer(df)
    assert len(scores) == len(df)
    assert scores.min() >= 0
    assert scores.max() <= 1
