"""Shared fixtures for FP-04 financial anomaly detection tests.

Provides synthetic PaySim-like data fixtures and common path helpers
so that unit tests can run without the full dataset.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def outputs_dir(project_root):
    """Return the outputs directory."""
    return project_root / "outputs"


@pytest.fixture(scope="session")
def models_dir(project_root):
    """Return the models directory."""
    return project_root / "outputs" / "models"


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Return the data directory."""
    return project_root / "data"


# ---------------------------------------------------------------------------
# Synthetic PaySim data fixture (20 rows, matching schema)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_paysim_df():
    """Create a minimal synthetic PaySim-like DataFrame (20 rows).

    Schema matches the real PaySim 100K dataset:
    - 20 features matching DATA_CONTRACT feature schema
    - ~15% fraud rate (3 of 20 rows) for test visibility
    - Deterministic (seed=42)
    """
    rng = np.random.RandomState(42)
    n = 20
    n_fraud = 3

    # Transaction features
    step = np.arange(n)  # temporal ordering
    tx_type = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"], size=n)
    amount = rng.exponential(scale=5000, size=n).round(2)
    oldbalance_org = rng.uniform(0, 50000, size=n).round(2)
    newbalance_org = (oldbalance_org - amount).clip(0).round(2)
    balance_diff = (oldbalance_org - newbalance_org).round(2)

    # CFA-informed features
    amt_to_median_ratio = (amount / np.median(amount)).round(4)
    velocity_1h = rng.poisson(lam=2, size=n).astype(float)
    merchant_risk_score = rng.uniform(0, 1, size=n).round(4)
    protonmail_flag = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    high_risk_country = rng.choice([0, 1], size=n, p=[0.85, 0.15])
    round_amount_flag = (amount % 100 == 0).astype(int)
    new_account_flag = rng.choice([0, 1], size=n, p=[0.8, 0.2])
    off_hours_flag = rng.choice([0, 1], size=n, p=[0.7, 0.3])

    # System-controlled features
    card_type = rng.choice(["visa", "mastercard", "amex"], size=n)
    device_fingerprint = rng.choice(["known", "unknown"], size=n, p=[0.8, 0.2])
    address_verification = rng.choice(["match", "mismatch", "unavailable"], size=n)
    auth_method = rng.choice(["3ds", "pin", "signature", "none"], size=n)

    # Derived features
    hour_of_day = rng.randint(0, 24, size=n)
    is_weekend = rng.choice([0, 1], size=n, p=[0.7, 0.3])

    # Label: first n_fraud rows are fraud for determinism
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[:n_fraud] = 1

    df = pd.DataFrame({
        "step": step,
        "type": tx_type,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "newbalanceOrg": newbalance_org,
        "balanceDiff": balance_diff,
        "amt_to_median_ratio": amt_to_median_ratio,
        "velocity_1h": velocity_1h,
        "merchant_risk_score": merchant_risk_score,
        "protonmail_flag": protonmail_flag,
        "high_risk_country": high_risk_country,
        "round_amount_flag": round_amount_flag,
        "new_account_flag": new_account_flag,
        "off_hours_flag": off_hours_flag,
        "card_type": card_type,
        "device_fingerprint": device_fingerprint,
        "address_verification": address_verification,
        "auth_method": auth_method,
        "hour_of_day": hour_of_day,
        "is_weekend": is_weekend,
        "isFraud": is_fraud,
    })
    return df


@pytest.fixture(scope="session")
def synthetic_paysim_split(synthetic_paysim_df):
    """Split synthetic PaySim into train/val/test using temporal ordering.

    Matches DATA_CONTRACT temporal split: 70/15/15.
    """
    df = synthetic_paysim_df.sort_values("step").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    return {
        "train": df.iloc[:train_end],
        "val": df.iloc[train_end:val_end],
        "test": df.iloc[val_end:],
    }
