#!/usr/bin/env python
"""Generate synthetic financial transaction data for fraud detection.

Creates a dataset similar to IEEE-CIS Fraud Detection with realistic
financial features. Avoids Kaggle auth dependency.

Usage:
    python scripts/generate_synthetic_data.py --n-transactions 100000 --fraud-rate 0.035
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def generate_transactions(n: int = 100000, fraud_rate: float = 0.035, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic financial transactions with fraud labels."""
    rng = np.random.RandomState(seed)

    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    # Transaction features
    data = {
        "TransactionID": np.arange(n),
        "isFraud": np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(int),
    }

    # Shuffle
    idx = rng.permutation(n)
    for k in data:
        data[k] = data[k][idx]

    df = pd.DataFrame(data)

    # Amount — fraudulent transactions tend to be higher
    df["TransactionAmt"] = np.where(
        df["isFraud"] == 1,
        rng.lognormal(5.5, 1.5, n),  # Fraud: higher amounts
        rng.lognormal(3.5, 1.2, n),  # Legit: lower amounts
    )
    df["TransactionAmt"] = df["TransactionAmt"].clip(0.5, 50000).round(2)

    # Time — sequential with some noise
    df["TransactionDT"] = np.sort(rng.uniform(0, 365 * 24 * 3600, n))

    # Card features (system-controlled — fraudster can't easily change)
    df["card_type"] = rng.choice(["visa", "mastercard", "amex", "discover"], n, p=[0.45, 0.30, 0.15, 0.10])
    df["card_bin"] = rng.randint(400000, 600000, n)

    # Device features (system-controlled)
    df["DeviceType"] = rng.choice(["desktop", "mobile", "tablet"], n, p=[0.50, 0.40, 0.10])
    df["DeviceInfo"] = rng.choice(["Windows", "iOS", "Android", "MacOS", "Linux", "Other"], n,
                                   p=[0.35, 0.25, 0.20, 0.10, 0.05, 0.05])

    # Merchant features
    n_merchants = 500
    df["merchant_id"] = rng.randint(0, n_merchants, n)
    merchant_risk = rng.beta(2, 5, n_merchants)  # Most merchants low risk
    df["merchant_risk_score"] = merchant_risk[df["merchant_id"]]

    # Velocity features (CFA-informed: transaction frequency signals)
    # Simulate: group by card_bin and count recent transactions
    df["hour_of_day"] = (df["TransactionDT"] / 3600 % 24).astype(int)
    df["day_of_week"] = (df["TransactionDT"] / 86400 % 7).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)

    # CFA-informed features
    df["amt_to_median_ratio"] = df["TransactionAmt"] / df.groupby("card_type")["TransactionAmt"].transform("median")
    df["amt_log"] = np.log1p(df["TransactionAmt"])

    # Fraudster-controlled features (can manipulate these)
    df["email_domain"] = np.where(
        df["isFraud"] == 1,
        rng.choice(["gmail.com", "yahoo.com", "protonmail.com", "temp-mail.org"], n,
                   p=[0.3, 0.2, 0.3, 0.2]),
        rng.choice(["gmail.com", "yahoo.com", "outlook.com", "company.com"], n,
                   p=[0.4, 0.25, 0.2, 0.15]),
    )

    df["billing_country"] = np.where(
        df["isFraud"] == 1,
        rng.choice(["US", "GB", "NG", "RU", "BR", "IN"], n, p=[0.3, 0.1, 0.2, 0.15, 0.15, 0.1]),
        rng.choice(["US", "GB", "CA", "AU", "DE", "FR"], n, p=[0.6, 0.15, 0.1, 0.05, 0.05, 0.05]),
    )

    # Address mismatch (strong fraud signal)
    df["addr_mismatch"] = np.where(
        df["isFraud"] == 1,
        rng.choice([0, 1], n, p=[0.3, 0.7]),  # 70% mismatch for fraud
        rng.choice([0, 1], n, p=[0.95, 0.05]),  # 5% mismatch for legit
    )

    # Missing values (realistic)
    for col in ["DeviceInfo", "email_domain", "billing_country"]:
        mask = rng.random(n) < 0.15
        df.loc[mask, col] = np.nan

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic financial transaction data")
    parser.add_argument("--n-transactions", type=int, default=100000)
    parser.add_argument("--fraud-rate", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/raw/transactions.csv")
    args = parser.parse_args()

    print(f"Generating {args.n_transactions:,} synthetic transactions (fraud rate: {args.fraud_rate:.1%})...")
    df = generate_transactions(args.n_transactions, args.fraud_rate, args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"  Transactions: {len(df):,}")
    print(f"  Fraud: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.1f}%)")
    print(f"  Features: {len(df.columns)}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
