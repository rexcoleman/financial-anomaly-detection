#!/usr/bin/env python
"""Generate learning curves for fraud detection models.

Trains LogisticRegression, RandomForest, and XGBoost on increasing fractions
of training data across multiple seeds. Uses temporal split (train on older,
test on newer transactions).

Seeds: [42, 123, 456, 789, 1024]
Fractions: [0.1, 0.25, 0.5, 0.75, 1.0]

Outputs:
  outputs/diagnostics/learning_curves_seed{seed}.json
  outputs/diagnostics/learning_curves_summary.json

Usage:
    python scripts/run_learning_curves.py
    python scripts/run_learning_curves.py --seeds 42 123 --sample-frac 0.1
    python scripts/run_learning_curves.py --data-dir data/raw
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))


FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


def load_and_prepare(data_dir, sample_frac, seed):
    """Load data and engineer features (mirrors run_pipeline.py)."""
    csv_path = Path(data_dir) / "transactions.csv"
    if not csv_path.exists():
        from scripts.generate_synthetic_data import generate_transactions
        df = generate_transactions(100000, 0.035, seed)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    if sample_frac and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)

    # Feature engineering (same as run_pipeline.py)
    for col in ["card_type", "DeviceType", "DeviceInfo", "email_domain", "billing_country"]:
        if col in df.columns:
            df[col] = df[col].fillna("missing")
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col])

    df["amt_zscore"] = (df["TransactionAmt"] - df["TransactionAmt"].mean()) / df["TransactionAmt"].std()
    df["high_amount"] = (df["TransactionAmt"] > df["TransactionAmt"].quantile(0.95)).astype(int)
    df["suspicious_time"] = (df["is_night"] & df["is_weekend"]).astype(int)
    df["high_risk_merchant"] = (df["merchant_risk_score"] > 0.5).astype(int)
    df["protonmail"] = df["email_domain"].str.contains("proton|temp", case=False, na=False).astype(int)
    df["high_risk_country"] = df["billing_country"].isin(["NG", "RU"]).astype(int)

    feature_cols = [
        "TransactionAmt", "amt_log", "amt_to_median_ratio", "amt_zscore", "high_amount",
        "hour_of_day", "day_of_week", "is_weekend", "is_night", "suspicious_time",
        "merchant_risk_score", "high_risk_merchant", "addr_mismatch",
        "card_type_enc", "DeviceType_enc", "DeviceInfo_enc",
        "email_domain_enc", "billing_country_enc",
        "protonmail", "high_risk_country",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Temporal split
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["isFraud"]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["isFraud"]

    return X_train, y_train, X_test, y_test, feature_cols


def get_models(seed):
    """Return model configs."""
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=seed, n_jobs=1,
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, random_state=seed,
            eval_metric="logloss", verbosity=0, nthread=1,
        )
    except ImportError:
        pass
    return models


def run_learning_curves(data_dir, seeds, sample_frac):
    """Run learning curve experiments."""
    output_dir = Path("outputs/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        np.random.seed(seed)

        X_train_full, y_train_full, X_test, y_test, feature_cols = load_and_prepare(
            data_dir, sample_frac, seed
        )

        scaler = StandardScaler()
        X_train_full_s = scaler.fit_transform(X_train_full)
        X_test_s = scaler.transform(X_test)

        seed_results = {"seed": seed, "fractions": FRACTIONS, "curves": {}}

        for model_name, model in get_models(seed).items():
            print(f"\n  {model_name}:")
            train_aucs, test_aucs, test_f1s = [], [], []
            n_samples_list = []

            for frac in FRACTIONS:
                n = int(len(X_train_full_s) * frac)
                X_sub = X_train_full_s[:n]
                y_sub = y_train_full.iloc[:n]

                from sklearn.base import clone
                m = clone(model)

                start = time.time()
                m.fit(X_sub, y_sub)
                elapsed = time.time() - start

                # Train metrics
                y_train_prob = m.predict_proba(X_sub)[:, 1]
                train_auc = roc_auc_score(y_sub, y_train_prob)

                # Test metrics
                y_test_prob = m.predict_proba(X_test_s)[:, 1]
                y_test_pred = m.predict(X_test_s)
                test_auc = roc_auc_score(y_test, y_test_prob)
                test_f1 = f1_score(y_test, y_test_pred)

                train_aucs.append(round(float(train_auc), 4))
                test_aucs.append(round(float(test_auc), 4))
                test_f1s.append(round(float(test_f1), 4))
                n_samples_list.append(n)

                print(f"    frac={frac:.2f} n={n:>7,}  "
                      f"train_auc={train_auc:.4f}  test_auc={test_auc:.4f}  "
                      f"test_f1={test_f1:.4f}  ({elapsed:.1f}s)")

            seed_results["curves"][model_name] = {
                "train_auc": train_aucs,
                "test_auc": test_aucs,
                "test_f1": test_f1s,
                "n_samples": n_samples_list,
            }

        seed_path = output_dir / f"learning_curves_seed{seed}.json"
        with open(seed_path, "w") as f:
            json.dump(seed_results, f, indent=2)
        print(f"\n  Saved: {seed_path}")
        all_results.append(seed_results)

    # Summary
    summary = {
        "experiment": "learning_curves",
        "seeds": seeds,
        "fractions": FRACTIONS,
        "models": list(get_models(42).keys()),
        "per_seed": all_results,
    }

    for model_name in summary["models"]:
        for metric in ["train_auc", "test_auc", "test_f1"]:
            means, stds = [], []
            for i in range(len(FRACTIONS)):
                vals = [r["curves"][model_name][metric][i]
                        for r in all_results if model_name in r["curves"]]
                means.append(round(float(np.mean(vals)), 4))
                stds.append(round(float(np.std(vals)), 4))
            summary[f"{model_name}_{metric}_mean"] = means
            summary[f"{model_name}_{metric}_std"] = stds

    summary_path = output_dir / "learning_curves_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate learning curves (FP-04)")
    parser.add_argument("--data-dir", default="data/raw",
                        help="Directory containing transactions.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Random seeds (default: 42 123 456 789 1024)")
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Data sampling fraction for smoke testing")
    args = parser.parse_args()
    run_learning_curves(args.data_dir, args.seeds, args.sample_frac)


if __name__ == "__main__":
    main()
