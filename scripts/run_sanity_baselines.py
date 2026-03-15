#!/usr/bin/env python
"""Run sanity-check baselines: DummyClassifier + shuffled labels.

Confirms that real models outperform trivial baselines, ruling out
data leakage or label correlation artifacts.

Baselines:
  1. DummyClassifier (most_frequent) — always predicts majority class
  2. DummyClassifier (stratified) — random predictions matching class distribution
  3. Shuffled labels — train real model on shuffled y, test on real y

Outputs:
  outputs/diagnostics/sanity_baselines_seed{seed}.json

Usage:
    python scripts/run_sanity_baselines.py
    python scripts/run_sanity_baselines.py --seeds 42 123 --sample-frac 0.1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))


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

    df = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["isFraud"]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["isFraud"]

    return X_train, y_train, X_test, y_test


def run_sanity_baselines(data_dir, seeds, sample_frac):
    """Run sanity baselines."""
    output_dir = Path("outputs/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        np.random.seed(seed)
        X_train, y_train, X_test, y_test = load_and_prepare(data_dir, sample_frac, seed)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        seed_results = {"seed": seed, "baselines": {}}

        # 1. DummyClassifier — most_frequent
        print("\n  DummyClassifier (most_frequent):")
        dummy_mf = DummyClassifier(strategy="most_frequent", random_state=seed)
        dummy_mf.fit(X_train_s, y_train)
        y_pred_mf = dummy_mf.predict(X_test_s)
        f1_mf = f1_score(y_test, y_pred_mf, zero_division=0)
        print(f"    F1: {f1_mf:.4f}")
        seed_results["baselines"]["dummy_most_frequent"] = {"f1": round(float(f1_mf), 4)}

        # 2. DummyClassifier — stratified
        print("\n  DummyClassifier (stratified):")
        dummy_strat = DummyClassifier(strategy="stratified", random_state=seed)
        dummy_strat.fit(X_train_s, y_train)
        y_pred_strat = dummy_strat.predict(X_test_s)
        y_prob_strat = dummy_strat.predict_proba(X_test_s)[:, 1]
        f1_strat = f1_score(y_test, y_pred_strat, zero_division=0)
        auc_strat = roc_auc_score(y_test, y_prob_strat)
        print(f"    F1: {f1_strat:.4f}  AUC: {auc_strat:.4f}")
        seed_results["baselines"]["dummy_stratified"] = {
            "f1": round(float(f1_strat), 4),
            "auc": round(float(auc_strat), 4),
        }

        # 3. Real model on real labels (reference)
        print("\n  RandomForest (real labels):")
        rf_real = RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=seed, n_jobs=1,
        )
        rf_real.fit(X_train_s, y_train)
        y_prob_real = rf_real.predict_proba(X_test_s)[:, 1]
        y_pred_real = rf_real.predict(X_test_s)
        auc_real = roc_auc_score(y_test, y_prob_real)
        f1_real = f1_score(y_test, y_pred_real)
        print(f"    F1: {f1_real:.4f}  AUC: {auc_real:.4f}")
        seed_results["baselines"]["rf_real"] = {
            "f1": round(float(f1_real), 4),
            "auc": round(float(auc_real), 4),
        }

        # 4. Shuffled labels — train on shuffled y, test on real y
        print("\n  RandomForest (shuffled labels):")
        rng = np.random.RandomState(seed)
        y_train_shuffled = y_train.values.copy()
        rng.shuffle(y_train_shuffled)

        rf_shuffled = RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=seed, n_jobs=1,
        )
        rf_shuffled.fit(X_train_s, y_train_shuffled)
        y_prob_shuf = rf_shuffled.predict_proba(X_test_s)[:, 1]
        y_pred_shuf = rf_shuffled.predict(X_test_s)
        auc_shuf = roc_auc_score(y_test, y_prob_shuf)
        f1_shuf = f1_score(y_test, y_pred_shuf, zero_division=0)
        print(f"    F1: {f1_shuf:.4f}  AUC: {auc_shuf:.4f}")
        seed_results["baselines"]["rf_shuffled"] = {
            "f1": round(float(f1_shuf), 4),
            "auc": round(float(auc_shuf), 4),
        }

        # Sanity check: real >> shuffled
        gap = auc_real - auc_shuf
        print(f"\n  Sanity gap (real - shuffled AUC): {gap:.4f}")
        print(f"  {'PASS' if gap > 0.1 else 'FAIL'}: real model significantly outperforms shuffled")
        seed_results["sanity_gap_auc"] = round(float(gap), 4)
        seed_results["sanity_pass"] = gap > 0.1

        seed_path = output_dir / f"sanity_baselines_seed{seed}.json"
        with open(seed_path, "w") as f:
            json.dump(seed_results, f, indent=2)
        print(f"\n  Saved: {seed_path}")
        all_results.append(seed_results)

    # Print summary
    print(f"\n{'='*60}")
    print("SANITY BASELINE SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        gap = r["sanity_gap_auc"]
        status = "PASS" if r["sanity_pass"] else "FAIL"
        print(f"  Seed {r['seed']}: gap={gap:.4f} [{status}]")


def main():
    parser = argparse.ArgumentParser(description="Run sanity-check baselines (FP-04)")
    parser.add_argument("--data-dir", default="data/raw",
                        help="Directory containing transactions.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Random seeds")
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Data sampling fraction for smoke testing")
    args = parser.parse_args()
    run_sanity_baselines(args.data_dir, args.seeds, args.sample_frac)


if __name__ == "__main__":
    main()
