#!/usr/bin/env python
"""Train expanded model set: SVM-RBF + LightGBM for fraud detection.

Extends LogisticRegression + RandomForest + XGBoost from run_pipeline.py with:
  - SVM-RBF: SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
              class_weight='balanced'). Subsampled to 50K.
  - LightGBM: LGBMClassifier(n_estimators=200, max_depth=8, is_unbalance=True).
    Graceful skip if lightgbm not installed.

Evaluation: AUC (matching existing pipeline metric).
Seeds: [42, 123, 456, 789, 1024]

Outputs:
  outputs/models/expanded_seed{seed}.json
  outputs/models/expanded_summary.json

Usage:
    python scripts/train_expanded_models.py
    python scripts/train_expanded_models.py --seeds 42 --sample-frac 0.1
    python scripts/train_expanded_models.py --data-dir data/raw
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
SVM_SUBSAMPLE = 50_000


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


def get_expanded_models(seed):
    """Return expanded model configs."""
    models = {
        "SVM-RBF": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=seed,
        ),
    }

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            is_unbalance=True,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
    except ImportError:
        print("  [WARN] lightgbm not installed — skipping LightGBM.")

    return models


def subsample_for_svm(X_train, y_train, max_n, seed):
    """Subsample training data for SVM if too large."""
    if len(X_train) <= max_n:
        return X_train, y_train
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_train), size=max_n, replace=False)
    idx.sort()
    return X_train.iloc[idx], y_train.iloc[idx]


def run_expanded_models(data_dir, seeds, sample_frac):
    """Run expanded model experiments."""
    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        np.random.seed(seed)

        X_train, y_train, X_test, y_test, feature_cols = load_and_prepare(
            data_dir, sample_frac, seed
        )
        print(f"  Train: {len(X_train):,} ({y_train.mean()*100:.1f}% fraud)")
        print(f"  Test: {len(X_test):,} ({y_test.mean()*100:.1f}% fraud)")

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = get_expanded_models(seed)
        seed_results = {"seed": seed, "results": []}

        for model_name, model in models.items():
            print(f"\n  --- {model_name} ---")

            # Subsample for SVM
            if "SVM" in model_name:
                X_tr_s, y_tr = subsample_for_svm(
                    pd.DataFrame(X_train_s), y_train, SVM_SUBSAMPLE, seed
                )
                X_tr_s = X_tr_s.values if hasattr(X_tr_s, "values") else X_tr_s
                y_tr = y_tr.values if hasattr(y_tr, "values") else y_tr
                if len(X_tr_s) < len(X_train_s):
                    print(f"  Subsampled {len(X_train_s):,} → {len(X_tr_s):,} for SVM")
            else:
                X_tr_s = X_train_s
                y_tr = y_train

            start = time.time()
            model.fit(X_tr_s, y_tr)
            train_time = time.time() - start

            y_prob = model.predict_proba(X_test_s)[:, 1]
            y_pred = model.predict(X_test_s)

            auc = roc_auc_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)

            print(f"  AUC: {auc:.4f} | F1: {f1:.4f} | "
                  f"Prec: {prec:.4f} | Rec: {rec:.4f} | Time: {train_time:.1f}s")

            result = {
                "model": model_name,
                "seed": seed,
                "train_time_seconds": round(train_time, 2),
                "auc": round(float(auc), 4),
                "f1": round(float(f1), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "n_train": len(X_tr_s),
                "n_test": len(X_test_s),
            }
            seed_results["results"].append(result)
            all_results.append(result)

        # Save per-seed
        seed_path = output_dir / f"expanded_seed{seed}.json"
        with open(seed_path, "w") as f:
            json.dump(seed_results, f, indent=2, default=str)
        print(f"\n  Saved: {seed_path}")

    # Summary
    summary = {
        "experiment": "expanded_models",
        "seeds": seeds,
        "sample_fraction": sample_frac,
        "models": list(set(r["model"] for r in all_results)),
        "results": all_results,
    }

    for model_name in summary["models"]:
        model_results = [r for r in all_results if r["model"] == model_name]
        aucs = [r["auc"] for r in model_results]
        summary[f"{model_name}_auc_mean"] = round(float(np.mean(aucs)), 4)
        summary[f"{model_name}_auc_std"] = round(float(np.std(aucs)), 4)

    summary_path = output_dir / "expanded_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("EXPANDED MODELS SUMMARY")
    print(f"{'='*60}")
    for model_name in sorted(summary["models"]):
        mean = summary[f"{model_name}_auc_mean"]
        std = summary[f"{model_name}_auc_std"]
        print(f"  {model_name:15s}: AUC = {mean:.4f} +/- {std:.4f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train expanded models (SVM-RBF + LightGBM)")
    parser.add_argument("--data-dir", default="data/raw",
                        help="Directory containing transactions.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Random seeds (default: 42 123 456 789 1024)")
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Data sampling fraction for smoke testing")
    args = parser.parse_args()
    run_expanded_models(args.data_dir, args.seeds, args.sample_frac)


if __name__ == "__main__":
    main()
