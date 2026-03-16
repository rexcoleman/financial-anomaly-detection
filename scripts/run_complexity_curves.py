#!/usr/bin/env python
"""Generate model complexity curves for fraud detection models.

Varies a key hyperparameter per model while holding others fixed:
  - XGBoost: max_depth [2, 3, 5, 7, 10]
  - RandomForest: n_estimators [10, 50, 100, 200, 500]
  - SVM-RBF: C [0.01, 0.1, 1.0, 10.0, 100.0] (subsampled to 50K)
  - LightGBM: max_depth [2, 3, 5, 7, 10, 15] (if importable)

Evaluation: AUC (matching existing pipeline metric).
Seeds: [42, 123, 456, 789, 1024]

Outputs:
  outputs/diagnostics/complexity_curves_seed{seed}.json
  outputs/diagnostics/complexity_curves_summary.json

Usage:
    python scripts/run_complexity_curves.py
    python scripts/run_complexity_curves.py --seeds 42 --sample-frac 0.1
    python scripts/run_complexity_curves.py --data-dir data/raw
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
SVM_SUBSAMPLE = 50_000

# Complexity sweep ranges
XGBOOST_MAX_DEPTHS = [2, 3, 5, 7, 10]
RF_N_ESTIMATORS = [10, 50, 100, 200, 500]
SVM_C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
LGBM_MAX_DEPTHS = [2, 3, 5, 7, 10, 15]


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


def subsample(X, y, max_n, seed):
    """Subsample if needed for SVM."""
    if len(X) <= max_n:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=max_n, replace=False)
    idx.sort()
    return X[idx], y.iloc[idx] if hasattr(y, "iloc") else y[idx]


def run_complexity_curves(data_dir, seeds, sample_frac):
    """Run complexity curve experiments."""
    output_dir = Path("outputs/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check LightGBM availability
    has_lgbm = False
    try:
        from lightgbm import LGBMClassifier
        has_lgbm = True
    except ImportError:
        print("[WARN] lightgbm not installed — skipping LightGBM curves.")

    # Check XGBoost availability
    has_xgb = False
    try:
        from xgboost import XGBClassifier
        has_xgb = True
    except ImportError:
        print("[WARN] xgboost not installed — skipping XGBoost curves.")

    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        np.random.seed(seed)

        X_train, y_train, X_test, y_test, feature_cols = load_and_prepare(
            data_dir, sample_frac, seed
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        seed_results = {"seed": seed, "curves": {}}

        # --- XGBoost: vary max_depth ---
        if has_xgb:
            from xgboost import XGBClassifier
            print(f"\n  XGBoost (max_depth):")
            train_aucs, test_aucs = [], []
            scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

            for depth in XGBOOST_MAX_DEPTHS:
                model = XGBClassifier(
                    n_estimators=100, max_depth=depth,
                    scale_pos_weight=scale_pos,
                    random_state=seed, eval_metric="logloss",
                    verbosity=0, nthread=1,
                )
                start = time.time()
                model.fit(X_train_s, y_train)
                elapsed = time.time() - start

                train_prob = model.predict_proba(X_train_s)[:, 1]
                test_prob = model.predict_proba(X_test_s)[:, 1]
                train_auc = roc_auc_score(y_train, train_prob)
                test_auc = roc_auc_score(y_test, test_prob)

                train_aucs.append(round(float(train_auc), 4))
                test_aucs.append(round(float(test_auc), 4))

                print(f"    max_depth={depth:<3d}  "
                      f"train_auc={train_auc:.4f}  test_auc={test_auc:.4f}  ({elapsed:.1f}s)")

            seed_results["curves"]["XGBoost"] = {
                "param_name": "max_depth",
                "param_values": XGBOOST_MAX_DEPTHS,
                "train_auc": train_aucs,
                "test_auc": test_aucs,
            }

        # --- RandomForest: vary n_estimators ---
        print(f"\n  RandomForest (n_estimators):")
        train_aucs, test_aucs = [], []

        for n_est in RF_N_ESTIMATORS:
            model = RandomForestClassifier(
                n_estimators=n_est, class_weight="balanced",
                random_state=seed, n_jobs=1,
            )
            start = time.time()
            model.fit(X_train_s, y_train)
            elapsed = time.time() - start

            train_prob = model.predict_proba(X_train_s)[:, 1]
            test_prob = model.predict_proba(X_test_s)[:, 1]
            train_auc = roc_auc_score(y_train, train_prob)
            test_auc = roc_auc_score(y_test, test_prob)

            train_aucs.append(round(float(train_auc), 4))
            test_aucs.append(round(float(test_auc), 4))

            print(f"    n_estimators={n_est:<4d}  "
                  f"train_auc={train_auc:.4f}  test_auc={test_auc:.4f}  ({elapsed:.1f}s)")

        seed_results["curves"]["RandomForest"] = {
            "param_name": "n_estimators",
            "param_values": RF_N_ESTIMATORS,
            "train_auc": train_aucs,
            "test_auc": test_aucs,
        }

        # --- SVM-RBF: vary C ---
        print(f"\n  SVM-RBF (C):")
        train_aucs, test_aucs = [], []

        X_svm_train, y_svm_train = subsample(X_train_s, y_train, SVM_SUBSAMPLE, seed)
        if len(X_svm_train) < len(X_train_s):
            print(f"    Subsampled {len(X_train_s):,} → {len(X_svm_train):,} for SVM")

        for c_val in SVM_C_VALUES:
            model = SVC(
                kernel="rbf", C=c_val, gamma="scale",
                probability=True, class_weight="balanced",
                random_state=seed,
            )
            start = time.time()
            model.fit(X_svm_train, y_svm_train)
            elapsed = time.time() - start

            train_prob = model.predict_proba(X_svm_train)[:, 1]
            test_prob = model.predict_proba(X_test_s)[:, 1]
            train_auc = roc_auc_score(y_svm_train, train_prob)
            test_auc = roc_auc_score(y_test, test_prob)

            train_aucs.append(round(float(train_auc), 4))
            test_aucs.append(round(float(test_auc), 4))

            print(f"    C={c_val:<7.2f}  "
                  f"train_auc={train_auc:.4f}  test_auc={test_auc:.4f}  ({elapsed:.1f}s)")

        seed_results["curves"]["SVM-RBF"] = {
            "param_name": "C",
            "param_values": SVM_C_VALUES,
            "train_auc": train_aucs,
            "test_auc": test_aucs,
        }

        # --- LightGBM: vary max_depth ---
        if has_lgbm:
            from lightgbm import LGBMClassifier
            print(f"\n  LightGBM (max_depth):")
            train_aucs, test_aucs = [], []

            for depth in LGBM_MAX_DEPTHS:
                model = LGBMClassifier(
                    n_estimators=200, max_depth=depth,
                    is_unbalance=True,
                    random_state=seed, n_jobs=1, verbose=-1,
                )
                start = time.time()
                model.fit(X_train_s, y_train)
                elapsed = time.time() - start

                train_prob = model.predict_proba(X_train_s)[:, 1]
                test_prob = model.predict_proba(X_test_s)[:, 1]
                train_auc = roc_auc_score(y_train, train_prob)
                test_auc = roc_auc_score(y_test, test_prob)

                train_aucs.append(round(float(train_auc), 4))
                test_aucs.append(round(float(test_auc), 4))

                print(f"    max_depth={depth:<3d}  "
                      f"train_auc={train_auc:.4f}  test_auc={test_auc:.4f}  ({elapsed:.1f}s)")

            seed_results["curves"]["LightGBM"] = {
                "param_name": "max_depth",
                "param_values": LGBM_MAX_DEPTHS,
                "train_auc": train_aucs,
                "test_auc": test_aucs,
            }

        # Save per-seed
        seed_path = output_dir / f"complexity_curves_seed{seed}.json"
        with open(seed_path, "w") as f:
            json.dump(seed_results, f, indent=2)
        print(f"\n  Saved: {seed_path}")
        all_results.append(seed_results)

    # Summary with mean/std across seeds
    model_names = list(all_results[0]["curves"].keys())
    summary = {
        "experiment": "complexity_curves",
        "seeds": seeds,
        "models": {},
        "per_seed": all_results,
    }

    for model_name in model_names:
        curve_info = all_results[0]["curves"][model_name]
        summary["models"][model_name] = {
            "param_name": curve_info["param_name"],
            "param_values": curve_info["param_values"],
        }

        for metric in ["train_auc", "test_auc"]:
            n_points = len(curve_info[metric])
            means, stds = [], []
            for i in range(n_points):
                vals = [r["curves"][model_name][metric][i]
                        for r in all_results if model_name in r["curves"]]
                means.append(round(float(np.mean(vals)), 4))
                stds.append(round(float(np.std(vals)), 4))
            summary[f"{model_name}_{metric}_mean"] = means
            summary[f"{model_name}_{metric}_std"] = stds

    summary_path = output_dir / "complexity_curves_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate complexity curves (FP-04)")
    parser.add_argument("--data-dir", default="data/raw",
                        help="Directory containing transactions.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Random seeds (default: 42 123 456 789 1024)")
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Data sampling fraction for smoke testing")
    args = parser.parse_args()
    run_complexity_curves(args.data_dir, args.seeds, args.sample_frac)


if __name__ == "__main__":
    main()
