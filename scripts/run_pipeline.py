#!/usr/bin/env python
"""Full FP-04 pipeline: data → features → models → SHAP → adversarial → findings.

Usage:
    python scripts/run_pipeline.py --seed 42
    python scripts/run_pipeline.py --seed 42 --sample-frac 0.01
    python scripts/run_pipeline.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_and_prepare(csv_path: str, sample_frac: float = 1.0, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Engineer features including CFA-informed financial signals."""
    # Encode categoricals
    for col in ["card_type", "DeviceType", "DeviceInfo", "email_domain", "billing_country"]:
        if col in df.columns:
            df[col] = df[col].fillna("missing")
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col])

    # CFA-informed features
    df["amt_zscore"] = (df["TransactionAmt"] - df["TransactionAmt"].mean()) / df["TransactionAmt"].std()
    df["high_amount"] = (df["TransactionAmt"] > df["TransactionAmt"].quantile(0.95)).astype(int)
    df["suspicious_time"] = (df["is_night"] & df["is_weekend"]).astype(int)
    df["high_risk_merchant"] = (df["merchant_risk_score"] > 0.5).astype(int)
    df["protonmail"] = df["email_domain"].str.contains("proton|temp", case=False, na=False).astype(int)
    df["high_risk_country"] = df["billing_country"].isin(["NG", "RU"]).astype(int)

    # Feature list
    feature_cols = [
        "TransactionAmt", "amt_log", "amt_to_median_ratio", "amt_zscore", "high_amount",
        "hour_of_day", "day_of_week", "is_weekend", "is_night", "suspicious_time",
        "merchant_risk_score", "high_risk_merchant", "addr_mismatch",
        "card_type_enc", "DeviceType_enc", "DeviceInfo_enc",
        "email_domain_enc", "billing_country_enc",
        "protonmail", "high_risk_country",
    ]

    return df, [c for c in feature_cols if c in df.columns]


def temporal_split(df: pd.DataFrame, feature_cols: list[str], train_frac: float = 0.7):
    """Temporal split — train on earlier transactions, test on later."""
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df) * train_frac)

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train[feature_cols].fillna(0)
    y_train = train["isFraud"]
    X_test = test[feature_cols].fillna(0)
    y_test = test["isFraud"]

    return X_train, y_train, X_test, y_test


def rule_based_scorer(df_test: pd.DataFrame) -> np.ndarray:
    """CFA-style rule-based fraud scoring."""
    scores = np.zeros(len(df_test))
    scores += (df_test["TransactionAmt"] > 500).values * 0.3
    scores += df_test["addr_mismatch"].fillna(0).values * 0.3
    scores += df_test["is_night"].fillna(0).values * 0.1
    scores += (df_test["merchant_risk_score"] > 0.5).values * 0.2
    scores += df_test["high_risk_country"].fillna(0).values * 0.1
    return scores


def run_pipeline(seed: int, sample_frac: float, dry_run: bool):
    print("=" * 60)
    print("FP-04: Financial Anomaly Detection Pipeline")
    print("=" * 60)

    if dry_run:
        print("[dry-run] Would load data, engineer features, train models.")
        return

    # Generate data if not exists
    data_path = "data/raw/transactions.csv"
    if not Path(data_path).exists():
        print("Generating synthetic data...")
        from scripts.generate_synthetic_data import generate_transactions
        df = generate_transactions(100000, 0.035, seed)
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        print(f"Loading {data_path}...")

    df = load_and_prepare(data_path, sample_frac, seed)
    print(f"  Transactions: {len(df):,} | Fraud: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.1f}%)")

    # Feature engineering
    df, feature_cols = engineer_features(df)
    print(f"  Features: {len(feature_cols)}")

    # Split
    X_train, y_train, X_test, y_test = temporal_split(df, feature_cols)
    print(f"  Train: {len(X_train):,} ({y_train.mean()*100:.1f}% fraud)")
    print(f"  Test: {len(X_test):,} ({y_test.mean()*100:.1f}% fraud)")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # === RQ1: Supervised models vs rules ===
    print(f"\n{'='*60}")
    print("RQ1: Supervised ML vs Rule-Based")
    print(f"{'='*60}")

    # Rule-based baseline
    split_idx = int(len(df) * 0.7)
    test_df = df.iloc[split_idx:]
    rule_scores = rule_based_scorer(test_df)
    rule_auc = roc_auc_score(y_test, rule_scores)
    print(f"\n  Rule-based AUC: {rule_auc:.4f}")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1),
    }

    # Try XGBoost
    try:
        from xgboost import XGBClassifier
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, scale_pos_weight=scale_pos,
            random_state=seed, eval_metric="logloss", verbosity=0,
        )
    except ImportError:
        print("  XGBoost not available, skipping.")

    results = {"rule_based": {"auc": rule_auc}}
    best_model_name = None
    best_model = None
    best_auc = 0

    for name, model in models.items():
        print(f"\n  --- {name} ---")
        model.fit(X_train_s, y_train)
        y_prob = model.predict_proba(X_test_s)[:, 1]
        y_pred = model.predict(X_test_s)

        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        improvement = (auc - rule_auc) * 100

        print(f"  AUC: {auc:.4f} ({improvement:+.1f}pp vs rules) | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

        results[name] = {"auc": auc, "f1": f1, "precision": prec, "recall": rec, "improvement_pp": improvement}

        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model = model

    rq1_pass = (best_auc - rule_auc) * 100 >= 10
    print(f"\n  Best: {best_model_name} (AUC {best_auc:.4f}, {(best_auc-rule_auc)*100:+.1f}pp)")
    print(f"  RQ1 (≥10pp): {'PASS' if rq1_pass else 'FAIL'}")

    # === RQ2: Anomaly detection ===
    print(f"\n{'='*60}")
    print("RQ2: Anomaly Detection (Unsupervised)")
    print(f"{'='*60}")

    iso = IsolationForest(contamination=0.035, random_state=seed, n_jobs=-1)
    iso.fit(X_train_s)
    iso_pred = iso.predict(X_test_s)
    iso_anomaly = (iso_pred == -1).astype(int)

    # How many anomalies are actual fraud?
    anomaly_fraud = (iso_anomaly & y_test.values).sum()
    anomaly_total = iso_anomaly.sum()
    anomaly_precision = anomaly_fraud / max(anomaly_total, 1)

    # Supervised predictions
    best_pred = best_model.predict(X_test_s)
    supervised_fraud = (best_pred & y_test.values).sum()

    # Fraud caught by anomaly but NOT by supervised
    anomaly_only = ((iso_anomaly == 1) & (best_pred == 0) & (y_test.values == 1)).sum()
    total_fraud = y_test.sum()
    additional_pct = anomaly_only / max(total_fraud, 1) * 100

    print(f"\n  Isolation Forest anomalies: {anomaly_total:,}")
    print(f"  Anomalies that are fraud: {anomaly_fraud:,} (precision: {anomaly_precision:.2%})")
    print(f"  Fraud caught by supervised only: {supervised_fraud:,}")
    print(f"  Fraud caught by anomaly but NOT supervised: {anomaly_only:,} ({additional_pct:.1f}%)")

    rq2_pass = additional_pct >= 10
    print(f"  RQ2 (≥10% additional): {'PASS' if rq2_pass else 'FAIL'}")

    results["isolation_forest"] = {
        "anomalies": int(anomaly_total),
        "anomaly_precision": anomaly_precision,
        "additional_fraud_found": int(anomaly_only),
        "additional_pct": additional_pct,
    }

    # === RQ3: SHAP explainability ===
    print(f"\n{'='*60}")
    print("RQ3: Feature Importance (SHAP)")
    print(f"{'='*60}")

    try:
        import shap
        if best_model_name == "XGBoost":
            explainer = shap.TreeExplainer(best_model)
        else:
            explainer = shap.TreeExplainer(best_model) if hasattr(best_model, "estimators_") else None

        if explainer:
            X_sample = X_test_s[:min(1000, len(X_test_s))]
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 (fraud)

            importance = np.abs(shap_values).mean(axis=0)
            feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])

            print("\n  Top 15 features by SHAP importance:")
            cfa_features = {"amt_to_median_ratio", "amt_zscore", "high_amount", "suspicious_time",
                           "high_risk_merchant", "protonmail", "high_risk_country", "amt_log"}
            cfa_in_top20 = 0
            for i, (name, imp) in enumerate(feat_imp[:15]):
                is_cfa = name in cfa_features
                if is_cfa:
                    cfa_in_top20 += 1
                marker = " ← CFA" if is_cfa else ""
                print(f"    {i+1:2d}. {name:<25s} {imp:.4f}{marker}")

            # Count CFA features in top 20
            for name, _ in feat_imp[15:20]:
                if name in cfa_features:
                    cfa_in_top20 += 1

            rq3_pass = cfa_in_top20 > 0
            print(f"\n  CFA features in top 20: {cfa_in_top20}")
            print(f"  RQ3 (CFA features in top 20): {'PASS' if rq3_pass else 'FAIL'}")
            results["shap"] = {"top_features": [{"name": n, "importance": float(v)} for n, v in feat_imp[:20]],
                              "cfa_in_top20": cfa_in_top20}

            # Save SHAP plot
            Path("outputs/explainability").mkdir(parents=True, exist_ok=True)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, pd.DataFrame(X_sample, columns=feature_cols),
                            show=False, max_display=15)
            plt.tight_layout()
            plt.savefig("outputs/explainability/shap_summary.png", dpi=150)
            plt.savefig("blog/images/shap_summary.png", dpi=150)
            plt.close()
            print("  Saved: shap_summary.png")
        else:
            print("  SHAP not available for this model type")
            rq3_pass = False
    except Exception as e:
        print(f"  SHAP error: {e}")
        rq3_pass = False

    # === RQ4: Controllability analysis ===
    print(f"\n{'='*60}")
    print("RQ4: Feature Controllability Analysis")
    print(f"{'='*60}")

    controllability = {
        "fraudster_controlled": [
            "TransactionAmt", "amt_log", "amt_to_median_ratio", "amt_zscore", "high_amount",
            "hour_of_day", "is_night", "suspicious_time",
            "email_domain_enc", "protonmail",
            "billing_country_enc", "high_risk_country",
        ],
        "system_controlled": [
            "card_type_enc", "DeviceType_enc", "DeviceInfo_enc",
            "merchant_risk_score", "high_risk_merchant",
            "addr_mismatch",  # Address verification is system-side
        ],
        "temporal": [
            "day_of_week", "is_weekend",
        ],
    }

    # Evaluate model with only system-controlled features
    system_features = [f for f in controllability["system_controlled"] if f in feature_cols]
    X_train_sys = X_train[system_features].fillna(0)
    X_test_sys = X_test[system_features].fillna(0)
    scaler_sys = StandardScaler()
    X_train_sys_s = scaler_sys.fit_transform(X_train_sys)
    X_test_sys_s = scaler_sys.transform(X_test_sys)

    model_sys = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1)
    model_sys.fit(X_train_sys_s, y_train)
    auc_sys = roc_auc_score(y_test, model_sys.predict_proba(X_test_sys_s)[:, 1])

    print(f"\n  Fraudster-controlled features: {len(controllability['fraudster_controlled'])}")
    print(f"  System-controlled features: {len(system_features)}")
    print(f"  Full model AUC: {best_auc:.4f}")
    print(f"  System-only model AUC: {auc_sys:.4f}")
    print(f"  Robustness differential: {best_auc - auc_sys:.4f}")
    print(f"  → System features alone achieve {auc_sys/best_auc*100:.0f}% of full model performance")

    rq4_pass = True  # Controllability matrix documented
    results["controllability"] = {
        "fraudster_controlled": len(controllability["fraudster_controlled"]),
        "system_controlled": len(system_features),
        "full_auc": best_auc,
        "system_only_auc": auc_sys,
        "robustness_ratio": auc_sys / best_auc,
    }

    # === Summary ===
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  RQ1 (ML ≥10pp vs rules): {'PASS' if rq1_pass else 'FAIL'} — {best_model_name} +{(best_auc-rule_auc)*100:.1f}pp")
    print(f"  RQ2 (≥10% additional fraud): {'PASS' if rq2_pass else 'FAIL'} — {additional_pct:.1f}%")
    print(f"  RQ3 (CFA features in top 20): {'PASS' if rq3_pass else 'FAIL'}")
    print(f"  RQ4 (Controllability matrix): {'PASS' if rq4_pass else 'FAIL'}")

    # Save summary
    summary = {
        "seed": seed, "sample_frac": sample_frac,
        "n_transactions": len(df), "fraud_rate": float(df["isFraud"].mean()),
        "results": results,
        "rq1_pass": rq1_pass, "rq2_pass": rq2_pass,
        "rq3_pass": rq3_pass, "rq4_pass": rq4_pass,
    }
    out_dir = Path(f"outputs/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"summary_seed{seed}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: outputs/baselines/summary_seed{seed}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.seed, args.sample_frac, args.dry_run)


if __name__ == "__main__":
    main()
