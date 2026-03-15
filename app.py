"""Streamlit demo app for Financial Anomaly Detection.

Run: streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Financial Anomaly Detector", page_icon="🔍", layout="wide")

st.title("🔍 Financial Anomaly Detector")
st.markdown("**CFA × ML**: Domain expertise meets machine learning for fraud detection.")
st.markdown("---")

# Sidebar: configuration
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("Fraud probability threshold", 0.0, 1.0, 0.5, 0.05)
show_shap = st.sidebar.checkbox("Show SHAP feature importance", value=True)

# Load results if available
results_path = Path("outputs/baselines/summary_seed42.json")
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)

    # Model comparison
    st.header("📊 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rule-Based (CFA)", f"{results['results']['rule_based']['auc']:.3f}", "baseline")
    col2.metric("Logistic Regression", f"{results['results']['LogisticRegression']['auc']:.3f}",
                f"+{results['results']['LogisticRegression']['improvement_pp']:.1f}pp")
    col3.metric("Random Forest", f"{results['results']['RandomForest']['auc']:.3f}",
                f"+{results['results']['RandomForest']['improvement_pp']:.1f}pp")
    col4.metric("XGBoost", f"{results['results']['XGBoost']['auc']:.3f}",
                f"+{results['results']['XGBoost']['improvement_pp']:.1f}pp")

    st.markdown("---")

    # Controllability
    st.header("🛡️ Adversarial Controllability Analysis")
    ctrl = results["results"].get("controllability", {})
    if ctrl:
        col1, col2, col3 = st.columns(3)
        col1.metric("Full Model AUC", f"{ctrl['full_auc']:.3f}")
        col2.metric("System-Only AUC", f"{ctrl['system_only_auc']:.3f}")
        col3.metric("Robustness Floor", f"{ctrl['robustness_ratio']*100:.0f}%",
                    "adversary-resistant")

        st.info(
            "**System-controlled features** (device fingerprint, card BIN, merchant risk, "
            "address verification) retain **81%** of detection capability even if a fraudster "
            "perfectly manipulates all controllable features."
        )

    st.markdown("---")

    # SHAP
    if show_shap and "shap" in results["results"]:
        st.header("🔬 Feature Importance (SHAP)")
        shap_data = results["results"]["shap"]["top_features"]
        shap_df = pd.DataFrame(shap_data)

        cfa_features = {"amt_to_median_ratio", "amt_zscore", "high_amount", "suspicious_time",
                       "high_risk_merchant", "protonmail", "high_risk_country", "amt_log"}
        shap_df["CFA Feature"] = shap_df["name"].isin(cfa_features)
        shap_df["color"] = shap_df["CFA Feature"].map({True: "#e74c3c", False: "#3498db"})

        st.bar_chart(shap_df.set_index("name")["importance"].head(15))
        st.caption(f"🔴 {results['results']['shap']['cfa_in_top20']} CFA-informed features in top 20")

    st.markdown("---")

    # Anomaly detection
    if "isolation_forest" in results["results"]:
        st.header("🔎 Unsupervised Anomaly Detection")
        iso = results["results"]["isolation_forest"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Anomalies Detected", f"{iso['anomalies']:,}")
        col2.metric("Anomaly Precision", f"{iso['anomaly_precision']:.1%}")
        col3.metric("Additional Fraud Found", f"{iso['additional_fraud_found']}")

else:
    st.warning("No results found. Run the pipeline first: `python scripts/run_pipeline.py --seed 42`")

# Interactive transaction scorer
st.markdown("---")
st.header("🧪 Score a Transaction")

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=50000.0, value=250.0)
    hour = st.slider("Hour of Day", 0, 23, 14)
    is_weekend = st.checkbox("Weekend?")
with col2:
    addr_mismatch = st.checkbox("Address Mismatch?")
    high_risk_country = st.checkbox("High-Risk Country?")
    protonmail = st.checkbox("ProtonMail / Privacy Email?")

# Simple rule-based score
score = 0.0
score += min(amount / 5000, 1.0) * 0.2
score += (1 if addr_mismatch else 0) * 0.3
score += (1 if hour >= 22 or hour <= 5 else 0) * 0.1
score += (1 if is_weekend else 0) * 0.05
score += (1 if high_risk_country else 0) * 0.2
score += (1 if protonmail else 0) * 0.15

risk_level = "🔴 HIGH" if score >= 0.5 else "🟡 MEDIUM" if score >= 0.3 else "🟢 LOW"
st.metric("Risk Score (CFA Rule-Based)", f"{score:.2f}", risk_level)

st.markdown("---")
st.caption("Built with [govML](https://github.com/rexcoleman/govML) v2.5 | "
          "[Source Code](https://github.com/rexcoleman/financial-anomaly-detection)")
