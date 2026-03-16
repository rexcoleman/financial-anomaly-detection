#!/usr/bin/env python
"""Generate all report figures from JSON output data.

Reads outputs/{baselines,models,diagnostics,explainability}/ and produces
publication-ready PNGs in outputs/figures/ and blog/images/.

No hardcoded values — everything is loaded from JSON.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
FIG_DIRS = [OUTPUTS / "figures", ROOT / "blog" / "images"]


def _save(fig, name):
    for d in FIG_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ── 1. Model comparison bar chart (AUC across all models x seeds) ──────────

def fig_model_comparison():
    """Bar chart of mean AUC per model, with per-seed scatter and error bars."""
    # Gather per-seed results from baselines + expanded models
    model_seeds: dict[str, list[float]] = defaultdict(list)

    # Baselines (LogisticRegression, RandomForest, XGBoost, rule_based)
    for p in sorted(OUTPUTS.glob("baselines/summary_seed*.json")):
        with open(p) as f:
            data = json.load(f)
        res = data["results"]
        if "rule_based" in res:
            model_seeds["Rule-Based\n(CFA)"].append(res["rule_based"]["auc"])
        for m in ("LogisticRegression", "RandomForest", "XGBoost"):
            if m in res:
                model_seeds[m].append(res[m]["auc"])

    # Expanded models (SVM-RBF, LightGBM)
    expanded_path = OUTPUTS / "models" / "expanded_summary.json"
    if expanded_path.exists():
        with open(expanded_path) as f:
            data = json.load(f)
        for row in data.get("results", []):
            model_seeds[row["model"]].append(row["auc"])

    if not model_seeds:
        print("  SKIP model_comparison — no data found")
        return

    # Compute stats
    names = list(model_seeds.keys())
    means = [np.mean(model_seeds[n]) for n in names]
    stds = [np.std(model_seeds[n]) for n in names]

    # Sort by mean AUC
    order = np.argsort(means)
    names = [names[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]
    seeds_sorted = [model_seeds[n] for n in names]

    # Colour: grey for rule-based, blue for ML, green for best
    colors = []
    best_idx = len(means) - 1
    for i, n in enumerate(names):
        if "Rule" in n:
            colors.append("#95a5a6")
        elif i == best_idx:
            colors.append("#2ecc71")
        else:
            colors.append("#3498db")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=colors, edgecolor="#2c3e50", linewidth=1.2)

    # Scatter individual seeds
    for i, vals in enumerate(seeds_sorted):
        ax.scatter([x[i]] * len(vals), vals, color="black", s=18, zorder=5, alpha=0.6)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{m:.3f}", ha="center", fontweight="bold", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("Fraud Detection: Model Comparison (multi-seed)", fontsize=13, fontweight="bold")
    ax.set_ylim(0.85, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "model_comparison.png")


# ── 2. Complexity curves ───────────────────────────────────────────────────

def fig_complexity_curves():
    """Train/test AUC vs complexity parameter for each model."""
    summary_path = OUTPUTS / "diagnostics" / "complexity_curves_summary.json"
    if not summary_path.exists():
        print("  SKIP complexity_curves — no summary data")
        return

    with open(summary_path) as f:
        data = json.load(f)

    models_info = data.get("models", {})
    if not models_info:
        print("  SKIP complexity_curves — no model info in summary")
        return

    n_models = len(models_info)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)
    axes = axes[0]

    for idx, (model_name, info) in enumerate(models_info.items()):
        ax = axes[idx]
        param_name = info["param_name"]
        param_vals = info["param_values"]

        train_mean_key = f"{model_name}_train_auc_mean"
        test_mean_key = f"{model_name}_test_auc_mean"
        train_std_key = f"{model_name}_train_auc_std"
        test_std_key = f"{model_name}_test_auc_std"

        train_mean = np.array(data.get(train_mean_key, []))
        test_mean = np.array(data.get(test_mean_key, []))
        train_std = np.array(data.get(train_std_key, []))
        test_std = np.array(data.get(test_std_key, []))

        x = np.arange(len(param_vals))
        ax.errorbar(x, train_mean, yerr=train_std, marker="o", label="Train AUC",
                     color="#e74c3c", capsize=3)
        ax.errorbar(x, test_mean, yerr=test_std, marker="s", label="Test AUC",
                     color="#3498db", capsize=3)

        # Shade the gap
        ax.fill_between(x, test_mean, train_mean, alpha=0.15, color="#e74c3c")

        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in param_vals], fontsize=9)
        ax.set_xlabel(param_name, fontsize=11)
        ax.set_ylabel("AUC-ROC", fontsize=11)
        ax.set_title(model_name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="lower left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Complexity Curves: Train/Test AUC vs Hyperparameter (5-seed mean +/- std)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "complexity_curves.png")


# ── 3. CFA feature importance (SHAP) ──────────────────────────────────────

def fig_shap_features():
    """Horizontal bar chart of SHAP feature importances, CFA features highlighted."""
    # Try to find SHAP data in any baseline summary
    shap_data = None
    for p in sorted(OUTPUTS.glob("baselines/summary_seed*.json")):
        with open(p) as f:
            data = json.load(f)
        if "shap" in data.get("results", {}):
            shap_data = data["results"]["shap"]
            break

    if shap_data is None or "top_features" not in shap_data:
        print("  SKIP shap_features — no SHAP data found")
        return

    features = shap_data["top_features"]
    # Filter to non-zero importance
    features = [f for f in features if f["importance"] > 0]

    names = [f["name"] for f in features]
    importances = [f["importance"] for f in features]

    # CFA-informed features (from domain knowledge)
    cfa_features = {
        "amt_to_median_ratio", "protonmail", "merchant_risk_score",
        "high_risk_country", "suspicious_time", "high_amount",
        "is_night", "is_weekend", "high_risk_merchant", "addr_mismatch"
    }
    colors = ["#2ecc71" if n in cfa_features else "#3498db" for n in names]

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(names))
    ax.barh(y, importances, color=colors, edgecolor="#2c3e50", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("Feature Importance (SHAP): CFA-Informed Features Highlighted",
                 fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#2ecc71", edgecolor="#2c3e50", label="CFA-informed"),
        Patch(facecolor="#3498db", edgecolor="#2c3e50", label="Data-derived"),
    ], loc="lower right", fontsize=11)

    fig.tight_layout()
    _save(fig, "shap_features.png")


# ── 4. Controllability analysis chart ──────────────────────────────────────

def fig_controllability():
    """Grouped bar: full model vs system-only AUC across seeds."""
    full_aucs = []
    sys_aucs = []

    for p in sorted(OUTPUTS.glob("baselines/summary_seed*.json")):
        with open(p) as f:
            data = json.load(f)
        ctrl = data.get("results", {}).get("controllability", {})
        if ctrl:
            full_aucs.append(ctrl["full_auc"])
            sys_aucs.append(ctrl["system_only_auc"])

    if not full_aucs:
        print("  SKIP controllability — no controllability data found")
        return

    categories = ["Full Model\n(all features)", "System-Only\n(adversary-resistant)"]
    means = [np.mean(full_aucs), np.mean(sys_aucs)]
    stds = [np.std(full_aucs), np.std(sys_aucs)]
    robustness = means[1] / means[0] if means[0] > 0 else 0

    colors = ["#3498db", "#2ecc71"]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(categories, means, yerr=stds, capsize=5,
                  color=colors, edgecolor="#2c3e50", linewidth=1.2, width=0.5)

    # Scatter individual seeds
    for i, vals in enumerate([full_aucs, sys_aucs]):
        ax.scatter([i] * len(vals), vals, color="black", s=20, zorder=5, alpha=0.6)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{m:.3f}", ha="center", fontweight="bold", fontsize=14)

    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("Controllability Analysis: Adversary-Resistant Detection Floor",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.6, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.annotate(f"{robustness:.0%} robustness\nfloor",
                xy=(1, means[1]), xytext=(1.35, 0.9),
                fontsize=12, fontweight="bold", color="#27ae60",
                arrowprops=dict(arrowstyle="->", color="#27ae60"))

    fig.tight_layout()
    _save(fig, "controllability.png")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("Generating report figures from JSON outputs...")
    fig_model_comparison()
    fig_complexity_curves()
    fig_shap_features()
    fig_controllability()
    print("Done.")


if __name__ == "__main__":
    main()
