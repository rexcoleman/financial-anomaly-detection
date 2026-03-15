#!/usr/bin/env python
"""Generate publication figures for FP-04."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def model_comparison():
    models = ["Rule-based\n(CFA)", "Logistic\nRegression", "Random\nForest", "XGBoost"]
    aucs = [0.898, 0.977, 0.974, 0.987]
    colors = ["#95a5a6", "#3498db", "#3498db", "#2ecc71"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, aucs, color=colors, edgecolor="#2c3e50", linewidth=1.2)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{auc:.3f}", ha="center", fontweight="bold", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("Fraud Detection: ML vs CFA-Informed Rules", fontsize=13, fontweight="bold")
    ax.set_ylim(0.85, 1.02)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(0.98, 0.05, "CFA rules = 0.898 floor\nML adds +8.9pp", transform=ax.transAxes,
            fontsize=10, ha="right", bbox=dict(boxstyle="round", facecolor="#f0f0f0"))
    plt.tight_layout()
    for p in ["outputs/figures/model_comparison.png", "blog/images/model_comparison.png"]:
        Path(p).parent.mkdir(parents=True, exist_ok=True); plt.savefig(p, dpi=150)
    print("Generated: model_comparison.png")

def controllability():
    categories = ["Full Model\n(all features)", "System-Only\n(adversary-resistant)"]
    aucs = [0.987, 0.798]
    colors = ["#3498db", "#2ecc71"]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(categories, aucs, color=colors, edgecolor="#2c3e50", linewidth=1.2, width=0.5)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{auc:.3f}", ha="center", fontweight="bold", fontsize=14)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("Controllability Analysis: Adversary-Resistant Detection Floor", fontsize=13, fontweight="bold")
    ax.set_ylim(0.6, 1.1)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.annotate("81% robustness\nfloor", xy=(1, 0.798), xytext=(1.3, 0.9),
               fontsize=12, fontweight="bold", color="#27ae60",
               arrowprops=dict(arrowstyle="->", color="#27ae60"))
    plt.tight_layout()
    for p in ["outputs/figures/controllability.png", "blog/images/controllability.png"]:
        plt.savefig(p, dpi=150)
    print("Generated: controllability.png")

def cross_domain():
    domains = ["FP-01\nIDS", "FP-05\nCVE", "FP-02\nAgents", "FP-03\nCrypto", "FP-04\nFraud"]
    attacker = [57, 13, 3, 20, 12]
    defender = [14, 11, 2, 70, 6]
    x = np.arange(len(domains)); width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width/2, attacker, width, label="Attacker-Controlled", color="#e74c3c", edgecolor="#2c3e50")
    ax.bar(x + width/2, defender, width, label="Defender/System-Controlled", color="#3498db", edgecolor="#2c3e50")
    for i, (a, d) in enumerate(zip(attacker, defender)):
        ax.text(x[i]-width/2, a+1, str(a), ha="center", fontsize=10, fontweight="bold")
        ax.text(x[i]+width/2, d+1, str(d), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Count (features / input types / %)", fontsize=11)
    ax.set_title("Adversarial Control Analysis: 5 Domains, 1 Methodology", fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(domains, fontsize=10); ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for p in ["outputs/figures/cross_domain_5.png", "blog/images/cross_domain_5.png"]:
        plt.savefig(p, dpi=150)
    print("Generated: cross_domain_5.png")

if __name__ == "__main__":
    print("Generating FP-04 figures...")
    model_comparison(); controllability(); cross_domain()
    print("Done.")
