# DATA CONTRACT

<!-- version: 1.0 -->
<!-- created: 2026-03-16 -->

## 1) Dataset

| Property | Value |
|----------|-------|
| Name | PaySim (synthetic financial transactions) |
| Size | 100,000 transactions (sampled from full PaySim) |
| Fraud rate | ~3.5% (class imbalance ratio ~28:1) |
| Features | 20 engineered features |
| Source | `scripts/generate_synthetic_data.py` |

## 2) Feature Schema

| Category | Count | Examples |
|----------|-------|---------|
| Transaction | 6 | amount, type, step (time), oldbalanceOrg, newbalanceOrg, balanceDiff |
| CFA-informed | 8 | amt_to_median_ratio, velocity_1h, merchant_risk_score, protonmail_flag |
| System-controlled | 4 | card_type, device_fingerprint, address_verification, auth_method |
| Derived | 2 | hour_of_day, is_weekend |

## 3) Split Strategy

**Temporal split** (NOT random):
- Train: first 70% of transactions by step (time index)
- Validation: next 15%
- Test: final 15%

Rationale: Temporal split prevents future-data leakage, which random splits allow in time-series fraud data.

## 4) Leakage Prevention

- Fit preprocessing (scaler, encoder) on train only
- No test metrics in per-run outputs
- Temporal ordering preserved (no shuffle before split)
- Validation used for threshold selection; test used for final reporting only

## 5) Data Provenance

| Artifact | Path |
|----------|------|
| Generator script | `scripts/generate_synthetic_data.py` |
| Raw data | `data/paysim_100k.csv` (generated, not downloaded) |
| EDA summary | `outputs/baselines/summary_seed42.json` |

## 6) Synthetic Data Disclosure

All results carry the `SYNTHETIC` tag. PaySim encodes known fraud patterns programmatically. Real-data validation (IEEE-CIS) is a future work item.

## 7) Change Control

Changes to dataset size, feature schema, split ratios, or generation seed require a `CONTRACT_CHANGE` commit.
