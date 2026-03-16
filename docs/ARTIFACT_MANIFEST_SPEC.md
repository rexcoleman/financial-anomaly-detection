# ARTIFACT MANIFEST SPECIFICATION

<!-- version: 1.0 -->
<!-- created: 2026-03-16 -->

## 1) Run ID Format

```
{model}_{experiment}_{seed}
```

**Components:**
- `model`: classifier name — `lr`, `rf`, `svm_rbf`, `xgboost`, `lightgbm`, `isolation_forest`
- `experiment`: experiment type — `baseline`, `complexity`, `learning_curve`, `shap`, `anomaly`
- `seed`: integer seed from [42, 123, 456, 789, 1024]

**Examples:**
- `xgboost_baseline_42` — XGBoost baseline training, seed 42
- `rf_complexity_123` — Random Forest complexity curve, seed 123

## 2) Per-Run Files

Every run MUST produce:

| File | Format | Description |
|------|--------|-------------|
| `metrics.json` | JSON | AUC, F1, precision, recall per model |
| `config_resolved.yaml` | YAML | Full resolved config |
| `summary.json` | JSON | Aggregated results, wall-clock time |

## 3) Output Directory Structure

```
outputs/
├── baselines/          # Per-seed baseline results
├── models/             # Expanded model comparison (multi-seed)
├── diagnostics/        # Complexity curves, learning curves
├── explainability/     # SHAP outputs
├── anomaly/            # Isolation Forest results
├── adversarial/        # Adversarial evaluation (if applicable)
├── figures/            # Report-ready plots
├── provenance/         # Global provenance files
```

## 4) Global Provenance

Location: `outputs/provenance/`

| File | Contents |
|------|----------|
| `versions.txt` | Python + library versions |
| `git_info.txt` | Git commit SHA and dirty state |
| `config_resolved.yaml` | Global config snapshot |

## 5) Integrity

All hashes use SHA-256. Run `scripts/verify_manifests.py` to validate.

## 6) Change Control

Changes to run_id format or per-run file requirements require a `CONTRACT_CHANGE` commit.
