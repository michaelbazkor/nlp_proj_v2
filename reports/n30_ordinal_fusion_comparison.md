# Ordinal train / binary test — 30-user POC (PCA vs attention)

- **Cohort:** 30 users (5 high-risk)

- **Train target:** suicide score 0–6 (MSE)

- **Dev select / test metrics:** binary high-risk (`suicide >= 3`, AUC/F1)

- **Reps:** reused from `artifacts/text_only/reps`


| Fusion | Variant | AUC | PR-AUC | F1 | Cohen's d |
|---|---|---:|---:|---:|---:|
| pca | stm_ordinal | 0.760 | 0.707 | 0.333 | 3.867 |
| pca | mtm_ordinal | 0.520 | 0.417 | 0.200 | 0.072 |
| attention | stm_ordinal | 0.560 | 0.330 | 0.190 | 0.273 |
| attention | mtm_ordinal | 0.580 | 0.373 | 0.124 | 0.362 |

## PCA vs attention delta (AUC)

- **mtm_ordinal:** +0.060
- **stm_ordinal:** -0.200

## Compare to binary-train baseline (30 users, PCA, `prompt_rerun_comparison.json`)

| Variant | Binary-train PCA AUC | Ordinal-train PCA AUC | Ordinal-train Attention AUC |
|---|---:|---:|---:|
| stm_high | 0.520 | 0.760 | 0.560 |
| mtm_high | 0.520 | 0.520 | 0.580 |