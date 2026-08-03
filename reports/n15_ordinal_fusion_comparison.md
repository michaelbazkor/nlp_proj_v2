# Ordinal train / binary test — 15-user subset (PCA vs attention)

- **Cohort:** 15 users stratified from fixed `cohort_n30` (2 high-risk)

- **Train target:** suicide score 0–6 (MSE)

- **Dev select / test metrics:** binary high-risk (`suicide >= 3`, AUC/F1)

- **Reps:** reused from `artifacts/text_only/reps`


| Fusion | Variant | AUC | PR-AUC | F1 | Cohen's d |
|---|---|---:|---:|---:|---:|
| pca | stm_ordinal | 0.500 | 0.667 | 0.133 | -0.000 |
| pca | mtm_ordinal | 0.500 | 0.667 | 0.100 | -0.000 |
| attention | stm_ordinal | 0.000 | 0.333 | 0.000 | -6.722 |
| attention | mtm_ordinal | 0.500 | 0.667 | 0.200 | -0.000 |

## PCA vs attention delta (AUC)

- **mtm_ordinal:** +0.000
- **stm_ordinal:** -0.500