# Prompt v3 re-run comparison (high-risk only)

Same 30 users. Posts inserted at `[insert posts here]`. Caption max_new=128, LLM max_new=256.

| Experiment | Variant | Original AUC | v3 AUC | Δ AUC | Original PR | v3 PR | Original F1 | v3 F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| text_only | stm_high | 0.550 | 0.560 | +0.010 | 0.475 | 0.447 | 0.267 | 0.280 |
| text_only | mtm_high | 0.738 | 0.560 | -0.178 | 0.604 | 0.530 | 0.080 | 0.100 |
| standard | stm_high | 0.412 | 0.360 | -0.052 | 0.438 | 0.373 | 0.200 | 0.100 |
| standard | mtm_high | 0.287 | 0.160 | -0.128 | 0.256 | 0.197 | 0.080 | 0.000 |
