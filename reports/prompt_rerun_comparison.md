# Prompt re-run comparison (high-risk only)
Same 30 users as original POC. New caption + task prompts; captions regenerated.
| Experiment | Variant | Original AUC | New AUC | Δ AUC | Original PR | New PR | Original F1 | New F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| text_only | stm_high | 0.550 | 0.520 | -0.030 | 0.475 | 0.440 | 0.267 | 0.233 |
| text_only | mtm_high | 0.738 | 0.520 | -0.218 | 0.604 | 0.417 | 0.080 | 0.213 |
| standard | stm_high | 0.412 | 0.640 | +0.228 | 0.438 | 0.500 | 0.200 | 0.233 |
| standard | mtm_high | 0.287 | 0.360 | +0.073 | 0.256 | 0.363 | 0.080 | 0.200 |
