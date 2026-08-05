# all_n5 — 5 users × 4 models (no NN train)

## Setup
- Config: `configs/real_all_n5.yaml`
- Blackwell fix: **no `device_map: auto`**. Fit-on-one-GPU models load with `device_map: null` on `cuda:0`.
- Llama-3.3-70B bf16 does not fit one 98GB card → **`load_in_8bit: true`** on `cuda:0` (same single-device principle; multi-GPU still broken).
- Gemma HF id corrected to `google/gemma-4-26B-A4B-it` (old `gemma-4-26b-it` 404s).

## Predictions (`RISK`) vs `y_high`

| UserId | y_high | Qwen3-32B | DeepSeek-R1-32B | Gemma4-26B-MoE | Llama-70B (8bit) |
|--------|--------|-----------|-----------------|----------------|------------------|
| …88713 | 0 | 0 | 0 | 0 | *(wrote `Prediction: 0`)* |
| …76545 | 0 | 0 | 0 | 0 | 0 |
| …03223 | 0 | 0 | 0 | 0 | 0 |
| …36978 | **1** | 0 | *(no RISK= line)* | 0 | 0 |
| …36997 | 0 | 0 | 0 | 0 | 0 |

All models lean **RISK=0**; the one true high-risk user is missed by every model that emitted a parseable pred.

## Rep health
| Model | files | blocks/user | NaN/Inf blocks |
|-------|-------|-------------|----------------|
| qwen3_32b | 5/5 | 16 × 5120 | 0 |
| deepseek_r1_distill_qwen_32b | 5/5 | 16 × 5120 | 0 |
| gemma4_26b_moe | 5/5 | 8 (layers 10,12) | 0 |
| llama_3_3_70b | 5/5 | 16 × 8192 | 0 |

Artifacts: `artifacts/all_n5/reps/{model}/` · full texts: `artifacts/all_n5/all_models_gen_summary.json`
