# qwen3_n5 smoke batch (Qwen3-32B bf16, no NN train)

## Setup
- Config: `configs/real_qwen3_n5.yaml`
- Model: `Qwen/Qwen3-32B`, bf16, **single GPU** (`cuda:0`, `device_map: null`)
- Why single GPU: `device_map: auto` multi-GPU sharding produces garbage on this Blackwell host (same failure mode as Llama-70B)
- Cohort: same 5 stratified users as `llama_n5` (`seed=42`)
- Caps: 40 posts / 12 images per user; captions reused from Qwen2.5-VL

## Representations
- Path: `artifacts/qwen3_n5/reps/qwen3_32b/`
- 5/5 `.npz` files
- 16 blocks: layers `[20,40,60,64]` × `{input_only,last_prompt,cot,final_pred}`, dim **5120**
- Vector health: **no NaN/Inf/zero blocks**

## Predictions vs labels

| UserId | y_high | pred | match |
|--------|--------|------|-------|
| 10100271303188713 -id | 0 | *(truncated in saved meta)* | — |
| 10155627703976545 -id | 0 | 0 | yes |
| 10156459020503223 -id | 0 | 0 | yes |
| 10212467798318978 -id | 1 | 0 | no |
| 10213660402236997 -id | 0 | 0 | yes |

Parsed accuracy on 4 complete preds: **75%** (missed the one true high-risk user).

Full texts: `artifacts/qwen3_n5/qwen3_gen_summary.json`
