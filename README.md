# SSR — Suicide-risk representations (H100 full cohort)

Modernizes Ophir et al. (2020): LLM hidden states → fused vectors → STM/MTM classifiers.

## H100 full run (single command)

```bash
pip install -e ".[h100]"
export HF_TOKEN=...                    # for gated models (Llama-3.3-70B)
python scripts/run_h100_full.py
```

### What it does

**Phase 1 — represent (~3–5 h on H100)**  
Cohort (N≈1006) → posts → VLM captions → corpora → sequential LLM extraction to `.npz`:
- Qwen3-32B, DeepSeek-R1-Distill-32B, Gemma-3-27B — BF16 + FlashAttention-2
- Llama-3.3-70B — 4-bit NF4 on single 80 GB H100 (change to `dtype: bfloat16` on dual-GPU)

**Phase 2 — train matrix (~15–30 min)**  
8 configurations on cached reps (all evaluated on binary `suicide >= 3`):

| Fusion | Train target | Architecture |
|--------|-------------|--------------|
| PCA | Ordinal (0–6) | STM / MTM |
| PCA | Binary (≥3) | STM / MTM |
| Attention | Ordinal (0–6) | STM / MTM |
| Attention | Binary (≥3) | STM / MTM |

### Outputs

```
artifacts/h100_full/
  reps/{model}/*.npz          # cached hidden states (reuse for all 8 trains)
  train_matrix/{run_id}/      # per-experiment checkpoints + metrics.json
  matrix_comparison.json      # aggregated results
reports/h100_matrix_comparison.md
```

### Options

```bash
python scripts/run_h100_full.py --phase train          # skip extraction
python scripts/run_h100_full.py --phase represent      # extract only
python scripts/run_h100_full.py --skip-captions        # text-only corpus
python scripts/run_h100_full.py --force                # re-extract all users
```

### Dual H100 (160 GB) — run Llama-3.3-70B in BF16

In `configs/h100_full.yaml`, for `llama_3_3_70b`:
```yaml
dtype: bfloat16
load_in_4bit: false
device_map: auto
```

## Config

Primary config: `configs/h100_full.yaml`

## Cohort (paper filter)

- Yields **N=1006** on this CSV (paper text says 1002; post total **83292** matches the paper).
- After dropping null labels/aux: **~1003** training users.
- **High risk:** `y_high = (suicide >= 3)` → 132

## Ethics

Research use only. Not a clinical tool. Participant data must stay private.
