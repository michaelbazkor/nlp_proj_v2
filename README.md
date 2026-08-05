# Suicide-risk representations (SSR)

LLM hidden-state representations → STM/MTM classifiers for suicide-risk prediction,
modernizing Ophir et al. (2020).

## Quick start (POC on CPU)

```bash
pip install -e .
PYTHONPATH=src python -m ssr.cli --config configs/poc.yaml all --skip-captions
# with images (SmolVLM):
PYTHONPATH=src python -m ssr.cli --config configs/poc_standard.yaml all --force
```

## Real H100 run (full cohort)

POC artifacts must not poison the full run. Wipe shared caches once, then use a
`real_*` config (non-colliding experiment names):

```bash
# One-time: remove POC-contaminated global caches
rm -f artifacts/posts.parquet artifacts/posts_meta.json
rm -rf artifacts/captions.parquet artifacts/captions_meta.json artifacts/captions/

# Standard (text + Qwen2.5-VL captions + 4 LLMs)
PYTHONPATH=src python -m ssr.cli --config configs/real.yaml all --force
```

Ablations (same code path):

| Config | Experiment dir | Notes |
|--------|----------------|-------|
| `configs/real.yaml` | `artifacts/real_standard/` | Images + all LLMs |
| `configs/real_text_only.yaml` | `artifacts/real_text_only/` | No images |
| `configs/real_subject_details.yaml` | `artifacts/real_subject_details/` | Demographics prefix |
| `configs/real_llama_only.yaml` | `artifacts/llama_only/` | Llama-3.3-70B only |

POC uses `experiment: text_only` / `standard` — do **not** reuse those names for H100.

### Cache hygiene

| Cache | Scope | Risk |
|-------|-------|------|
| `artifacts/posts.parquet` | Global (full table) | Rebuild with `--force` if n_users looks POC-sized |
| `artifacts/captions/{model_slug}/` | Per VLM | Never reuses SmolVLM for Qwen2.5-VL |
| `artifacts/{experiment}/corpora.parquet` | Per experiment | `--force` rebuilds |
| `artifacts/{experiment}/reps/{model}/` | Per experiment+model | `--force` rebuilds |

The CLI prints `[preflight] WARNING` if posts/corpora look POC-contaminated.

### Large-model loading

`dtype: bfloat16`, `attn_implementation: sdpa`. Unknown dtypes (e.g. `float8_e4m3fn`)
raise instead of silently falling back to float32.

**Do not use `device_map: auto` on the Blackwell node.** Sharding a model across the
four RTX PRO 6000 cards produced numerically corrupt activations (NaN/Inf blocks and
degenerate generations such as endless `.`) for both Llama-3.3-70B and Qwen3-32B.
`configs/real_full.yaml` therefore pins one model per GPU (`device: cuda:0`,
`device_map: null`) and runs Llama-3.3-70B with `load_in_8bit: true`, since it does not
fit a single 98 GB card in bf16.

Thinking models (Qwen3 / DeepSeek-R1) use `max_new_tokens: 512`; others 256.
`max_context: 32768` — longer corpora are split on `[post]` boundaries and averaged.

### Full-cohort run (1003 users, 4 LLMs)

```bash
# captions: batched + one shard per GPU (~6x the serial captioner)
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python scripts/caption_shard.py \
    --config configs/real_full.yaml --shard $i --num-shards 4 --batch-size 48 & done
bash scripts/run_full_pipeline.sh   # merge -> corpus -> represent -> train -> report
```

| Script | Purpose |
|--------|---------|
| `scripts/caption_shard.py` | Batched VLM captioning, resumable, one shard per GPU |
| `scripts/merge_caption_shards.py` | Folds shards into the model-scoped caption cache |
| `scripts/run_full_pipeline.sh` | Drives all remaining stages, rebalances GPUs |
| `scripts/train_parallel.py` | 504-point grid x 5 folds x 2 variants across GPUs+CPUs |
| `scripts/make_report.py` | Writes `reports/{experiment}/final_report.md` |
| `scripts/validate_taps.py` | Proves hook-based extraction matches the old code bit-for-bit |

Representation extraction taps hidden states with forward hooks
(`ssr.represent.extract._LayerTaps`) instead of `output_hidden_states=True`; the latter
materializes every layer for the whole prompt (tens of GB at full corpus length) and
needed two prefills per chunk.

## Stages

| Command | What it does |
|---------|--------------|
| `cohort` | Filter users, build `y_high` (suicide >= 3) |
| `posts` | Decode & clean Facebook posts (full table cached) |
| `captions` | VLM image captions (model-scoped cache) |
| `corpus` | Build per-user `[post]` / `[image]` corpora |
| `represent` | Extract hidden-state blocks from LLMs |
| `train` | 5-fold CV STM/MTM + metrics |
| `all` | Run the full pipeline |
| `report` | Write `reports/{experiment}/stage_examples.md` |

Flags: `--force` rebuilds caches; `--skip-captions` for text-only.

## Configs

- `configs/poc.yaml` — 30 users, tiny models, CPU, text-only
- `configs/poc_standard.yaml` — 30 users + SmolVLM captions
- `configs/real.yaml` — full cohort, Llama-3.3-70B / Qwen3-32B / R1-Distill / Gemma, H100
- `configs/real_text_only.yaml` / `real_subject_details.yaml` / `real_llama_only.yaml` — paper ablations

## Labels / cohort filter

```python
df_users = df_metadata[(grp.isin([0, 1])) & (status_posts > 9)]
             .sort_values(by=['status_posts', 'sui_cat']).reset_index()
df_profiles = df_profiles[df_profiles['UserId'].isin(df_users['UserId'])]
```

- Yields **N=1006** on this CSV (paper text says 1002; post total **83292** matches the paper).
- After dropping null labels/aux: **~1003** training users.
- **High risk only:** `y_high = (suicide >= 3)` → 132
- STM/MTM train and evaluate on `y_high` only

## Ethics

Research use only. Not a clinical tool. Participant data must stay private.
