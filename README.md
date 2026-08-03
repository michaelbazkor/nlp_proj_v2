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

Real configs set `device_map: auto`, `dtype: bfloat16`, `attn_implementation: sdpa`.
Unknown dtypes (e.g. `float8_e4m3fn`) raise instead of silently falling back to float32.

Thinking models (Qwen3 / DeepSeek-R1) use `max_new_tokens: 512`; others 256.
Context window is `131072` (no artificial 4096 clamp beyond config).

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
