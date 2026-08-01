# Suicide-risk representations (SSR)

LLM hidden-state representations → STM/MTM classifiers for suicide-risk prediction,
modernizing Ophir et al. (2020).

## Quick start (POC on CPU)

```bash
pip install -e .
python -m ssr.cli --config configs/poc.yaml all --skip-captions
```

## Stages

| Command | What it does |
|---------|--------------|
| `cohort` | Filter users, build `y_high` (suicide >= 3) |
| `posts` | Decode & clean Facebook posts |
| `captions` | VLM image captions (cached by blob GUID) |
| `corpus` | Build per-user `[post] ...` corpora |
| `represent` | Extract hidden-state blocks from LLMs |
| `train` | 5-fold CV STM/MTM + metrics |
| `all` | Run the full pipeline |
| `report` | Write `reports/stage_examples.md` |

## Configs

- `configs/poc.yaml` — 30 users, tiny models, CPU
- `configs/real.yaml` — full cohort, Llama-3.3-70B / Qwen3-32B / R1-Distill / Gemma, H100

## Labels / cohort filter

```python
df_users = df_metadata[(grp.isin([0, 1])) & (status_posts > 9)]
             .sort_values(by=['status_posts', 'sui_cat']).reset_index()
df_profiles = df_profiles[df_profiles['UserId'].isin(df_users['UserId'])]
```

- Yields **N=1006** on this CSV (paper text says 1002; post total **83292** matches the paper).
- **High risk only:** `y_high = (suicide >= 3)` → 132
- STM/MTM train and evaluate on `y_high` only

## Ethics

Research use only. Not a clinical tool. Participant data must stay private.
