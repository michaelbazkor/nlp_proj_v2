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
| `cohort` | Filter users, build `y_general` / `y_high` |
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

## Labels (validated against the paper)

- Cohort: `status_posts >= 10` and `grp in {0,1}` → N≈1006
- General risk: `suicide >= 1` → 361
- High risk: `suicide >= 3` → 132

## Ethics

Research use only. Not a clinical tool. Participant data must stay private.
