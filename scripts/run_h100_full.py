#!/usr/bin/env python3
"""H100 full-cohort pipeline — one command, two phases.

Phase 1 (represent): cohort → posts → captions → corpus → LLM hidden states (.npz)
  - Models loaded sequentially; VRAM cleared between each
  - FlashAttention-2 + BF16 (32B) / 4-bit NF4 (70B on single H100)

Phase 2 (train): 8-experiment matrix on cached reps (~15–30 min)
  - 2 fusion (PCA, attention) × 2 train target (ordinal, binary) × 2 arch (STM, MTM)
  - All evaluated on binary high-risk (>=3)

Usage:
  pip install -e ".[h100]"
  export HF_TOKEN=...          # required for gated models (Llama)
  python scripts/run_h100_full.py
  python scripts/run_h100_full.py --phase train     # reps already cached
  python scripts/run_h100_full.py --skip-captions   # text-only corpus
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssr.config import load_config
from ssr.data.cohort import build_cohort
from ssr.data.corpus import build_corpora
from ssr.data.posts import build_posts, cap_posts_per_user
from ssr.io_utils import atomic_write_text
from ssr.represent.extract import run_representations
from ssr.train.matrix import format_comparison_md, run_train_matrix


def _phase1(cfg, *, skip_captions: bool, force: bool):
    t0 = time.time()
    print("\n" + "=" * 60)
    print("PHASE 1: represent (LLM hidden-state extraction)")
    print("=" * 60)

    cohort = build_cohort(cfg, assert_paper=True)
    print(f"[cohort] n={len(cohort)} high={int(cohort.y_high.sum())} ({100*cohort.y_high.mean():.1f}%)")

    posts = build_posts(cfg, set(cohort["UserId"]))
    posts = cap_posts_per_user(
        posts,
        cfg.posts.get("max_posts_per_user"),
        int(cfg.posts.get("shuffle_seed", cfg.seed)),
    )
    out = cfg.exp_dir("posts_capped.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    posts.to_parquet(out, index=False)
    print(f"[posts] rows={len(posts)} users={posts.UserId.nunique()}")

    captions = {}
    if skip_captions or not cfg.corpus.get("include_images", True):
        print("[captions] skipped")
    else:
        from ssr.caption.run import run_captions

        captions = run_captions(cfg, posts, force=force)
        print(f"[captions] n={len(captions)}")

    corpora = build_corpora(cfg, cohort, posts, captions, force=force)
    print(f"[corpus] users={len(corpora)} mean_chars={corpora.n_chars.mean():.0f}")

    rep_roots = run_representations(cfg, corpora, force=force)
    for name, root in rep_roots.items():
        n = len(list(root.glob("*.npz")))
        print(f"[represent] {name}: {n} user files")

    elapsed = (time.time() - t0) / 3600
    print(f"\n[phase1] done in {elapsed:.2f} h")
    return cohort, rep_roots


def _phase2(cfg, cohort, rep_roots):
    t0 = time.time()
    print("\n" + "=" * 60)
    print("PHASE 2: train (8-experiment matrix)")
    print("=" * 60)

    results = run_train_matrix(cfg, cohort, rep_roots)
    md = format_comparison_md(results)
    md_path = cfg.report("h100_matrix_comparison.md")
    atomic_write_text(md_path, md)
    print(f"\n[phase2] comparison written to {md_path}")

    elapsed = (time.time() - t0) / 60
    print(f"[phase2] done in {elapsed:.1f} min")
    return results


def _load_rep_roots(cfg) -> dict[str, Path]:
    roots = {}
    for mcfg in cfg.represent["models"]:
        root = cfg.exp_dir("reps", mcfg["name"])
        if not root.exists():
            raise FileNotFoundError(f"Missing reps for {mcfg['name']}: {root}")
        roots[mcfg["name"]] = root
    return roots


def main():
    ap = argparse.ArgumentParser(description="H100 full cohort pipeline")
    ap.add_argument("--config", default="configs/h100_full.yaml")
    ap.add_argument(
        "--phase",
        choices=["all", "represent", "train"],
        default="all",
        help="all=phase1+phase2, represent=phase1 only, train=phase2 only",
    )
    ap.add_argument("--skip-captions", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-extract reps even if cached")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    print(f"[h100] config={args.config} experiment={cfg.experiment}")

    cohort = None
    rep_roots = None

    if args.phase in ("all", "represent"):
        cohort, rep_roots = _phase1(cfg, skip_captions=args.skip_captions, force=args.force)

    if args.phase in ("all", "train"):
        if cohort is None:
            cohort = build_cohort(cfg, assert_paper=True)
        if rep_roots is None:
            rep_roots = _load_rep_roots(cfg)
        _phase2(cfg, cohort, rep_roots)

    print("\n[h100] pipeline complete.")


if __name__ == "__main__":
    main()
