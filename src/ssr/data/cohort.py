"""Cohort builder: filter users and create binary suicide-risk labels.

Validated against Ophir et al. (2020):
  status_posts > 9 & grp in {0,1}  -> N ~= 1006 (paper 1002)
  y_general = (suicide >= 1)       -> 361 (paper 361)
  y_high    = (suicide >= 3)       -> 132 (paper 132)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ssr.config import Config
from ssr.io_utils import atomic_write_json, exists_nonempty

AUX_TARGETS = [
    "PHQ9",
    "GAD",
    "Brooding",
    "Worry",
    "SWL",
    "Lonely",
    "BFI_O",
    "BFI_C",
    "BFI_E",
    "BFI_A",
    "BFI_N",
]

PAPER_CHECKS = {
    "n_users_min": 1000,
    "n_users_max": 1010,
    "n_general": 361,
    "n_high": 132,
    "sum_status_posts": 83292,
}


def load_users(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def build_cohort(cfg: Config, *, assert_paper: bool = True) -> pd.DataFrame:
    """Return filtered cohort with y_general / y_high labels."""
    n_users = cfg.cohort.get("n_users")
    tag = f"n{n_users}" if n_users is not None else "full"
    out = cfg.art(f"cohort_{tag}.parquet")
    meta_path = cfg.art(f"cohort_{tag}_meta.json")

    if exists_nonempty(out) and exists_nonempty(meta_path):
        return pd.read_parquet(out)

    u = load_users(cfg.paths.user_csv)
    min_posts = int(cfg.cohort.get("min_status_posts", 10))
    groups = set(cfg.cohort.get("groups", [0, 1]))

    # status_posts > 9  <=>  status_posts >= min_status_posts when min=10
    mask = (u["status_posts"] >= min_posts) & (u["grp"].isin(groups))
    c_all = u.loc[mask].copy()

    # Paper-check BEFORE dropping null labels (sum_status_posts == 83292 at N=1006)
    paper_meta = {
        "n_users_prefilter": int(len(c_all)),
        "sum_status_posts": int(c_all["status_posts"].sum()),
        "n_general": int((c_all["suicide"] >= 1).sum()),
        "n_high": int((c_all["suicide"] >= 3).sum()),
    }
    if assert_paper and cfg.cohort.get("n_users") is None:
        assert PAPER_CHECKS["n_users_min"] <= paper_meta["n_users_prefilter"] <= PAPER_CHECKS["n_users_max"] + 10, paper_meta
        assert paper_meta["n_general"] == PAPER_CHECKS["n_general"], paper_meta
        assert paper_meta["n_high"] == PAPER_CHECKS["n_high"], paper_meta
        assert paper_meta["sum_status_posts"] == PAPER_CHECKS["sum_status_posts"], paper_meta

    c = c_all[c_all["suicide"].notna()].copy()
    # Aux targets must be present for MTM
    for col in AUX_TARGETS:
        if col not in c.columns:
            raise KeyError(f"Missing auxiliary target column: {col}")
    c = c.dropna(subset=AUX_TARGETS).copy()

    c["y_general"] = (c["suicide"] >= 1).astype(int)
    c["y_high"] = (c["suicide"] >= 3).astype(int)

    # Gender: column named `female` but mean matches paper "% male"
    # so 1 => Male, 0 => Female for experiment-3 subject details.
    c["gender_label"] = c["female"].map({1.0: "Male", 0.0: "Female", 1: "Male", 0: "Female"})
    c["gender_label"] = c["gender_label"].fillna("Unknown")

    meta: dict[str, Any] = {
        **paper_meta,
        "n_users": int(len(c)),
        "n_general_labeled": int(c["y_general"].sum()),
        "n_high_labeled": int(c["y_high"].sum()),
        "mean_status_posts": float(c["status_posts"].mean()),
        "std_status_posts": float(c["status_posts"].std()),
        "mean_female_col": float(c["female"].mean()),
        "pos_rate_general": float(c["y_general"].mean()),
        "pos_rate_high": float(c["y_high"].mean()),
    }

    # Optional stratified subsample for POC
    n_users = cfg.cohort.get("n_users")
    if n_users is not None:
        c = _stratified_sample(c, int(n_users), cfg.seed, cfg.cohort.get("stratify_on", "y_general"))
        meta["n_users_sampled"] = int(len(c))
        meta["n_general_sampled"] = int(c["y_general"].sum())
        meta["n_high_sampled"] = int(c["y_high"].sum())

    keep = [
        "UserId",
        "status_posts",
        "posts",
        "FriendCount",
        "age",
        "female",
        "gender_label",
        "educ",
        "inc_num",
        "grp",
        "suicide",
        "sui_cat",
        "y_general",
        "y_high",
        *AUX_TARGETS,
    ]
    c = c[keep].reset_index(drop=True)

    c.to_parquet(out, index=False)
    atomic_write_json(meta_path, meta)
    return c


def _stratified_sample(df: pd.DataFrame, n: int, seed: int, strat_col: str) -> pd.DataFrame:
    """Sample n users stratified on strat_col, preserving class balance as much as possible."""
    rng = np.random.default_rng(seed)
    groups = list(df.groupby(strat_col, sort=True))
    # Allocate proportional, then fill remainder
    sizes = {k: len(g) for k, g in groups}
    total = sum(sizes.values())
    alloc = {k: max(1, int(round(n * sizes[k] / total))) for k in sizes}
    # Fix sum to n
    while sum(alloc.values()) > n:
        k = max(alloc, key=lambda x: alloc[x] - n * sizes[x] / total)
        if alloc[k] > 1:
            alloc[k] -= 1
        else:
            break
    while sum(alloc.values()) < n:
        k = max(sizes, key=lambda x: sizes[x] - alloc[x])
        if alloc[k] < sizes[k]:
            alloc[k] += 1
        else:
            break

    parts = []
    for k, g in groups:
        take = min(alloc[k], len(g))
        idx = rng.choice(g.index.to_numpy(), size=take, replace=False)
        parts.append(df.loc[idx])
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out.head(n)
