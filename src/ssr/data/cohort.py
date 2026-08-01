"""Cohort builder: filter users and create binary suicide-risk labels.

User filter (matches the paper pipeline code):
  df_users = df_metadata[(grp.isin([0, 1])) & (status_posts > 9)]
               .sort_values(by=['status_posts', 'sui_cat']).reset_index()

On this CSV that yields N=1006 with sum(status_posts)=83292 (paper's reported
post count). Paper text says N=1002 — likely a typo; we follow the filter code.
High risk: y_high = (suicide >= 3) → 132 (matches paper).
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

# Checks against the filter code + this CSV (not the paper's N=1002 typo).
PAPER_CHECKS = {
    "n_users": 1006,
    "n_high": 132,
    "sum_status_posts": 83292,
}


def load_users(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def filter_users(df_metadata: pd.DataFrame) -> pd.DataFrame:
    """Exact paper-pipeline user filter + sort."""
    df_users = (
        df_metadata[
            (df_metadata["grp"].isin([0, 1])) & (df_metadata["status_posts"] > 9)
        ]
        .sort_values(by=["status_posts", "sui_cat"])
        .reset_index(drop=True)
    )
    return df_users


def build_cohort(cfg: Config, *, assert_paper: bool = True) -> pd.DataFrame:
    """Return filtered cohort with y_high labels."""
    n_users = cfg.cohort.get("n_users")
    tag = f"n{n_users}" if n_users is not None else "full"
    out = cfg.art(f"cohort_{tag}.parquet")
    meta_path = cfg.art(f"cohort_{tag}_meta.json")

    if exists_nonempty(out) and exists_nonempty(meta_path):
        return pd.read_parquet(out)

    u = load_users(cfg.paths.user_csv)
    c_all = filter_users(u)

    paper_meta = {
        "n_users_filtered": int(len(c_all)),
        "sum_status_posts": int(c_all["status_posts"].sum()),
        "n_high": int((c_all["suicide"] >= 3).sum()),
        "n_sui_cat_null": int(c_all["sui_cat"].isna().sum()),
        "n_suicide_null": int(c_all["suicide"].isna().sum()),
    }
    if assert_paper and cfg.cohort.get("n_users") is None:
        assert paper_meta["n_users_filtered"] == PAPER_CHECKS["n_users"], paper_meta
        assert paper_meta["n_high"] == PAPER_CHECKS["n_high"], paper_meta
        assert paper_meta["sum_status_posts"] == PAPER_CHECKS["sum_status_posts"], paper_meta

    # Keep labeled users for training (null suicide/sui_cat cannot form y_high)
    c = c_all[c_all["suicide"].notna()].copy()
    for col in AUX_TARGETS:
        if col not in c.columns:
            raise KeyError(f"Missing auxiliary target column: {col}")
    c = c.dropna(subset=AUX_TARGETS).copy()

    c["y_high"] = (c["suicide"] >= 3).astype(int)
    # Kept for diagnostics / optional subject-details; not used for training.
    c["y_general"] = (c["suicide"] >= 1).astype(int)

    # Gender: column named `female` but mean matches paper "% male"
    # so 1 => Male, 0 => Female for experiment-3 subject details.
    c["gender_label"] = c["female"].map({1.0: "Male", 0.0: "Female", 1: "Male", 0: "Female"})
    c["gender_label"] = c["gender_label"].fillna("Unknown")

    meta: dict[str, Any] = {
        **paper_meta,
        "n_users": int(len(c)),
        "n_high_labeled": int(c["y_high"].sum()),
        "mean_status_posts": float(c["status_posts"].mean()),
        "std_status_posts": float(c["status_posts"].std()),
        "mean_female_col": float(c["female"].mean()),
        "pos_rate_high": float(c["y_high"].mean()),
    }

    # Optional stratified subsample for POC
    if n_users is not None:
        c = _stratified_sample(c, int(n_users), cfg.seed, cfg.cohort.get("stratify_on", "y_high"))
        meta["n_users_sampled"] = int(len(c))
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
    sizes = {k: len(g) for k, g in groups}
    total = sum(sizes.values())
    alloc = {k: max(1, int(round(n * sizes[k] / total))) for k in sizes}
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
