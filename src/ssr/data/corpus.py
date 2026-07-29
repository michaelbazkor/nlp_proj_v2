"""Corpus builder: assemble per-user text for LLM representation extraction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ssr.config import Config
from ssr.io_utils import atomic_write_json, exists_nonempty


def _format_post(free_text: str, image_descs: list[str], post_marker: str, image_marker: str) -> str:
    text = (free_text or "").strip()
    if image_descs:
        joined = " ".join(d.strip() for d in image_descs if d and d.strip())
        if text:
            body = f"{text} {image_marker} {joined}"
        else:
            body = f"{image_marker} {joined}"
    else:
        body = text
    return f"{post_marker} {body}".strip()


def _subject_prefix(row: pd.Series, marker: str) -> str:
    # gender_label already corrected (female col == male)
    parts = [
        f"Gender: {row.get('gender_label', 'Unknown')}",
        f"Age: {row.get('age', 'NA')}",
        f"Num_Friends: {row.get('FriendCount', 'NA')}",
    ]
    if pd.notna(row.get("educ")):
        parts.append(f"Educ: {row.get('educ')}")
    if pd.notna(row.get("inc_num")):
        parts.append(f"Income: {row.get('inc_num')}")
    return f"{marker} " + ", ".join(parts)


def build_corpora(
    cfg: Config,
    cohort: pd.DataFrame,
    posts: pd.DataFrame,
    captions: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return DataFrame with columns UserId, corpus, n_posts, n_images, n_chars."""
    out = cfg.exp_dir("corpora.parquet")
    meta_path = cfg.exp_dir("corpora_meta.json")
    if exists_nonempty(out):
        return pd.read_parquet(out)

    captions = captions or {}
    include_images = bool(cfg.corpus.get("include_images", True))
    include_subj = bool(cfg.corpus.get("include_subject_details", False))
    post_marker = cfg.corpus.get("post_marker", "[post]")
    image_marker = cfg.corpus.get("image_marker", "IMAGE_DESCRIPTIONS:")
    subject_marker = cfg.corpus.get("subject_marker", "[subject details]")
    seed = int(cfg.posts.get("shuffle_seed", cfg.seed))
    max_posts = cfg.posts.get("max_posts_per_user")
    max_images = cfg.images.get("max_images_per_user")

    rng = np.random.default_rng(seed)
    cohort_idx = cohort.set_index("UserId")
    rows = []

    for uid, g in posts.groupby("UserId"):
        if uid not in cohort_idx.index:
            continue
        g = g.copy()
        # Treat posts as unordered set: shuffle then optionally cap
        order = rng.permutation(len(g))
        g = g.iloc[order].reset_index(drop=True)
        if max_posts is not None:
            g = g.head(int(max_posts))

        post_strs: list[str] = []
        n_images = 0
        images_used = 0
        for _, prow in g.iterrows():
            descs: list[str] = []
            if include_images and prow.get("has_image"):
                # Prefer PostId-based image_key (matches pics.zip); fall back to blob_guid
                key = prow.get("image_key") or prow.get("blob_guid")
                if key and not (isinstance(key, float) and __import__("math").isnan(key)):
                    key = str(key)
                    if max_images is not None and images_used >= int(max_images):
                        pass
                    elif key in captions:
                        descs.append(captions[key])
                        images_used += 1
                        n_images += 1
            text = str(prow.get("free_text") or "")
            if not text.strip() and not descs:
                continue
            post_strs.append(_format_post(text, descs, post_marker, image_marker))

        corpus = " ".join(post_strs)
        if include_subj:
            corpus = _subject_prefix(cohort_idx.loc[uid], subject_marker) + " " + corpus

        rows.append(
            {
                "UserId": uid,
                "corpus": corpus,
                "n_posts": len(post_strs),
                "n_images": n_images,
                "n_chars": len(corpus),
            }
        )

    df = pd.DataFrame(rows)
    # Ensure all cohort users present (empty corpus if no posts)
    missing = set(cohort["UserId"]) - set(df["UserId"])
    for uid in missing:
        df = pd.concat(
            [df, pd.DataFrame([{"UserId": uid, "corpus": "", "n_posts": 0, "n_images": 0, "n_chars": 0}])],
            ignore_index=True,
        )

    meta = {
        "experiment": cfg.experiment,
        "n_users": int(len(df)),
        "include_images": include_images,
        "include_subject_details": include_subj,
        "mean_posts": float(df["n_posts"].mean()) if len(df) else 0,
        "mean_chars": float(df["n_chars"].mean()) if len(df) else 0,
        "mean_images": float(df["n_images"].mean()) if len(df) else 0,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    atomic_write_json(meta_path, meta)
    return df
