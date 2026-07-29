"""Facebook posts loader.

Handles non-clean UTF-8 (`encoding_errors='replace'`), drops malformed rows,
and selects posts that have text and/or an image attachment.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from ssr.config import Config
from ssr.io_utils import atomic_write_json, exists_nonempty

BLOB_GUID_RE = re.compile(
    r"attachments/([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)


def blob_guid_from_url(url: str | None) -> str | None:
    if not url or not isinstance(url, str):
        return None
    m = BLOB_GUID_RE.search(url)
    if m:
        return m.group(1).lower()
    # Fallback: last path segment
    try:
        return Path(urlparse(url).path).name.lower() or None
    except Exception:
        return None


def load_raw_posts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace",
        dtype=str,
    )
    # Drop malformed rows (embedded newlines broke Revoked into free text)
    df = df[df["Revoked"].isin(["0", "1"])].copy()
    return df


def build_posts(cfg: Config, user_ids: set[str] | None = None) -> pd.DataFrame:
    """Build cleaned post table for the (optional) user subset."""
    out = cfg.art("posts.parquet")
    meta_path = cfg.art("posts_meta.json")
    if exists_nonempty(out) and exists_nonempty(meta_path):
        posts = pd.read_parquet(out)
        if user_ids is not None:
            return posts[posts["UserId"].isin(user_ids)].reset_index(drop=True)
        return posts

    raw = load_raw_posts(cfg.paths.posts_csv)
    if user_ids is not None:
        raw = raw[raw["UserId"].isin(user_ids)].copy()

    include_desc = bool(cfg.posts.get("include_post_description", False))
    include_title = bool(cfg.posts.get("include_attachment_title", False))

    def free_text(row: pd.Series) -> str:
        parts: list[str] = []
        msg = row.get("PostMessage")
        if isinstance(msg, str) and msg.strip():
            parts.append(msg.strip())
        if include_desc:
            desc = row.get("PostDescription")
            if isinstance(desc, str) and desc.strip():
                parts.append(desc.strip())
        if include_title:
            title = row.get("PostAttachmentTitle")
            if isinstance(title, str) and title.strip():
                parts.append(title.strip())
        return " ".join(parts)

    raw["free_text"] = raw.apply(free_text, axis=1)
    raw["blob_url"] = raw["PostAttachmentBlobUrl"].where(
        raw["PostAttachmentBlobUrl"].notna() & (raw["PostAttachmentBlobUrl"].str.len() > 0),
        other=None,
    )
    raw["blob_guid"] = raw["blob_url"].map(blob_guid_from_url)
    raw["has_text"] = raw["free_text"].str.len() > 0
    raw["has_image"] = raw["blob_guid"].notna()

    if cfg.posts.get("require_text_or_image", True):
        raw = raw[raw["has_text"] | raw["has_image"]].copy()

    keep = [
        "UserId",
        "PostId",
        "PostType",
        "PostStatusType",
        "free_text",
        "blob_url",
        "blob_guid",
        "has_text",
        "has_image",
        "Revoked",
        "PostFromOwner",
        "Fixed",
    ]
    posts = raw[keep].reset_index(drop=True)

    meta: dict[str, Any] = {
        "n_rows": int(len(posts)),
        "n_users": int(posts["UserId"].nunique()),
        "n_with_text": int(posts["has_text"].sum()),
        "n_with_image": int(posts["has_image"].sum()),
        "n_unique_images": int(posts["blob_guid"].nunique(dropna=True)),
        "post_type_counts": posts["PostType"].value_counts(dropna=False).head(10).to_dict(),
    }
    posts.to_parquet(out, index=False)
    atomic_write_json(meta_path, meta)

    if user_ids is not None:
        return posts[posts["UserId"].isin(user_ids)].reset_index(drop=True)
    return posts


def cap_posts_per_user(posts: pd.DataFrame, max_posts: int | None, seed: int) -> pd.DataFrame:
    if max_posts is None:
        return posts
    rng = __import__("numpy").random.default_rng(seed)

    def _cap(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) <= max_posts:
            return g
        idx = rng.choice(g.index.to_numpy(), size=max_posts, replace=False)
        return g.loc[sorted(idx)]

    return posts.groupby("UserId", group_keys=False).apply(_cap).reset_index(drop=True)
