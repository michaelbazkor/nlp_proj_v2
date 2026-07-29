"""Representation store: per-user float16 npz caches."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ssr.io_utils import exists_nonempty


def rep_path(root: Path, model_name: str, user_id: str) -> Path:
    # Sanitize user id for filesystem
    safe = str(user_id).replace("/", "_").replace(" ", "")
    return root / model_name / f"{safe}.npz"


def save_user_reps(path: Path, arrays: dict[str, np.ndarray], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez_compressed appends .npz if missing — use a temp stem carefully
    tmp = path.parent / (path.stem + ".partial.npz")
    payload = {k: np.asarray(v, dtype=np.float16) for k, v in arrays.items()}
    meta_json = np.array([str(meta)])
    np.savez_compressed(tmp, __meta__=meta_json, **payload)
    # np may write exactly `tmp` already ending in .npz
    written = tmp if tmp.exists() else Path(str(tmp) + ".npz")
    written.replace(path)


def load_user_reps(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].astype(np.float32) for k in z.files if k != "__meta__"}


def has_user_reps(path: Path) -> bool:
    return exists_nonempty(path)
