"""Fold-safe standardization + per-block PCA projection to target_dim."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class BlockSpec:
    key: str  # e.g. "qwen3_0_6b:12:input_only"
    dim: int


@dataclass
class FusionFit:
    block_keys: list[str]
    means: dict[str, np.ndarray]
    stds: dict[str, np.ndarray]
    pcas: dict[str, PCA]
    k_per_block: int
    target_dim: int
    out_dim: int

    def transform(self, blocks: dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for key in self.block_keys:
            x = blocks[key].astype(np.float64)
            mu = self.means[key]
            sd = self.stds[key]
            x = (x - mu) / sd
            z = self.pcas[key].transform(x.reshape(1, -1))[0]
            # Pad if PCA produced fewer components than k_per_block
            if z.shape[0] < self.k_per_block:
                pad = np.zeros(self.k_per_block - z.shape[0], dtype=np.float64)
                z = np.concatenate([z, pad])
            parts.append(z[: self.k_per_block])
        out = np.concatenate(parts, axis=0)
        # Pad / truncate to target_dim
        if out.shape[0] < self.target_dim:
            out = np.concatenate([out, np.zeros(self.target_dim - out.shape[0])])
        return out[: self.target_dim].astype(np.float32)


def _allocate_k(n_blocks: int, target_dim: int, n_train: int) -> int:
    if n_blocks <= 0:
        return target_dim
    k = max(1, target_dim // n_blocks)
    # Clamp to n_train - 1 (PCA rank)
    k = min(k, max(1, n_train - 1))
    return k


def fit_fusion(
    train_blocks: list[dict[str, np.ndarray]],
    target_dim: int = 1024,
) -> FusionFit:
    """Fit z-score + per-block PCA on the training users only.

    train_blocks: list over users, each a dict key -> (dim,) vector.
    Keys must be identical across users.
    """
    if not train_blocks:
        raise ValueError("empty train_blocks")
    block_keys = sorted(train_blocks[0].keys())
    n_train = len(train_blocks)
    k = _allocate_k(len(block_keys), target_dim, n_train)

    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    pcas: dict[str, PCA] = {}

    for key in block_keys:
        X = np.stack([b[key] for b in train_blocks], axis=0).astype(np.float64)
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xz = (X - mu) / sd
        n_comp = min(k, Xz.shape[0], Xz.shape[1])
        n_comp = max(1, n_comp)
        pca = PCA(n_components=n_comp, random_state=0)
        pca.fit(Xz)
        means[key] = mu
        stds[key] = sd
        pcas[key] = pca

    out_dim = k * len(block_keys)
    return FusionFit(
        block_keys=block_keys,
        means=means,
        stds=stds,
        pcas=pcas,
        k_per_block=k,
        target_dim=target_dim,
        out_dim=min(out_dim, target_dim) if out_dim >= target_dim else out_dim,
    )


def collect_user_blocks(
    rep_roots: dict[str, Path],
    user_id: str,
) -> dict[str, np.ndarray]:
    """Load all (model, layer, position) blocks for one user."""
    from ssr.represent.store import load_user_reps

    blocks: dict[str, np.ndarray] = {}
    safe = str(user_id).replace("/", "_").replace(" ", "")
    for model_name, root in rep_roots.items():
        path = root / f"{safe}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        arrays = load_user_reps(path)
        for key, vec in arrays.items():
            if key.startswith("__"):
                continue
            blocks[f"{model_name}:{key}"] = vec
    return blocks
