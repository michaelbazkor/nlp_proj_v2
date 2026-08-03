"""Fusion method dispatch (PCA vs task-aware attention pooling)."""
from __future__ import annotations

from typing import Any

import numpy as np

from ssr.fusion.attention import AttentionFusionFit, fit_attention_fusion
from ssr.fusion.project import FusionFit, fit_fusion


def fit_fusion_method(
    method: str,
    train_blocks: list[dict[str, np.ndarray]],
    *,
    target_dim: int = 1024,
    dev_blocks: list[dict[str, np.ndarray]] | None = None,
    train_y: np.ndarray | None = None,
    dev_y: np.ndarray | None = None,
    dev_y_eval: np.ndarray | None = None,
    attention_cfg: dict[str, Any] | None = None,
    seed: int = 0,
    train_loss: str = "bce",
) -> FusionFit | AttentionFusionFit:
    if method in (None, "", "per_block_pca", "pca"):
        return fit_fusion(train_blocks, target_dim=target_dim)
    if method in ("attention_pool", "attention"):
        if dev_blocks is None or train_y is None:
            raise ValueError("attention_pool requires dev_blocks and train_y")
        eval_y = dev_y_eval if dev_y_eval is not None else dev_y
        if eval_y is None:
            raise ValueError("attention_pool requires dev_y or dev_y_eval")
        cfg = attention_cfg or {}
        return fit_attention_fusion(
            train_blocks,
            train_y,
            dev_blocks,
            eval_y,
            target_dim=target_dim,
            lr=float(cfg.get("lr", 0.01)),
            epochs=int(cfg.get("epochs", 500)),
            patience=int(cfg.get("patience", 50)),
            seed=seed,
            device=str(cfg.get("device", "cpu")),
            train_loss=train_loss,
        )
    raise ValueError(f"Unknown fusion method: {method!r}")
