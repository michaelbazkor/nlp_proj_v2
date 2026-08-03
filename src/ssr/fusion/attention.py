"""Task-aware attention pooling over position vectors within each (model, layer) group."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ssr.train.metrics import auc_roc

POSITION_ORDER = ["input_only", "last_prompt", "cot", "final_pred"]


def _parse_block_key(key: str) -> tuple[str, str, str]:
    model, layer, position = key.rsplit(":", 2)
    return model, layer, position


def _group_block_keys(block_keys: list[str]) -> list[tuple[str, str]]:
    groups: dict[tuple[str, str], list[str]] = {}
    for key in block_keys:
        model, layer, position = _parse_block_key(key)
        groups.setdefault((model, layer), []).append(position)

    def _layer_sort(layer: str) -> tuple[int, str]:
        try:
            return (int(layer), layer)
        except ValueError:
            return (10**9, layer)

    return sorted(groups.keys(), key=lambda g: (g[0], _layer_sort(g[1])))


def _group_dims(
    train_blocks: list[dict[str, np.ndarray]], group_keys: list[tuple[str, str]]
) -> list[int]:
    dims: list[int] = []
    for model, layer in group_keys:
        for position in POSITION_ORDER:
            key = f"{model}:{layer}:{position}"
            for blocks in train_blocks:
                if key in blocks:
                    dims.append(int(blocks[key].shape[0]))
                    break
            else:
                dims.append(0)
            if dims[-1] > 0:
                break
        if dims[-1] == 0:
            raise ValueError(f"Could not infer hidden dim for group {(model, layer)}")
    return dims


def _stack_groups(
    blocks_list: list[dict[str, np.ndarray]],
    group_keys: list[tuple[str, str]],
    group_dims: list[int],
    means: dict[str, np.ndarray],
    stds: dict[str, np.ndarray],
) -> list[np.ndarray]:
    if not blocks_list:
        raise ValueError("empty blocks_list")
    n = len(blocks_list)
    p = len(POSITION_ORDER)
    pos_idx = {name: i for i, name in enumerate(POSITION_ORDER)}
    out_list: list[np.ndarray] = []
    for (model, layer), dim in zip(group_keys, group_dims):
        out = np.zeros((n, p, dim), dtype=np.float32)
        for ui, blocks in enumerate(blocks_list):
            for position in POSITION_ORDER:
                key = f"{model}:{layer}:{position}"
                if key not in blocks:
                    continue
                vec = blocks[key].astype(np.float32)
                out[ui, pos_idx[position]] = (vec - means[key]) / stds[key]
        out_list.append(out)
    return out_list


class _AttentionFusionNet(nn.Module):
    def __init__(self, group_dims: list[int], out_dim: int):
        super().__init__()
        self.scorers = nn.ModuleList([nn.Linear(d, 1, bias=False) for d in group_dims])
        self.proj = nn.Linear(sum(group_dims), out_dim)
        self.probe = nn.Linear(out_dim, 1)

    def _pool(self, xs: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        pooled: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for scorer, x in zip(self.scorers, xs):
            scores = scorer(x).squeeze(-1)
            w = torch.softmax(scores, dim=-1)
            pooled.append((w.unsqueeze(-1) * x).sum(dim=1))
            weights.append(w)
        return torch.cat(pooled, dim=-1), weights

    def forward(self, xs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        flat, weights = self._pool(xs)
        features = self.proj(flat)
        logits = self.probe(features).squeeze(-1)
        return logits, features, weights

    def encode(self, xs: list[torch.Tensor]) -> torch.Tensor:
        flat, _ = self._pool(xs)
        return self.proj(flat)


@dataclass
class AttentionFusionFit:
    group_keys: list[tuple[str, str]]
    group_dims: list[int]
    block_keys: list[str]
    means: dict[str, np.ndarray]
    stds: dict[str, np.ndarray]
    target_dim: int
    state_dict: dict[str, torch.Tensor]
    k_per_block: int = 0

    @property
    def out_dim(self) -> int:
        return self.target_dim

    def transform(self, blocks: dict[str, np.ndarray]) -> np.ndarray:
        arrs = _stack_groups([blocks], self.group_keys, self.group_dims, self.means, self.stds)
        net = _AttentionFusionNet(self.group_dims, self.target_dim)
        net.load_state_dict(self.state_dict)
        net.eval()
        with torch.no_grad():
            xs = [torch.tensor(a, dtype=torch.float32) for a in arrs]
            out = net.encode(xs).squeeze(0).numpy()
        return out.astype(np.float32)


def _zscore_fit_blocks(
    train_blocks: list[dict[str, np.ndarray]], block_keys: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    for key in block_keys:
        X = np.stack([b[key] for b in train_blocks], axis=0).astype(np.float64)
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        means[key] = mu.astype(np.float32)
        stds[key] = sd.astype(np.float32)
    return means, stds


def fit_attention_fusion(
    train_blocks: list[dict[str, np.ndarray]],
    train_y: np.ndarray,
    dev_blocks: list[dict[str, np.ndarray]],
    dev_y_eval: np.ndarray,
    *,
    target_dim: int = 1024,
    lr: float = 0.01,
    epochs: int = 500,
    patience: int = 50,
    seed: int = 0,
    device: str = "cpu",
    train_loss: str = "bce",
) -> AttentionFusionFit:
    """Fit attention pooling; train_y is fusion target, dev_y_eval selects best checkpoint."""
    if not train_blocks:
        raise ValueError("empty train_blocks")
    block_keys = sorted(train_blocks[0].keys())
    group_keys = _group_block_keys(block_keys)
    group_dims = _group_dims(train_blocks, group_keys)
    means, stds = _zscore_fit_blocks(train_blocks, block_keys)

    torch.manual_seed(seed)
    X_tr = _stack_groups(train_blocks, group_keys, group_dims, means, stds)
    X_dv = _stack_groups(dev_blocks, group_keys, group_dims, means, stds)
    y_tr = train_y.astype(np.float32)
    y_dv_eval = dev_y_eval.astype(np.float32)

    net = _AttentionFusionNet(group_dims, target_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    if train_loss == "mse":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1.0)], device=device
            )
        )

    X_tr_t = [torch.tensor(x, device=device) for x in X_tr]
    X_dv_t = [torch.tensor(x, device=device) for x in X_dv]
    y_tr_t = torch.tensor(y_tr, device=device)

    best_auc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    bad = 0

    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        logits, _, _ = net(X_tr_t)
        loss = loss_fn(logits, y_tr_t)
        loss.backward()
        opt.step()

        net.eval()
        with torch.no_grad():
            dv_logits, _, _ = net(X_dv_t)
            if train_loss == "mse":
                dv_scores = dv_logits.cpu().numpy()
            else:
                dv_scores = torch.sigmoid(dv_logits).cpu().numpy()
        auc = auc_roc(y_dv_eval, dv_scores)
        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}

    return AttentionFusionFit(
        group_keys=group_keys,
        group_dims=group_dims,
        block_keys=block_keys,
        means=means,
        stds=stds,
        target_dim=target_dim,
        state_dict=best_state,
    )
