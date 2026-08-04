"""5-fold stratified CV with grid search, STM/MTM training.

Supports:
  - train_target: high (binary) | ordinal (suicide 0-6)
  - eval_target: high (binary >=3) for dev model selection and test metrics
  - fusion.method: per_block_pca | attention_pool
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from ssr.config import Config
from ssr.fusion.project import collect_user_blocks
from ssr.fusion.registry import fit_fusion_method
from ssr.io_utils import atomic_write_json
from ssr.models import MTM, STM
from ssr.train.metrics import (
    auc_roc,
    mtm_loss,
    mtm_loss_ordinal,
    stm_loss,
    stm_loss_ordinal,
    summarize_scores,
)

PERSONALITY = MTM.PERSONALITY
PSYCHOSOCIAL = MTM.PSYCHOSOCIAL
PSYCHIATRIC = MTM.PSYCHIATRIC


@dataclass
class FoldData:
    X: np.ndarray
    y_train: np.ndarray
    y_eval: np.ndarray
    aux: dict[str, np.ndarray]
    user_ids: list[str]


def _resolve_targets(train_cfg: dict[str, Any]) -> tuple[str, str]:
    train_target = str(train_cfg.get("train_target", "high"))
    eval_target = str(train_cfg.get("eval_target", "high"))
    if train_target not in ("high", "ordinal"):
        raise ValueError(f"train_target must be high|ordinal (got {train_target!r})")
    if eval_target != "high":
        raise ValueError(f"Only eval_target=high is supported (got {eval_target!r})")
    return train_target, eval_target


def _label_columns(train_target: str) -> tuple[str, str]:
    if train_target == "ordinal":
        return "suicide", "y_high"
    return "y_high", "y_high"


def _make_splits(y: np.ndarray, n_folds: int, seed: int, train_frac: float, dev_frac: float):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    indices = np.arange(len(y))
    for fold_i, (rest_idx, test_idx) in enumerate(skf.split(indices, y)):
        y_rest = y[rest_idx]
        try:
            train_idx, dev_idx = train_test_split(
                rest_idx,
                test_size=dev_frac / (train_frac + dev_frac),
                stratify=y_rest,
                random_state=seed + fold_i,
            )
        except ValueError:
            train_idx, dev_idx = train_test_split(
                rest_idx,
                test_size=dev_frac / (train_frac + dev_frac),
                random_state=seed + fold_i,
            )
        yield fold_i, train_idx, dev_idx, test_idx


def _zscore_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def _train_one(
    model: torch.nn.Module,
    is_mtm: bool,
    train: FoldData,
    dev: FoldData,
    *,
    lr: float,
    epochs: int,
    batch_size: int,
    momentum: float,
    patience: int,
    aux_mu: dict[str, np.ndarray],
    aux_sd: dict[str, np.ndarray],
    ordinal: bool,
    device: str = "cpu",
) -> tuple[torch.nn.Module, float]:
    model = model.to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum)

    def _pack(fd: FoldData):
        X = torch.tensor(fd.X, dtype=torch.float32)
        y_train = torch.tensor(fd.y_train, dtype=torch.float32)
        if not is_mtm:
            return TensorDataset(X, y_train), None
        pers = torch.tensor(
            (fd.aux["personality"] - aux_mu["personality"]) / aux_sd["personality"],
            dtype=torch.float32,
        )
        psy = torch.tensor(
            (fd.aux["psychosocial"] - aux_mu["psychosocial"]) / aux_sd["psychosocial"],
            dtype=torch.float32,
        )
        psych = torch.tensor(
            (fd.aux["psychiatric"] - aux_mu["psychiatric"]) / aux_sd["psychiatric"],
            dtype=torch.float32,
        )
        return TensorDataset(X, y_train, pers, psy, psych), None

    train_ds, _ = _pack(train)
    loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)

    best_auc = -1.0
    best_state = None
    bad = 0

    for _ in range(epochs):
        model.train()
        for batch in loader:
            opt.zero_grad()
            if is_mtm:
                xb, yb, pers, psy, psych = [t.to(device) for t in batch]
                out = model(xb)
                if ordinal:
                    loss, _ = mtm_loss_ordinal(out, yb, pers, psy, psych)
                else:
                    loss, _ = mtm_loss(out, yb, pers, psy, psych)
            else:
                xb, yb = [t.to(device) for t in batch]
                out = model(xb)
                if ordinal:
                    loss = stm_loss_ordinal(out["suicide_logit"], yb)
                else:
                    loss = stm_loss(out["suicide_logit"], yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            xd = torch.tensor(dev.X, dtype=torch.float32, device=device)
            raw = model(xd)["suicide_logit"].cpu().numpy()
            scores = raw if ordinal else 1 / (1 + np.exp(-raw))
        auc = auc_roc(dev.y_eval, scores)
        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_auc) if best_auc >= 0 else float("nan")


def _predict(model: torch.nn.Module, X: np.ndarray, *, ordinal: bool, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xd = torch.tensor(X, dtype=torch.float32, device=device)
        raw = model(xd)["suicide_logit"].cpu().numpy()
    return raw if ordinal else 1 / (1 + np.exp(-raw))


def run_training(cfg: Config, cohort: pd.DataFrame, rep_roots: dict[str, Path]) -> dict[str, Any]:
    train_cfg = cfg.train
    target_dim = int(cfg.fusion.get("target_dim", 1024))
    fusion_method = str(cfg.fusion.get("method", "per_block_pca"))
    attention_cfg = cfg.fusion.get("attention") or {}
    train_target, eval_target = _resolve_targets(train_cfg)
    train_col, eval_col = _label_columns(train_target)
    ordinal = train_target == "ordinal"
    fusion_train_loss = "mse" if ordinal else "bce"

    n_folds = int(train_cfg["n_folds"])
    seed = cfg.seed
    grid = train_cfg["grid"]
    variants = train_cfg["variants"]
    batch_size = int(train_cfg.get("batch_size", 32))
    momentum = float(train_cfg.get("momentum", 0.9))
    patience = int(train_cfg.get("early_stop_patience", 50))

    user_ids = cohort["UserId"].tolist()
    y_strat = cohort[eval_col].to_numpy()

    print(
        f"[train] n={len(user_ids)} train_target={train_target} eval_target={eval_target} "
        f"fusion={fusion_method}"
    )
    all_blocks = {uid: collect_user_blocks(rep_roots, uid) for uid in user_ids}

    results: dict[str, Any] = {
        "train_target": train_target,
        "eval_target": eval_target,
        "fusion_method": fusion_method,
        "folds": [],
        "summary": {},
    }
    out_dir = cfg.exp_dir("train")
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        model_kind = variant["model"]
        key = f"{model_kind}_{train_target}"
        fold_metrics = []

        for fold_i, tr_idx, dv_idx, te_idx in _make_splits(
            y_strat,
            n_folds,
            seed,
            float(train_cfg["train_frac"]),
            float(train_cfg["dev_frac"]),
        ):
            tr_ids = [user_ids[i] for i in tr_idx]
            dv_ids = [user_ids[i] for i in dv_idx]
            te_ids = [user_ids[i] for i in te_idx]

            tr_blocks = [all_blocks[u] for u in tr_ids]
            dv_blocks = [all_blocks[u] for u in dv_ids]
            sub_tr = cohort.set_index("UserId").loc[tr_ids]
            sub_dv = cohort.set_index("UserId").loc[dv_ids]

            fusion = fit_fusion_method(
                fusion_method,
                tr_blocks,
                target_dim=target_dim,
                dev_blocks=dv_blocks,
                train_y=sub_tr[train_col].to_numpy().astype(np.float32),
                dev_y=sub_dv[train_col].to_numpy().astype(np.float32),
                dev_y_eval=sub_dv[eval_col].to_numpy().astype(np.float32),
                attention_cfg=attention_cfg,
                seed=seed + fold_i,
                train_loss=fusion_train_loss,
            )

            def _xy(ids):
                X = np.stack([fusion.transform(all_blocks[u]) for u in ids], axis=0)
                sub = cohort.set_index("UserId").loc[ids]
                return FoldData(
                    X=X,
                    y_train=sub[train_col].to_numpy().astype(np.float32),
                    y_eval=sub[eval_col].to_numpy().astype(np.float32),
                    aux={
                        "personality": sub[PERSONALITY].to_numpy().astype(np.float32),
                        "psychosocial": sub[PSYCHOSOCIAL].to_numpy().astype(np.float32),
                        "psychiatric": sub[PSYCHIATRIC].to_numpy().astype(np.float32),
                    },
                    user_ids=ids,
                )

            train_fd = _xy(tr_ids)
            dev_fd = _xy(dv_ids)
            test_fd = _xy(te_ids)

            aux_mu, aux_sd = {}, {}
            for k in ("personality", "psychosocial", "psychiatric"):
                mu, sd = _zscore_fit(train_fd.aux[k])
                aux_mu[k] = mu
                aux_sd[k] = sd

            best = None
            param_grid = list(
                itertools.product(
                    grid["n_layers"],
                    grid["n_neurons"],
                    grid["activation"],
                    grid["lr"],
                    grid["epochs"],
                )
            )
            k_block = getattr(fusion, "k_per_block", "?")
            print(
                f"[train] {key} fold {fold_i}: grid={len(param_grid)} "
                f"train={len(tr_ids)} dev={len(dv_ids)} test={len(te_ids)} "
                f"in_dim={train_fd.X.shape[1]} k_block={k_block}"
            )

            for n_layers, n_neurons, activation, lr, epochs in param_grid:
                if model_kind == "stm":
                    m = STM(train_fd.X.shape[1], n_layers, n_neurons, activation)
                    is_mtm = False
                else:
                    m = MTM(train_fd.X.shape[1], n_layers, n_neurons, activation)
                    is_mtm = True
                trained, dev_auc = _train_one(
                    m,
                    is_mtm,
                    train_fd,
                    dev_fd,
                    lr=float(lr),
                    epochs=int(epochs),
                    batch_size=batch_size,
                    momentum=momentum,
                    patience=patience,
                    aux_mu=aux_mu,
                    aux_sd=aux_sd,
                    ordinal=ordinal,
                )
                if best is None or (np.isfinite(dev_auc) and dev_auc > best["dev_auc"]):
                    best = {
                        "model": trained,
                        "dev_auc": dev_auc if np.isfinite(dev_auc) else -1.0,
                        "params": {
                            "n_layers": n_layers,
                            "n_neurons": n_neurons,
                            "activation": activation,
                            "lr": lr,
                            "epochs": epochs,
                        },
                        "is_mtm": is_mtm,
                    }

            assert best is not None
            scores = _predict(best["model"], test_fd.X, ordinal=ordinal)
            metrics = summarize_scores(test_fd.y_eval, scores)
            metrics.update(
                {
                    "fold": fold_i,
                    "dev_auc": best["dev_auc"],
                    **{f"p_{k}": v for k, v in best["params"].items()},
                }
            )
            fold_metrics.append(metrics)

            wpath = out_dir / f"{key}_fold{fold_i}.pt"
            torch.save(
                {
                    "state_dict": best["model"].state_dict(),
                    "params": best["params"],
                    "metrics": metrics,
                },
                wpath,
            )
            print(
                f"[train] {key} fold {fold_i}: test AUC={metrics['auc_roc']:.3f} "
                f"PR-AUC={metrics['pr_auc']:.3f} F1={metrics['f1']:.3f} d={metrics['cohens_d']:.3f}"
            )

        def _agg(name):
            vals = [m[name] for m in fold_metrics if np.isfinite(m.get(name, np.nan))]
            if not vals:
                return {"mean": float("nan"), "std": float("nan"), "ci95": [float("nan"), float("nan")]}
            arr = np.asarray(vals, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            se = std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
            return {
                "mean": mean,
                "std": std,
                "ci95": [mean - 1.96 * se, mean + 1.96 * se],
                "values": vals,
            }

        results["summary"][key] = {
            "auc_roc": _agg("auc_roc"),
            "pr_auc": _agg("pr_auc"),
            "f1": _agg("f1"),
            "cohens_d": _agg("cohens_d"),
            "folds": fold_metrics,
        }

    atomic_write_json(out_dir / "metrics.json", results)
    return results
