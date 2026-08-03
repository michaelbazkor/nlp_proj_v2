"""Losses and metrics for STM/MTM training."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def stm_loss(suicide_logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(suicide_logit, y.float())


def stm_loss_ordinal(suicide_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(suicide_pred, y.float())


def mtm_loss(
    outputs: dict[str, torch.Tensor],
    y_suicide: torch.Tensor,
    y_personality: torch.Tensor,
    y_psychosocial: torch.Tensor,
    y_psychiatric: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """BCE on suicide + sum of MSEs on auxiliary targets (already z-scored)."""
    L_sui = F.binary_cross_entropy_with_logits(outputs["suicide_logit"], y_suicide.float())
    L_pers = F.mse_loss(outputs["personality"], y_personality)
    L_psy = F.mse_loss(outputs["psychosocial"], y_psychosocial)
    L_psych = F.mse_loss(outputs["psychiatric"], y_psychiatric)
    # Paper: L = L_suicide + L_aux where L_aux = sum_a MSE (with 1/2N factor absorbed)
    total = L_sui + L_pers + L_psy + L_psych
    parts = {
        "loss_suicide": float(L_sui.detach()),
        "loss_personality": float(L_pers.detach()),
        "loss_psychosocial": float(L_psy.detach()),
        "loss_psychiatric": float(L_psych.detach()),
    }
    return total, parts


def mtm_loss_ordinal(
    outputs: dict[str, torch.Tensor],
    y_suicide: torch.Tensor,
    y_personality: torch.Tensor,
    y_psychosocial: torch.Tensor,
    y_psychiatric: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    L_sui = F.mse_loss(outputs["suicide_logit"], y_suicide.float())
    L_pers = F.mse_loss(outputs["personality"], y_personality)
    L_psy = F.mse_loss(outputs["psychosocial"], y_psychosocial)
    L_psych = F.mse_loss(outputs["psychiatric"], y_psychiatric)
    total = L_sui + L_pers + L_psy + L_psych
    parts = {
        "loss_suicide": float(L_sui.detach()),
        "loss_personality": float(L_pers.detach()),
        "loss_psychosocial": float(L_psy.detach()),
        "loss_psychiatric": float(L_psych.detach()),
    }
    return total, parts


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def f1_at_05(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_score) >= 0.5).astype(int)
    return float(f1_score(y_true, pred, zero_division=0))


def cohens_d_from_auc(auc: float) -> float:
    """Salgado (2018) conversion used in the paper: d = √2 * Φ^{-1}(AUC)."""
    from scipy.stats import norm

    if not np.isfinite(auc):
        return float("nan")
    auc = min(max(auc, 1e-6), 1 - 1e-6)
    return float(np.sqrt(2.0) * norm.ppf(auc))


def summarize_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    auc = auc_roc(y_true, y_score)
    return {
        "auc_roc": auc,
        "pr_auc": pr_auc(y_true, y_score),
        "f1": f1_at_05(y_true, y_score),
        "cohens_d": cohens_d_from_auc(auc) if np.isfinite(auc) else float("nan"),
        "n": int(len(y_true)),
        "n_pos": int(np.sum(y_true)),
    }
