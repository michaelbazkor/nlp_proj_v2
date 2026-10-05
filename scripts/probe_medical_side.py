"""Paired STM vs medical-auxiliary variants at the published STM hyperparameters.

The suicide development AUC chooses the checkpoint. Test AUC is only logged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import continue_pw15 as base
import train_pw15_mtm as saved
from ssr.config import load_config
from ssr.data.cohort import build_cohort
from ssr.fusion.project import collect_user_blocks
from ssr.models.stm import MTM, STM, _act, _fc_stack
from ssr.train.cv import PSYCHIATRIC, _make_splits, _zscore_fit
from ssr.train.metrics import auc_roc, mtm_loss, stm_loss

OUT_DIR = ROOT / "artifacts" / "pw_1.5_mtm_side"
# Published pw_1.5 STM winners.
WINNERS = {
    0: {"n_layers": 3, "n_neurons": 1024, "activation": "tanh", "lr": 0.01, "epochs": 1000},
    1: {"n_layers": 1, "n_neurons": 64, "activation": "tanh", "lr": 0.005, "epochs": 1000},
    2: {"n_layers": 3, "n_neurons": 1024, "activation": "tanh", "lr": 0.005, "epochs": 5000},
    3: {"n_layers": 1, "n_neurons": 512, "activation": "tanh", "lr": 0.001, "epochs": 5000},
    4: {"n_layers": 1, "n_neurons": 512, "activation": "tanh", "lr": 0.05, "epochs": 2500},
}


class UngatedMTM(MTM):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(x) if len(list(self.shared.children())) else x
        pers = self.personality(shared)
        psy = self.psychosocial(torch.cat([pers, shared], dim=-1))
        psych = self.psychiatric(torch.cat([psy, shared], dim=-1))
        sui = self.suicide(torch.cat([psych, shared], dim=-1)).squeeze(-1)
        return {
            "suicide_logit": sui,
            "personality": pers,
            "psychosocial": psy,
            "psychiatric": psych,
        }


class SidePsych(nn.Module):
    """STM suicide path, plus a psychiatric head on the same trunk."""

    def __init__(self, in_dim: int, n_layers: int, n_neurons: int, activation: str):
        super().__init__()
        self.trunk = _fc_stack(in_dim, n_layers, n_neurons, activation)
        h = n_neurons if n_layers > 0 else in_dim
        self.head = nn.Linear(h, 1)
        self.psych = nn.Sequential(nn.Linear(h, n_neurons), _act(activation), nn.Linear(n_neurons, 2))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x) if len(list(self.trunk.children())) else x
        return {"suicide_logit": self.head(h).squeeze(-1), "psychiatric": self.psych(h)}


class MedicalCascade(nn.Module):
    """Trunk, then PHQ-9/GAD, then suicide on tanh(those scores) plus the trunk."""

    def __init__(self, in_dim: int, n_layers: int, n_neurons: int, activation: str):
        super().__init__()
        self.shared = _fc_stack(in_dim, n_layers, n_neurons, activation)
        h = n_neurons if n_layers > 0 else in_dim
        self.psychiatric = nn.Sequential(nn.Linear(h, n_neurons), _act(activation), nn.Linear(n_neurons, 2))
        self.suicide = nn.Sequential(nn.Linear(2 + h, n_neurons), _act(activation), nn.Linear(n_neurons, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(x) if len(list(self.shared.children())) else x
        psych = self.psychiatric(shared)
        sui = self.suicide(torch.cat([torch.tanh(psych), shared], dim=-1)).squeeze(-1)
        return {"suicide_logit": sui, "psychiatric": psych}


def build(name: str, in_dim: int, params: dict) -> nn.Module:
    args = (in_dim, int(params["n_layers"]), int(params["n_neurons"]), params["activation"])
    if name == "stm":
        return STM(*args)
    if name == "side":
        return SidePsych(*args)
    if name == "medical":
        return MedicalCascade(*args)
    if name == "full":
        return UngatedMTM(*args)
    raise ValueError(name)


def train(name: str, scale: float, params: dict, tensors: dict, seed: int, device: str):
    torch.manual_seed(seed)
    model = build(name, tensors["Xtr"].shape[1], params).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=float(params["lr"]), momentum=0.9)
    Xtr, ytr = tensors["Xtr"], tensors["ytr"]
    n, bs = Xtr.shape[0], min(32, Xtr.shape[0])
    best, bad, state = -1.0, 0, None
    for _ in range(int(params["epochs"])):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            ii = perm[s : s + bs]
            opt.zero_grad()
            out = model(Xtr[ii])
            if name == "stm":
                loss = stm_loss(out["suicide_logit"], ytr[ii])
            elif name == "full":
                loss, _ = mtm_loss(out, ytr[ii], tensors["pers"][ii], tensors["psy"][ii], tensors["psych"][ii])
                loss = stm_loss(out["suicide_logit"], ytr[ii]) + scale * (loss - stm_loss(out["suicide_logit"], ytr[ii]))
            else:
                loss = stm_loss(out["suicide_logit"], ytr[ii]) + scale * torch.nn.functional.mse_loss(
                    out["psychiatric"], tensors["psych"][ii]
                )
            if not torch.isfinite(loss):
                bad = 200
                break
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(tensors["Xdv"])["suicide_logit"].float().cpu().numpy()
        score = auc_roc(tensors["ydv_np"], 1 / (1 + np.exp(-np.clip(logits, -60, 60))))
        if np.isfinite(score) and score > best:
            best, bad, state = float(score), 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 200:
                break
    if state is not None:
        model.load_state_dict(state)
    return best, model


def score(model, X, y, psych_true, mu, device):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32, device=device))
        logits = out["suicide_logit"].float().cpu().numpy()
        auc = auc_roc(y, 1 / (1 + np.exp(-np.clip(logits, -60, 60))))
        phq = float("nan")
        if "psychiatric" in out:
            pred = out["psychiatric"].float().cpu().numpy() * mu[1] + mu[0]
            if np.std(pred[:, 0]) > 1e-8:
                phq = float(np.corrcoef(psych_true[:, 0], pred[:, 0])[0, 1])
    return float(auc), phq


def main() -> None:
    folds = [int(v) for v in sys.argv[1:]] or [1, 4, 3]
    variants = [
        ("stm", 0.0),
        ("side", 0.05),
        ("side", 0.10),
        ("side", 0.20),
        ("medical", 0.10),
        ("medical", 0.33),
        ("full", 0.10),
        ("full", 0.33),
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "eli_matrix.yaml")
    cohort = build_cohort(cfg, assert_paper=False)
    rep_roots = {m["name"]: (ROOT / "more stuff" / "reps" / m["name"]).resolve() for m in cfg.represent["models"]}
    user_ids = cohort["UserId"].tolist()
    print("loading", flush=True)
    blocks = {uid: collect_user_blocks(rep_roots, uid) for uid in user_ids}
    y_all = cohort["y_high"].to_numpy()
    splits = list(_make_splits(y_all, 5, cfg.seed, float(cfg.train["train_frac"]), float(cfg.train["dev_frac"])))
    idx = cohort.set_index("UserId")
    for fold_i, tr, dv, te in splits:
        if fold_i not in folds:
            continue
        done_path = OUT_DIR / f"fold{fold_i}.json"
        if done_path.exists():
            print(f"fold {fold_i} exists", flush=True)
            continue
        params = WINNERS[fold_i]
        device = "cuda:0" if params["n_neurons"] >= 512 and torch.cuda.is_available() else "cpu"

        def ids(ix):
            return [user_ids[i] for i in ix]

        fusion = saved.load_fusion(ROOT / "artifacts" / "pw_1.5" / f"fusion_fold{fold_i}.pt")

        def pack(ix):
            return fusion.transform_many([blocks[u] for u in ids(ix)]), idx.loc[ids(ix)]

        Xtr, str_ = pack(tr)
        Xdv, sdv = pack(dv)
        Xte, ste = pack(te)
        mu = _zscore_fit(str_[PSYCHIATRIC].to_numpy().astype(np.float32))

        def zpsych(frame):
            return (frame[PSYCHIATRIC].to_numpy().astype(np.float32) - mu[0]) / mu[1]

        from ssr.train.cv import PERSONALITY, PSYCHOSOCIAL

        def zcols(frame, cols):
            m, s = _zscore_fit(str_[cols].to_numpy().astype(np.float32))
            return (frame[cols].to_numpy().astype(np.float32) - m) / s

        fd = {
            "Xtr": Xtr,
            "ytr": str_["y_high"].to_numpy().astype(np.float32),
            "Xdv": Xdv,
            "ydv": sdv["y_high"].to_numpy().astype(np.float32),
            "pers": zcols(str_, PERSONALITY),
            "psy": zcols(str_, PSYCHOSOCIAL),
            "psych": zpsych(str_),
        }
        tensors = base.to_tensors(fd, device)
        yte = ste["y_high"].to_numpy().astype(np.float32)
        true = ste[PSYCHIATRIC].to_numpy().astype(np.float32)
        # Match the STM search seed for this fold's winning index.
        layers, neurons, acts, lrs, epochs = [1, 2, 3], [16, 32, 64, 128, 256, 512, 1024], ["tanh", "sigmoid"], [0.001, 0.005, 0.01, 0.05], [1000, 2500, 5000]
        cfg_idx = (
            epochs.index(params["epochs"])
            + 3 * (
                lrs.index(params["lr"])
                + 4 * (
                    acts.index(params["activation"])
                    + 2 * (neurons.index(params["n_neurons"]) + 7 * layers.index(params["n_layers"]))
                )
            )
        )
        seed = base.config_seed(cfg.seed, fold_i, "stm", cfg_idx)
        rows = []
        for name, scale in variants:
            print(f"fold {fold_i} {name} scale={scale} on {device}", flush=True)
            dev, model = train(name, scale, params, tensors, seed, device)
            te, phq = score(model, Xte, yte, true, mu, device)
            row = {"name": name, "scale": scale, "dev_auc": dev, "test_auc": te, "pearson_phq9": phq}
            rows.append(row)
            print(f"  dev={dev:.3f} test={te:.3f} phq_r={phq:.3f}", flush=True)
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        done_path.write_text(json.dumps({"fold": fold_i, "params": params, "rows": rows}, indent=2), encoding="utf-8")
        del tensors


if __name__ == "__main__":
    main()
