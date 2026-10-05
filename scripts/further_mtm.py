"""Try to beat the saved config-123 MTM without replacing it.

Candidates are kept only when their mean development AUC exceeds the
saved model's. Test AUC is recorded after that comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import continue_pw15 as base
import train_pw15_mtm as saved
from reselect_mtm import UngatedMTM
from ssr.config import load_config
from ssr.data.cohort import build_cohort
from ssr.fusion.project import collect_user_blocks
from ssr.train.cv import PERSONALITY, PSYCHIATRIC, PSYCHOSOCIAL, _make_splits, _zscore_fit
from ssr.train.metrics import auc_roc

OUT_DIR = ROOT / "artifacts" / "pw_1.5_mtm_further"
SHARED = ROOT / "artifacts" / "pw_1.5_mtm_shared"
PARAMS = {"n_layers": 1, "n_neurons": 512, "activation": "tanh", "lr": 0.005, "epochs": 1000}
CFG_IDX = 123
BASELINE_DEV = [0.7284980744544287, 0.8207103123662816, 0.8299101412066753, 0.795250320924262, 0.7351305091998289]
BASELINE_TEST = [0.7286813186813187, 0.7130693912303107, 0.7674542358450405, 0.7317639257294429, 0.6896551724137931]


def probs_from_logits(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))


def forward_logits(model, X, device: str) -> np.ndarray:
    model.eval()
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(X)["suicide_logit"].float().cpu().numpy()


def make_model(in_dim: int, device: str, state=None) -> UngatedMTM:
    model = UngatedMTM(in_dim, PARAMS["n_layers"], PARAMS["n_neurons"], PARAMS["activation"]).to(device)
    if state is not None:
        model.load_state_dict(state)
    return model


def suite_loss(out, y, pers, psy, psych, pos_weight):
    sui = F.binary_cross_entropy_with_logits(out["suicide_logit"], y, pos_weight=pos_weight)
    return (
        sui
        + F.mse_loss(out["personality"], pers)
        + F.mse_loss(out["psychosocial"], psy)
        + F.mse_loss(out["psychiatric"], psych)
    )


def fit(model, tensors, pos_weight, lr, epochs, patience, device):
    opt = torch.optim.RMSprop([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9)
    Xtr, ytr = tensors["Xtr"], tensors["ytr"]
    n, bs = Xtr.shape[0], min(32, Xtr.shape[0])
    best, bad, state = -1.0, 0, None
    ydv = tensors["ydv_np"]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            ii = perm[s : s + bs]
            opt.zero_grad()
            out = model(Xtr[ii])
            loss = suite_loss(out, ytr[ii], tensors["pers"][ii], tensors["psy"][ii], tensors["psych"][ii], pos_weight)
            if not torch.isfinite(loss):
                bad = patience
                break
            loss.backward()
            opt.step()
        score = auc_roc(ydv, probs_from_logits(forward_logits(model, tensors["Xdv"], device)))
        if np.isfinite(score) and score > best:
            best, bad = float(score), 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if state is not None:
        model.load_state_dict(state)
    return best, model


def pos_weight_of(y: torch.Tensor) -> torch.Tensor:
    n_pos = float(y.sum().item())
    n_neg = float(y.shape[0] - n_pos)
    return torch.tensor([n_neg / n_pos * 1.5], dtype=torch.float32, device=y.device)


def main() -> None:
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rows = []
    for fold_i, tr, dv, te in splits:
        done_path = OUT_DIR / f"fold{fold_i}.json"
        if done_path.exists():
            rows.append(json.loads(done_path.read_text(encoding="utf-8")))
            print(f"fold {fold_i} cached", flush=True)
            continue

        def ids(ix):
            return [user_ids[i] for i in ix]

        fusion = saved.load_fusion(ROOT / "artifacts" / "pw_1.5" / f"fusion_fold{fold_i}.pt")

        def pack(ix):
            return fusion.transform_many([blocks[u] for u in ids(ix)]), idx.loc[ids(ix)]

        Xtr, str_ = pack(tr)
        Xdv, sdv = pack(dv)
        Xte, ste = pack(te)
        ydv = sdv["y_high"].to_numpy().astype(np.float32)
        yte = ste["y_high"].to_numpy().astype(np.float32)
        mu = {
            key: _zscore_fit(str_[cols].to_numpy().astype(np.float32))
            for key, cols in (("pers", PERSONALITY), ("psy", PSYCHOSOCIAL), ("psych", PSYCHIATRIC))
        }

        def z(frame, key, cols):
            m, s = mu[key]
            return (frame[cols].to_numpy().astype(np.float32) - m) / s

        fd = {
            "Xtr": Xtr,
            "ytr": str_["y_high"].to_numpy().astype(np.float32),
            "Xdv": Xdv,
            "ydv": ydv,
            "pers": z(str_, "pers", PERSONALITY),
            "psy": z(str_, "psy", PSYCHOSOCIAL),
            "psych": z(str_, "psych", PSYCHIATRIC),
        }
        tensors = base.to_tensors(fd, device)
        pw = pos_weight_of(tensors["ytr"])
        base_seed = base.config_seed(cfg.seed, fold_i, "mtm", CFG_IDX)
        saved_state = torch.load(SHARED / f"mtm_high_fold{fold_i}.pt", map_location="cpu", weights_only=False)["state_dict"]

        torch.manual_seed(base_seed)
        model = make_model(Xtr.shape[1], device)
        dev_pw, model = fit(model, tensors, pw, PARAMS["lr"], PARAMS["epochs"], 200, device)
        te_pw = float(auc_roc(yte, probs_from_logits(forward_logits(model, Xte, device))))
        print(f"fold {fold_i} pos_weight dev={dev_pw:.3f} test={te_pw:.3f}", flush=True)
        del model

        model = make_model(Xtr.shape[1], device, saved_state)
        for name, mod in model.named_parameters():
            if not name.startswith("suicide"):
                mod.requires_grad = False
        dev_head, model = fit(model, tensors, pw, 0.001, 200, 40, device)
        te_head = float(auc_roc(yte, probs_from_logits(forward_logits(model, Xte, device))))
        print(f"fold {fold_i} head dev={dev_head:.3f} test={te_head:.3f}", flush=True)
        del model

        seed_dev, seed_te = [], []
        dev_probs, te_probs = [], []
        for k in range(5):
            torch.manual_seed(base_seed + k * 10007)
            model = make_model(Xtr.shape[1], device)
            dev_k, model = fit(model, tensors, torch.tensor([1.0], device=device), PARAMS["lr"], PARAMS["epochs"], 200, device)
            dev_p = probs_from_logits(forward_logits(model, Xdv, device))
            te_p = probs_from_logits(forward_logits(model, Xte, device))
            dev_k = float(auc_roc(ydv, dev_p))
            te_k = float(auc_roc(yte, te_p))
            seed_dev.append(dev_k)
            seed_te.append(te_k)
            dev_probs.append(dev_p)
            te_probs.append(te_p)
            print(f"fold {fold_i} seed {k} dev={dev_k:.3f} test={te_k:.3f}", flush=True)
            del model
        ens_dev = float(auc_roc(ydv, np.mean(dev_probs, axis=0)))
        ens_te = float(auc_roc(yte, np.mean(te_probs, axis=0)))
        print(f"fold {fold_i} seed-ensemble dev={ens_dev:.3f} test={ens_te:.3f}", flush=True)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        rec = {
            "fold": fold_i,
            "baseline_dev": BASELINE_DEV[fold_i],
            "baseline_test": BASELINE_TEST[fold_i],
            "pos_weight_dev": dev_pw,
            "pos_weight_test": te_pw,
            "head_dev": dev_head,
            "head_test": te_head,
            "seed_dev": seed_dev,
            "seed_test": seed_te,
            "seed_ensemble_dev": ens_dev,
            "seed_ensemble_test": ens_te,
        }
        done_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        rows.append(rec)
    summary = {
        "baseline_mean_dev": float(np.mean(BASELINE_DEV)),
        "baseline_mean_test": float(np.mean(BASELINE_TEST)),
        "pos_weight_mean_dev": float(np.mean([r["pos_weight_dev"] for r in rows])),
        "pos_weight_mean_test": float(np.mean([r["pos_weight_test"] for r in rows])),
        "head_mean_dev": float(np.mean([r["head_dev"] for r in rows])),
        "head_mean_test": float(np.mean([r["head_test"] for r in rows])),
        "seed_ensemble_mean_dev": float(np.mean([r["seed_ensemble_dev"] for r in rows])),
        "seed_ensemble_mean_test": float(np.mean([r["seed_ensemble_test"] for r in rows])),
        "folds": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"baseline dev={summary['baseline_mean_dev']:.3f} test={summary['baseline_mean_test']:.3f} | "
        f"pos dev={summary['pos_weight_mean_dev']:.3f} test={summary['pos_weight_mean_test']:.3f} | "
        f"head dev={summary['head_mean_dev']:.3f} test={summary['head_mean_test']:.3f} | "
        f"seeds dev={summary['seed_ensemble_mean_dev']:.3f} test={summary['seed_ensemble_mean_test']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
