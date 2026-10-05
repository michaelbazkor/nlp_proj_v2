"""Inject PHQ-9 and GAD into the saved MTM, starting from its current weights.

New input weights begin at zero, so the model matches the saved network
until training finds a development-AUC improvement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_pw15_mtm as saved
from reselect_mtm import UngatedMTM
from ssr.config import load_config
from ssr.data.cohort import build_cohort
from ssr.fusion.project import collect_user_blocks
from ssr.train.cv import PERSONALITY, PSYCHIATRIC, PSYCHOSOCIAL, _make_splits, _zscore_fit
from ssr.train.metrics import auc_roc, mtm_loss

OUT_DIR = ROOT / "artifacts" / "pw_1.5_mtm_clinical"
MTM_DIR = ROOT / "artifacts" / "pw_1.5_mtm_shared"
FUSION_DIR = ROOT / "artifacts" / "pw_1.5"


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))


def ridge_phq_gad(Xtr, ytr, Xdv, Xte):
    xmu, xsd = Xtr.mean(0), Xtr.std(0)
    xsd = np.where(xsd < 1e-6, 1.0, xsd)

    def norm(X):
        return (X - xmu) / xsd

    out = []
    for j in range(ytr.shape[1]):
        model = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(norm(Xtr), ytr[:, j])
        raw = [model.predict(norm(X)).astype(np.float32) for X in (Xtr, Xdv, Xte)]
        mu, sd = float(raw[0].mean()), float(raw[0].std())
        if sd < 1e-6:
            sd = 1.0
        out.append([(v - mu) / sd for v in raw])
    return [np.column_stack([out[0][i], out[1][i]]) for i in range(3)]


def forward_logits(model, extra, x, clinical):
    pre = model.shared[0](x) + extra(clinical)
    shared = torch.tanh(pre)
    pers = model.personality(shared)
    psy = model.psychosocial(torch.cat([pers, shared], dim=-1))
    psych = model.psychiatric(torch.cat([psy, shared], dim=-1))
    sui = model.suicide(torch.cat([psych, shared], dim=-1)).squeeze(-1)
    return {"suicide_logit": sui, "personality": pers, "psychosocial": psy, "psychiatric": psych}


def numpy_logits(model, extra, X, C, device):
    model.eval()
    extra.eval()
    with torch.no_grad():
        out = forward_logits(
            model,
            extra,
            torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(C, dtype=torch.float32, device=device),
        )
    return out["suicide_logit"].float().cpu().numpy()


def fit_extra(model, extra, tensors, lr, epochs, patience, device):
    opt = torch.optim.RMSprop(extra.parameters(), lr=lr, momentum=0.9)
    Xtr, Ctr, ytr = tensors["Xtr"], tensors["Ctr"], tensors["ytr"]
    n, bs = Xtr.shape[0], min(32, Xtr.shape[0])
    ydv = tensors["ydv_np"]
    best = auc_roc(ydv, sigmoid(numpy_logits(model, extra, tensors["Xdv_np"], tensors["Cdv_np"], device)))
    state = {k: v.detach().cpu().clone() for k, v in extra.state_dict().items()}
    bad = 0
    for _ in range(epochs):
        extra.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            ii = perm[s : s + bs]
            opt.zero_grad()
            out = forward_logits(model, extra, Xtr[ii], Ctr[ii])
            loss, _ = mtm_loss(out, ytr[ii], tensors["pers"][ii], tensors["psy"][ii], tensors["psych"][ii])
            if not torch.isfinite(loss):
                bad = patience
                break
            loss.backward()
            opt.step()
        score = auc_roc(ydv, sigmoid(numpy_logits(model, extra, tensors["Xdv_np"], tensors["Cdv_np"], device)))
        if np.isfinite(score) and score > best:
            best = float(score)
            bad = 0
            state = {k: v.detach().cpu().clone() for k, v in extra.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    extra.load_state_dict(state)
    return float(best)


def main() -> None:
    cfg = load_config(ROOT / "eli_matrix.yaml")
    cohort = build_cohort(cfg, assert_paper=False)
    rep_roots = {m["name"]: (ROOT / "more stuff" / "reps" / m["name"]).resolve() for m in cfg.represent["models"]}
    user_ids = cohort["UserId"].tolist()
    print("loading", flush=True)
    blocks = {uid: collect_user_blocks(rep_roots, uid) for uid in user_ids}
    splits = list(_make_splits(cohort["y_high"].to_numpy(), 5, cfg.seed, float(cfg.train["train_frac"]), float(cfg.train["dev_frac"])))
    idx = cohort.set_index("UserId")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    folds = []
    for fold_i, tr, dv, te in splits:
        def ids(ix):
            return [user_ids[i] for i in ix]

        fusion = saved.load_fusion(FUSION_DIR / f"fusion_fold{fold_i}.pt")

        def pack(ix):
            return fusion.transform_many([blocks[u] for u in ids(ix)]), idx.loc[ids(ix)]

        Xtr, ftr = pack(tr)
        Xdv, fdv = pack(dv)
        Xte, fte = pack(te)
        mu = _zscore_fit(ftr[PSYCHIATRIC].to_numpy().astype(np.float32))
        y_psych = (ftr[PSYCHIATRIC].to_numpy().astype(np.float32) - mu[0]) / mu[1]
        Ctr, Cdv, Cte = ridge_phq_gad(Xtr, y_psych, Xdv, Xte)
        scale = float(Xtr.std())
        model = UngatedMTM(Xtr.shape[1], 1, 512, "tanh").to(device)
        ck = torch.load(MTM_DIR / f"mtm_high_fold{fold_i}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        for p in model.parameters():
            p.requires_grad = False
        extra = torch.nn.Linear(2, 512, bias=False).to(device)
        torch.nn.init.zeros_(extra.weight)
        aux = {}
        for key, cols in (("pers", PERSONALITY), ("psy", PSYCHOSOCIAL), ("psych", PSYCHIATRIC)):
            m, s = _zscore_fit(ftr[cols].to_numpy().astype(np.float32))
            aux[key] = torch.tensor((ftr[cols].to_numpy().astype(np.float32) - m) / s, dtype=torch.float32, device=device)
        tensors = {
            "Xtr": torch.tensor(Xtr, dtype=torch.float32, device=device),
            "Ctr": torch.tensor(Ctr * scale, dtype=torch.float32, device=device),
            "ytr": torch.tensor(ftr["y_high"].to_numpy().astype(np.float32), device=device),
            "Xdv_np": Xdv,
            "Cdv_np": Cdv * scale,
            "ydv_np": fdv["y_high"].to_numpy().astype(np.float32),
            **aux,
        }
        yte = fte["y_high"].to_numpy().astype(np.float32)
        base_te = numpy_logits(model, extra, Xte, Cte * scale, device)
        print(f"fold {fold_i} init test={auc_roc(yte, sigmoid(base_te)):.3f}", flush=True)
        dev = fit_extra(model, extra, tensors, 1e-4, 80, 20, device)
        te_logit = numpy_logits(model, extra, Xte, Cte * scale, device)
        dv_logit = numpy_logits(model, extra, Xdv, Cdv * scale, device)
        te = float(auc_roc(yte, sigmoid(te_logit)))
        print(f"  extra dev={dev:.3f} test={te:.3f} |w|={float(extra.weight.detach().abs().mean()):.5f}", flush=True)
        folds.append(
            {
                "ydv": tensors["ydv_np"],
                "yte": yte,
                "dv": dv_logit,
                "te": te_logit,
                "phq_dv": Cdv[:, 0],
                "phq_te": Cte[:, 0],
                "base_te": base_te,
            }
        )
        del model, extra
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    best_w, best_dev = 0.0, -1.0
    for w in np.linspace(-2, 2, 81):
        dev = float(np.mean([auc_roc(f["ydv"], sigmoid(f["dv"] + w * f["phq_dv"])) for f in folds]))
        if dev > best_dev:
            best_dev, best_w = dev, float(w)
    tests = [float(auc_roc(f["yte"], sigmoid(f["te"] + best_w * f["phq_te"]))) for f in folds]
    base_tests = [float(auc_roc(f["yte"], sigmoid(f["base_te"]))) for f in folds]
    summary = {
        "extra_weight_mean_dev": best_dev,
        "phq_w": best_w,
        "test_aucs": tests,
        "mean_test": float(np.mean(tests)),
        "init_mean_test": float(np.mean(base_tests)),
        "previous_mean_test": 0.7344465964006194,
    }
    (OUT_DIR / "finetune_clinical.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"finetune+phq w={best_w:.3f} dev={best_dev:.3f} test={summary['mean_test']:.3f} "
        f"folds={[round(v, 3) for v in tests]} init={summary['init_mean_test']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
