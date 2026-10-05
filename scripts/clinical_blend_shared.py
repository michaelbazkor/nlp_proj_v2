"""One PHQ weight for every fold, chosen by mean development AUC."""
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
from ssr.train.cv import PSYCHIATRIC, _make_splits, _zscore_fit
from ssr.train.metrics import auc_roc

OUT_DIR = ROOT / "artifacts" / "pw_1.5_mtm_clinical"
SHARED = ROOT / "artifacts" / "pw_1.5_mtm_shared"


def probs(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))


def main() -> None:
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
    folds = []
    for fold_i, tr, dv, te in splits:
        def ids(ix):
            return [user_ids[i] for i in ix]

        fusion = saved.load_fusion(ROOT / "artifacts" / "pw_1.5" / f"fusion_fold{fold_i}.pt")

        def pack(ix):
            return fusion.transform_many([blocks[u] for u in ids(ix)]), idx.loc[ids(ix)]

        Xtr, str_ = pack(tr)
        Xdv, sdv = pack(dv)
        Xte, ste = pack(te)
        mu = _zscore_fit(str_[PSYCHIATRIC].to_numpy().astype(np.float32))
        ytr = (str_[PSYCHIATRIC].to_numpy().astype(np.float32) - mu[0]) / mu[1]
        xmu, xsd = Xtr.mean(0), np.where(Xtr.std(0) < 1e-6, 1.0, Xtr.std(0))
        def norm(X):
            return (X - xmu) / xsd
        ridge = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(norm(Xtr), ytr[:, 0])
        raw = [ridge.predict(norm(X)).astype(np.float32) for X in (Xtr, Xdv, Xte)]
        pmu, psd = float(raw[0].mean()), float(raw[0].std() or 1.0)
        phq = [(v - pmu) / psd for v in raw]
        ck = torch.load(SHARED / f"mtm_high_fold{fold_i}.pt", map_location="cpu", weights_only=False)
        model = UngatedMTM(Xtr.shape[1], 1, 512, "tanh").to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        with torch.no_grad():
            def logit(X):
                return model(torch.tensor(X, dtype=torch.float32, device=device))["suicide_logit"].float().cpu().numpy()
            folds.append({
                "ydv": sdv["y_high"].to_numpy().astype(np.float32),
                "yte": ste["y_high"].to_numpy().astype(np.float32),
                "ldv": logit(Xdv),
                "lte": logit(Xte),
                "pdv": phq[1],
                "pte": phq[2],
            })
        print(f"fold {fold_i} ready", flush=True)
    best_w, best_dev = 0.0, -1.0
    for w in np.linspace(-2, 2, 81):
        devs = [auc_roc(f["ydv"], probs(f["ldv"] + w * f["pdv"])) for f in folds]
        mean_dev = float(np.mean(devs))
        if mean_dev > best_dev:
            best_dev, best_w = mean_dev, float(w)
    tests = [float(auc_roc(f["yte"], probs(f["lte"] + best_w * f["pte"]))) for f in folds]
    base_tests = [float(auc_roc(f["yte"], probs(f["lte"]))) for f in folds]
    summary = {
        "rule": "one PHQ weight for all folds, maximizing mean development AUC",
        "w": best_w,
        "mean_dev": best_dev,
        "test_aucs": tests,
        "mean_test_auc": float(np.mean(tests)),
        "baseline_test_aucs": base_tests,
        "baseline_mean_test": float(np.mean(base_tests)),
    }
    (OUT_DIR / "shared_weight.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"w={best_w:.3f} dev={best_dev:.3f} test={summary['mean_test_auc']:.3f} baseline={summary['baseline_mean_test']:.3f}", flush=True)
    print("tests", [round(v, 3) for v in tests], flush=True)


if __name__ == "__main__":
    main()
