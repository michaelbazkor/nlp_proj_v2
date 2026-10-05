"""Refit the one MTM configuration with the highest mean development AUC.

Chosen from the original ungated grid before looking at these test scores:
1 layer, 512 units, tanh, lr 0.005, 1000 epochs (config 123).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import continue_pw15 as base
import train_pw15_mtm as saved
from reselect_mtm import eval_split, train_one
from ssr.config import load_config
from ssr.data.cohort import build_cohort
from ssr.fusion.project import collect_user_blocks
from ssr.train.cv import PERSONALITY, PSYCHIATRIC, PSYCHOSOCIAL, _make_splits, _zscore_fit

OUT_DIR = ROOT / "artifacts" / "pw_1.5_mtm_shared"
PARAMS = {"n_layers": 1, "n_neurons": 512, "activation": "tanh", "lr": 0.005, "epochs": 1000}
CFG_IDX = 123


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
    chosen = []
    for fold_i, tr, dv, te in splits:
        done_path = OUT_DIR / f"fold{fold_i}.json"
        if done_path.exists():
            rec = json.loads(done_path.read_text(encoding="utf-8"))
            chosen.append(rec)
            print(f"fold {fold_i} cached test={rec['test_auc']:.3f}", flush=True)
            continue

        def ids(ix):
            return [user_ids[i] for i in ix]

        fusion = saved.load_fusion(ROOT / "artifacts" / "pw_1.5" / f"fusion_fold{fold_i}.pt")

        def pack(ix):
            return fusion.transform_many([blocks[u] for u in ids(ix)]), idx.loc[ids(ix)]

        Xtr, str_ = pack(tr)
        Xdv, sdv = pack(dv)
        Xte, ste = pack(te)
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
            "ydv": sdv["y_high"].to_numpy().astype(np.float32),
            "pers": z(str_, "pers", PERSONALITY),
            "psy": z(str_, "psy", PSYCHOSOCIAL),
            "psych": z(str_, "psych", PSYCHIATRIC),
        }
        seed = base.config_seed(cfg.seed, fold_i, "mtm", CFG_IDX)
        print(f"fold {fold_i} cfg {CFG_IDX} on {device}", flush=True)
        dev_auc, model = train_one(PARAMS, base.to_tensors(fd, device), seed, device)
        te_auc, psych = eval_split(model, Xte, ste["y_high"].to_numpy().astype(np.float32), device)
        torch.save(
            {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "params": PARAMS, "fold": fold_i},
            OUT_DIR / f"mtm_high_fold{fold_i}.pt",
        )
        true = ste[PSYCHIATRIC].to_numpy().astype(np.float32)
        pred = psych * mu["psych"][1] + mu["psych"][0]
        phq = float(np.corrcoef(true[:, 0], pred[:, 0])[0, 1])
        rec = {"fold": fold_i, "dev_auc": dev_auc, "test_auc": float(te_auc), "pearson_phq9": phq, "params": PARAMS}
        done_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        chosen.append(rec)
        print(
            f"  dev={dev_auc:.3f} test={te_auc:.3f} phq_r={phq:.3f} "
            f"running mean={float(np.mean([c['test_auc'] for c in chosen])):.3f}",
            flush=True,
        )
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    summary = {
        "cfg_idx": CFG_IDX,
        "params": PARAMS,
        "selection": "highest mean search dev AUC across the five folds",
        "test_aucs": [c["test_auc"] for c in chosen],
        "mean_test_auc": float(np.mean([c["test_auc"] for c in chosen])),
        "stm_mean_test_auc": 0.716,
        "folds": chosen,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"MEAN test={summary['mean_test_auc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
