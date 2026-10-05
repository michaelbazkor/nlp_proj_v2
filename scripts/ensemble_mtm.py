"""Equal-weight ensemble of the five MTMs with the best mean development AUC.

The five configurations were fixed from the original grid's development
scores. Test labels are not used to choose members or weights.
The saved 0.726 model (config 123) is one of the five and is not overwritten.
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
from reselect_mtm import UngatedMTM, train_one
from ssr.config import load_config
from ssr.data.cohort import build_cohort
from ssr.fusion.project import collect_user_blocks
from ssr.models.stm import STM
from ssr.train.cv import PERSONALITY, PSYCHIATRIC, PSYCHOSOCIAL, _make_splits, _zscore_fit
from ssr.train.metrics import auc_roc

OUT_DIR = ROOT / "artifacts" / "pw_1.5_mtm_ensemble"
SHARED = ROOT / "artifacts" / "pw_1.5_mtm_shared"
CONFIGS = [
    (123, {"n_layers": 1, "n_neurons": 512, "activation": "tanh", "lr": 0.005, "epochs": 1000}),
    (227, {"n_layers": 2, "n_neurons": 64, "activation": "tanh", "lr": 0.05, "epochs": 5000}),
    (81, {"n_layers": 1, "n_neurons": 128, "activation": "tanh", "lr": 0.05, "epochs": 1000}),
    (399, {"n_layers": 3, "n_neurons": 64, "activation": "sigmoid", "lr": 0.005, "epochs": 1000}),
    (365, {"n_layers": 3, "n_neurons": 32, "activation": "tanh", "lr": 0.005, "epochs": 5000}),
]


def probabilities(model, X, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))["suicide_logit"].float().cpu().numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))


def load_mtm(path: Path, in_dim: int, params: dict, device: str) -> UngatedMTM:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = UngatedMTM(in_dim, int(params["n_layers"]), int(params["n_neurons"]), params["activation"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device)


def main() -> None:
    torch.set_num_threads(4)
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
    fold_rows = []
    for fold_i, tr, dv, te in splits:
        done_path = OUT_DIR / f"fold{fold_i}.json"
        if done_path.exists():
            rec = json.loads(done_path.read_text(encoding="utf-8"))
            fold_rows.append(rec)
            print(f"fold {fold_i} cached ensemble test={rec['ensemble_test_auc']:.3f}", flush=True)
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
        members = []
        dev_stack = []
        te_stack = []
        for cfg_idx, params in CONFIGS:
            device = "cuda:0" if params["n_neurons"] >= 512 and torch.cuda.is_available() else "cpu"
            ckpt = SHARED / f"mtm_high_fold{fold_i}.pt" if cfg_idx == 123 else OUT_DIR / f"cfg{cfg_idx}_fold{fold_i}.pt"
            if ckpt.exists():
                model = load_mtm(ckpt, Xtr.shape[1], params, device)
                print(f"fold {fold_i} cfg {cfg_idx} loaded", flush=True)
            else:
                seed = base.config_seed(cfg.seed, fold_i, "mtm", cfg_idx)
                print(f"fold {fold_i} cfg {cfg_idx} train on {device}", flush=True)
                dev_auc, model = train_one(params, base.to_tensors(fd, device), seed, device)
                torch.save(
                    {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "params": params, "fold": fold_i},
                    ckpt,
                )
                print(f"  trained dev={dev_auc:.3f}", flush=True)
            dev_p = probabilities(model, Xdv, device)
            te_p = probabilities(model, Xte, device)
            members.append(
                {
                    "cfg_idx": cfg_idx,
                    "params": params,
                    "dev_auc": float(auc_roc(ydv, dev_p)),
                    "test_auc": float(auc_roc(yte, te_p)),
                }
            )
            dev_stack.append(dev_p)
            te_stack.append(te_p)
            print(f"  cfg {cfg_idx} dev={members[-1]['dev_auc']:.3f} test={members[-1]['test_auc']:.3f}", flush=True)
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        dev_mean = np.mean(dev_stack, axis=0)
        te_mean = np.mean(te_stack, axis=0)
        rec = {
            "fold": fold_i,
            "members": members,
            "ensemble_dev_auc": float(auc_roc(ydv, dev_mean)),
            "ensemble_test_auc": float(auc_roc(yte, te_mean)),
        }
        done_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        fold_rows.append(rec)
        running = float(np.mean([r["ensemble_test_auc"] for r in fold_rows]))
        print(f"fold {fold_i} ENSEMBLE test={rec['ensemble_test_auc']:.3f} running mean={running:.3f}", flush=True)
    tests = [r["ensemble_test_auc"] for r in fold_rows]
    summary = {
        "rule": "equal-weight probability average of the five highest mean-dev MTM configs",
        "configs": [c[0] for c in CONFIGS],
        "test_aucs": tests,
        "mean_test_auc": float(np.mean(tests)),
        "shared_cfg123_mean_test_auc": 0.7261248087799812,
        "stm_mean_test_auc": 0.7162476854775705,
        "folds": fold_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"ENSEMBLE MEAN test={summary['mean_test_auc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
