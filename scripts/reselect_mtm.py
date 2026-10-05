"""Pick an MTM from near-tied development scores, without a new grid search.

For each fold, refit the three configurations with the highest search dev AUC
from the original (ungated) pw_1.5 MTM run. Keep the one with the highest
refit dev AUC. Test AUC is recorded after that choice.
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
from ssr.config import load_config
from ssr.data.cohort import build_cohort
from ssr.fusion.project import collect_user_blocks
from ssr.models.stm import MTM
from ssr.train.cv import PERSONALITY, PSYCHIATRIC, PSYCHOSOCIAL, _make_splits, _zscore_fit
from ssr.train.metrics import auc_roc, mtm_loss

GRID_DIR = Path.home() / "AppData/Local/nlp_proj_v2/pw15_mtm"
OUT_DIR = ROOT / "artifacts" / "pw_1.5_mtm_reselect"
LAYERS = [1, 2, 3]
NEURONS = [16, 32, 64, 128, 256, 512, 1024]
ACTS = ["tanh", "sigmoid"]
LRS = [0.001, 0.005, 0.01, 0.05]
EPOCHS = [1000, 2500, 5000]


class UngatedMTM(MTM):
    """The MTM that produced the saved grid, before the tanh cascade gate."""

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


def params_of(cfg_idx: int) -> dict:
    i = cfg_idx
    epoch = EPOCHS[i % 3]
    i //= 3
    lr = LRS[i % 4]
    i //= 4
    act = ACTS[i % 2]
    i //= 2
    neurons = NEURONS[i % 7]
    i //= 7
    return {
        "n_layers": LAYERS[i],
        "n_neurons": neurons,
        "activation": act,
        "lr": lr,
        "epochs": epoch,
    }


def load_top(fold: int, k: int = 3) -> list[dict]:
    rows, seen = [], set()
    path = GRID_DIR / f"fold{fold}" / "results.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["cfg_idx"] in seen or rec.get("dev_auc") is None:
            continue
        seen.add(rec["cfg_idx"])
        rows.append(rec)
    rows.sort(key=lambda r: -r["dev_auc"])
    return rows[:k]


def train_one(params, tensors, seed, device):
    torch.manual_seed(seed)
    model = UngatedMTM(
        tensors["Xtr"].shape[1], int(params["n_layers"]), int(params["n_neurons"]), params["activation"]
    ).to(device)
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
            loss, _ = mtm_loss(out, ytr[ii], tensors["pers"][ii], tensors["psy"][ii], tensors["psych"][ii])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(tensors["Xdv"])["suicide_logit"].float().cpu().numpy()
        score = auc_roc(tensors["ydv_np"], 1 / (1 + np.exp(-np.clip(logits, -60, 60))))
        if np.isfinite(score) and score > best:
            best, bad = float(score), 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 200:
                break
    model.load_state_dict(state)
    return best, model


def eval_split(model, X, y, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))["suicide_logit"].float().cpu().numpy()
        psych = model(torch.tensor(X, dtype=torch.float32, device=device))["psychiatric"].float().cpu().numpy()
    auc = auc_roc(y, 1 / (1 + np.exp(-np.clip(logits, -60, 60))))
    return auc, psych


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
    chosen = []
    for fold_i, tr, dv, te in splits:
        done_path = OUT_DIR / f"fold{fold_i}.json"
        if done_path.exists():
            rec = json.loads(done_path.read_text(encoding="utf-8"))
            chosen.append(rec["selected"])
            print(f"fold {fold_i} already selected test={rec['selected']['test_auc']:.3f}", flush=True)
            continue

        def ids(ix):
            return [user_ids[i] for i in ix]

        fusion = saved.load_fusion(ROOT / "artifacts" / "pw_1.5" / f"fusion_fold{fold_i}.pt")

        def pack(ix):
            return fusion.transform_many([blocks[u] for u in ids(ix)]), idx.loc[ids(ix)]

        Xtr, str_ = pack(tr)
        Xdv, sdv = pack(dv)
        Xte, ste = pack(te)
        ytr = str_["y_high"].to_numpy().astype(np.float32)
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
            "ytr": ytr,
            "Xdv": Xdv,
            "ydv": ydv,
            "pers": z(str_, "pers", PERSONALITY),
            "psy": z(str_, "psy", PSYCHOSOCIAL),
            "psych": z(str_, "psych", PSYCHIATRIC),
        }
        results = []
        for rank, rec in enumerate(load_top(fold_i, 3)):
            params = params_of(int(rec["cfg_idx"]))
            device = rec.get("device") or "cpu"
            if str(device).startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            seed = base.config_seed(cfg.seed, fold_i, "mtm", int(rec["cfg_idx"]))
            print(f"fold {fold_i} rank {rank} cfg {rec['cfg_idx']} {params} on {device}", flush=True)
            tensors = base.to_tensors(fd, device)
            dev_auc, model = train_one(params, tensors, seed, device)
            te_auc, psych = eval_split(model, Xte, yte, device)
            true = ste[PSYCHIATRIC].to_numpy().astype(np.float32)
            pred = psych * mu["psych"][1] + mu["psych"][0]
            phq = float(np.corrcoef(true[:, 0], pred[:, 0])[0, 1])
            row = {
                "cfg_idx": int(rec["cfg_idx"]),
                "search_dev_auc": float(rec["dev_auc"]),
                "refit_dev_auc": float(dev_auc),
                "test_auc": float(te_auc),
                "pearson_phq9": phq,
                "params": params,
                "device": device,
            }
            results.append(row)
            print(
                f"  search_dev={row['search_dev_auc']:.3f} refit_dev={dev_auc:.3f} "
                f"test={te_auc:.3f} phq_r={phq:.3f}",
                flush=True,
            )
            del tensors, model
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        selected = max(results, key=lambda r: (r["refit_dev_auc"], r["search_dev_auc"]))
        payload = {"fold": fold_i, "candidates": results, "selected": selected}
        done_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        chosen.append(selected)
        vals = [c["test_auc"] for c in chosen]
        print(
            f"fold {fold_i} SELECTED test={selected['test_auc']:.3f} "
            f"running mean={float(np.mean(vals)):.3f}",
            flush=True,
        )
    summary = {
        "rule": "refit top 3 by search dev AUC; keep highest refit dev AUC",
        "test_aucs": [c["test_auc"] for c in chosen],
        "mean_test_auc": float(np.mean([c["test_auc"] for c in chosen])),
        "stm_mean_test_auc": 0.716,
        "selected": chosen,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"MEAN test={summary['mean_test_auc']:.3f} folds={ [round(v, 3) for v in summary['test_aucs']] }", flush=True)


if __name__ == "__main__":
    main()
