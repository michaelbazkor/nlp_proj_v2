"""Give the saved MTM an explicit PHQ-9/GAD score estimated from the posts.

The estimator is fit on the training fold only. A candidate replaces the
saved model only when its mean development AUC is higher.
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

import continue_pw15 as base
import train_pw15_mtm as saved
from reselect_mtm import UngatedMTM, train_one
from ssr.config import load_config
from ssr.data.cohort import build_cohort
from ssr.fusion.project import collect_user_blocks
from ssr.train.cv import PERSONALITY, PSYCHIATRIC, PSYCHOSOCIAL, _make_splits, _zscore_fit
from ssr.train.metrics import auc_roc

OUT_DIR = ROOT / "artifacts" / "pw_1.5_mtm_clinical"
SHARED = ROOT / "artifacts" / "pw_1.5_mtm_shared"
PARAMS = {"n_layers": 1, "n_neurons": 512, "activation": "tanh", "lr": 0.005, "epochs": 1000}
CFG_IDX = 123
BASELINE_DEV = [0.7284980744544287, 0.8207103123662816, 0.8299101412066753, 0.795250320924262, 0.7351305091998289]


def logits_of(model, X, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32, device=device)
        return model(t)["suicide_logit"].float().cpu().numpy()


def probs(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))


def ridge_predict(Xtr, ytr, Xdv, Xte):
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    def norm(X):
        return (X - mu) / sd
    model = RidgeCV(alphas=np.logspace(-2, 3, 12))
    model.fit(norm(Xtr), ytr)
    def pack(X):
        pred = model.predict(norm(X)).astype(np.float32)
        return pred
    tr, dv, te = pack(Xtr), pack(Xdv), pack(Xte)
    # Put the appended scores on a unit scale fit from the training predictions.
    pmu, psd = tr.mean(axis=0), tr.std(axis=0)
    psd = np.where(psd < 1e-6, 1.0, psd)
    return (tr - pmu) / psd, (dv - pmu) / psd, (te - pmu) / psd


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
        psych_mu = _zscore_fit(str_[PSYCHIATRIC].to_numpy().astype(np.float32))
        all_cols = PERSONALITY + PSYCHOSOCIAL + PSYCHIATRIC
        all_mu = _zscore_fit(str_[all_cols].to_numpy().astype(np.float32))

        def z(frame, mu, cols):
            return (frame[cols].to_numpy().astype(np.float32) - mu[0]) / mu[1]

        y_psych = z(str_, psych_mu, PSYCHIATRIC)
        y_all_aux = z(str_, all_mu, all_cols)
        psych_tr, psych_dv, psych_te = ridge_predict(Xtr, y_psych, Xdv, Xte)
        all_tr, all_dv, all_te = ridge_predict(Xtr, y_all_aux, Xdv, Xte)
        phq_r = float(np.corrcoef(z(sdv, psych_mu, PSYCHIATRIC)[:, 0], psych_dv[:, 0])[0, 1])
        print(f"fold {fold_i} feature std={float(Xtr.std()):.3f} dev phq_r={phq_r:.3f}", flush=True)

        ck = torch.load(SHARED / f"mtm_high_fold{fold_i}.pt", map_location="cpu", weights_only=False)
        base_model = UngatedMTM(Xtr.shape[1], 1, 512, "tanh").to(device)
        base_model.load_state_dict(ck["state_dict"])
        base_logit_dv = logits_of(base_model, Xdv, device)
        base_logit_te = logits_of(base_model, Xte, device)
        best_w, best_dev = 0.0, float(auc_roc(ydv, probs(base_logit_dv)))
        for w in np.linspace(-2, 2, 41):
            score = auc_roc(ydv, probs(base_logit_dv + w * psych_dv[:, 0]))
            if np.isfinite(score) and score > best_dev:
                best_dev, best_w = float(score), float(w)
        blend_te = float(auc_roc(yte, probs(base_logit_te + best_w * psych_te[:, 0])))
        print(f"  blend w={best_w:.2f} dev={best_dev:.3f} test={blend_te:.3f}", flush=True)
        del base_model

        trained = {}
        for name, extra_tr, extra_dv, extra_te in (
            ("psych", psych_tr, psych_dv, psych_te),
            ("all_aux", all_tr, all_dv, all_te),
        ):
            fd = {
                "Xtr": np.concatenate([Xtr, extra_tr], axis=1),
                "ytr": str_["y_high"].to_numpy().astype(np.float32),
                "Xdv": np.concatenate([Xdv, extra_dv], axis=1),
                "ydv": ydv,
                "pers": z(str_, _zscore_fit(str_[PERSONALITY].to_numpy().astype(np.float32)), PERSONALITY),
                "psy": z(str_, _zscore_fit(str_[PSYCHOSOCIAL].to_numpy().astype(np.float32)), PSYCHOSOCIAL),
                "psych": y_psych,
            }
            seed = base.config_seed(cfg.seed, fold_i, "mtm", CFG_IDX)
            dev_auc, model = train_one(PARAMS, base.to_tensors(fd, device), seed, device)
            te_auc = float(auc_roc(yte, probs(logits_of(model, np.concatenate([Xte, extra_te], axis=1), device))))
            trained[name] = {"dev_auc": float(dev_auc), "test_auc": te_auc}
            print(f"  {name} dev={dev_auc:.3f} test={te_auc:.3f}", flush=True)
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        rec = {
            "fold": fold_i,
            "baseline_dev": BASELINE_DEV[fold_i],
            "dev_phq_r": phq_r,
            "blend_w": best_w,
            "blend_dev": best_dev,
            "blend_test": blend_te,
            **{f"{k}_{metric}": trained[k][metric] for k in trained for metric in ("dev_auc", "test_auc")},
        }
        done_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        rows.append(rec)
    summary = {
        "baseline_mean_dev": float(np.mean(BASELINE_DEV)),
        "baseline_mean_test": 0.7261248087799812,
        "blend_mean_dev": float(np.mean([r["blend_dev"] for r in rows])),
        "blend_mean_test": float(np.mean([r["blend_test"] for r in rows])),
        "psych_mean_dev": float(np.mean([r["psych_dev_auc"] for r in rows])),
        "psych_mean_test": float(np.mean([r["psych_test_auc"] for r in rows])),
        "all_aux_mean_dev": float(np.mean([r["all_aux_dev_auc"] for r in rows])),
        "all_aux_mean_test": float(np.mean([r["all_aux_test_auc"] for r in rows])),
        "folds": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"baseline dev={summary['baseline_mean_dev']:.3f} test={summary['baseline_mean_test']:.3f} | "
        f"blend dev={summary['blend_mean_dev']:.3f} test={summary['blend_mean_test']:.3f} | "
        f"psych dev={summary['psych_mean_dev']:.3f} test={summary['psych_mean_test']:.3f} | "
        f"all dev={summary['all_aux_mean_dev']:.3f} test={summary['all_aux_mean_test']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
