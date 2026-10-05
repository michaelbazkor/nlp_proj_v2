"""Shared-weight improvements on the saved MTM, chosen by mean development AUC.

The saved 0.726 checkpoints and the 0.734 PHQ blend are left in place.
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
from ssr.models.stm import STM
from ssr.train.cv import PERSONALITY, PSYCHIATRIC, PSYCHOSOCIAL, _make_splits, _zscore_fit
from ssr.train.metrics import auc_roc

OUT_DIR = ROOT / "artifacts" / "pw_1.5_mtm_clinical"
MTM_DIR = ROOT / "artifacts" / "pw_1.5_mtm_shared"
STM_DIR = ROOT / "artifacts" / "pw_1.5"
AUX = PERSONALITY + PSYCHOSOCIAL + PSYCHIATRIC
W_GRID = np.linspace(-2, 2, 81)
A_GRID = np.linspace(0, 1, 21)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))


def ridge_column(Xtr, ytr, others):
    xmu = Xtr.mean(axis=0)
    xsd = Xtr.std(axis=0)
    xsd = np.where(xsd < 1e-6, 1.0, xsd)

    def norm(X):
        return (X - xmu) / xsd

    model = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(norm(Xtr), ytr)
    raw = [model.predict(norm(X)).astype(np.float32) for X in others]
    pmu, psd = float(raw[0].mean()), float(raw[0].std())
    if psd < 1e-6:
        psd = 1.0
    return [(v - pmu) / psd for v in raw]


def mean_auc(folds, score_fn) -> float:
    return float(np.mean([auc_roc(f["ydv"], score_fn(f)) for f in folds]))


def test_aucs(folds, score_fn) -> list[float]:
    return [float(auc_roc(f["yte"], score_fn(f, test=True))) for f in folds]


def best_1d(folds, key: str):
    best_w, best_dev = 0.0, -1.0
    for w in W_GRID:
        dev = mean_auc(folds, lambda f, w=w: f["mtm_dv"] + w * f[key + "_dv"])
        if dev > best_dev:
            best_dev, best_w = dev, float(w)

    def score(f, test=False, w=best_w):
        return (f["mtm_te"] if test else f["mtm_dv"]) + w * f[key + ("_te" if test else "_dv")]

    return {"w": best_w, "mean_dev": best_dev, "test_aucs": test_aucs(folds, score), "mean_test": float(np.mean(test_aucs(folds, score)))}


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
    folds = []
    for fold_i, tr, dv, te in splits:
        def ids(ix, fold_i=fold_i):
            return [user_ids[i] for i in ix]

        fusion = saved.load_fusion(STM_DIR / f"fusion_fold{fold_i}.pt")

        def pack(ix):
            return fusion.transform_many([blocks[u] for u in ids(ix)]), idx.loc[ids(ix)]

        Xtr, frame_tr = pack(tr)
        Xdv, frame_dv = pack(dv)
        Xte, frame_te = pack(te)
        ytr = frame_tr["y_high"].to_numpy().astype(np.float32)
        mu = _zscore_fit(frame_tr[AUX].to_numpy().astype(np.float32))
        z_tr = (frame_tr[AUX].to_numpy().astype(np.float32) - mu[0]) / mu[1]
        preds = {}
        for j, name in enumerate(AUX):
            packed = ridge_column(Xtr, z_tr[:, j], (Xtr, Xdv, Xte))
            preds[name] = {"tr": packed[0], "dv": packed[1], "te": packed[2]}
        corrs = {}
        for j, name in enumerate(AUX):
            if np.std(z_tr[:, j]) < 1e-8:
                corrs[name] = 0.0
            else:
                corrs[name] = float(np.corrcoef(ytr, z_tr[:, j])[0, 1])
        index = {split: sum(corrs[name] * preds[name][split] for name in AUX) for split in ("tr", "dv", "te")}
        imu, isd = float(index["tr"].mean()), float(index["tr"].std())
        if isd < 1e-6:
            isd = 1.0
        index = {k: (v - imu) / isd for k, v in index.items()}
        psych = {
            split: 0.5 * (preds["PHQ9"][split] + preds["GAD"][split]) for split in ("tr", "dv", "te")
        }
        pmu, psd = float(psych["tr"].mean()), float(psych["tr"].std())
        if psd < 1e-6:
            psd = 1.0
        psych = {k: (v - pmu) / psd for k, v in psych.items()}

        mtm_ck = torch.load(MTM_DIR / f"mtm_high_fold{fold_i}.pt", map_location="cpu", weights_only=False)
        mtm = UngatedMTM(Xtr.shape[1], 1, 512, "tanh")
        mtm.load_state_dict(mtm_ck["state_dict"])
        mtm.eval()
        stm_ck = torch.load(STM_DIR / f"stm_high_fold{fold_i}.pt", map_location="cpu", weights_only=False)
        p = stm_ck["params"]
        stm = STM(Xtr.shape[1], int(p["n_layers"]), int(p["n_neurons"]), p["activation"])
        stm.load_state_dict(stm_ck["state_dict"])
        stm.eval()

        def logit(model, X):
            with torch.no_grad():
                return model(torch.tensor(X, dtype=torch.float32))["suicide_logit"].numpy()

        folds.append(
            {
                "ydv": frame_dv["y_high"].to_numpy().astype(np.float32),
                "yte": frame_te["y_high"].to_numpy().astype(np.float32),
                "mtm_dv": logit(mtm, Xdv),
                "mtm_te": logit(mtm, Xte),
                "stm_dv": logit(stm, Xdv),
                "stm_te": logit(stm, Xte),
                "phq_dv": preds["PHQ9"]["dv"],
                "phq_te": preds["PHQ9"]["te"],
                "gad_dv": preds["GAD"]["dv"],
                "gad_te": preds["GAD"]["te"],
                "psych_dv": psych["dv"],
                "psych_te": psych["te"],
                "index_dv": index["dv"],
                "index_te": index["te"],
                "corrs": corrs,
            }
        )
        print(f"fold {fold_i} ready phq_corr={corrs['PHQ9']:.3f} gad_corr={corrs['GAD']:.3f}", flush=True)
        scale = float(Xtr.std())
        extra = {
            "tr": np.column_stack([preds["PHQ9"]["tr"], preds["GAD"]["tr"]]) * scale,
            "dv": np.column_stack([preds["PHQ9"]["dv"], preds["GAD"]["dv"]]) * scale,
            "te": np.column_stack([preds["PHQ9"]["te"], preds["GAD"]["te"]]) * scale,
        }
        all_mu = {
            key: _zscore_fit(frame_tr[cols].to_numpy().astype(np.float32))
            for key, cols in (("pers", PERSONALITY), ("psy", PSYCHOSOCIAL), ("psych", PSYCHIATRIC))
        }

        def z(frame, key, cols):
            m, s = all_mu[key]
            return (frame[cols].to_numpy().astype(np.float32) - m) / s

        fd = {
            "Xtr": np.column_stack([Xtr, extra["tr"]]),
            "ytr": ytr,
            "Xdv": np.column_stack([Xdv, extra["dv"]]),
            "ydv": frame_dv["y_high"].to_numpy().astype(np.float32),
            "pers": z(frame_tr, "pers", PERSONALITY),
            "psy": z(frame_tr, "psy", PSYCHOSOCIAL),
            "psych": z(frame_tr, "psych", PSYCHIATRIC),
        }
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        seed = base.config_seed(cfg.seed, fold_i, "mtm", 123)
        params = {"n_layers": 1, "n_neurons": 512, "activation": "tanh", "lr": 0.005, "epochs": 1000}
        dev_auc, model = train_one(params, base.to_tensors(fd, device), seed, device)
        model = model.cpu().eval()
        te_auc = float(
            auc_roc(
                frame_te["y_high"].to_numpy().astype(np.float32),
                sigmoid(logit(model, np.column_stack([Xte, extra["te"]]))),
            )
        )
        folds[-1]["scaled_dev"] = float(dev_auc)
        folds[-1]["scaled_test"] = te_auc
        print(f"  scaled-input dev={dev_auc:.3f} test={te_auc:.3f}", flush=True)
        del model

    results = {name: best_1d(folds, name) for name in ("phq", "gad", "psych", "index")}
    for name, rec in results.items():
        print(
            f"{name} w={rec['w']:.3f} dev={rec['mean_dev']:.3f} test={rec['mean_test']:.3f} "
            f"folds={[round(v, 3) for v in rec['test_aucs']]}",
            flush=True,
        )
    best_pair, best_pair_dev = (0.1, 0.0), -1.0
    for w_phq in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
        for w_gad in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
            dev = mean_auc(folds, lambda f, w_phq=w_phq, w_gad=w_gad: f["mtm_dv"] + w_phq * f["phq_dv"] + w_gad * f["gad_dv"])
            if dev > best_pair_dev:
                best_pair_dev, best_pair = dev, (w_phq, w_gad)

    def pair_score(f, test=False, pair=best_pair):
        tag = "_te" if test else "_dv"
        return f["mtm" + tag] + pair[0] * f["phq" + tag] + pair[1] * f["gad" + tag]

    pair_tests = test_aucs(folds, pair_score)
    print(
        f"phq+gad w={best_pair} dev={best_pair_dev:.3f} test={float(np.mean(pair_tests)):.3f} "
        f"folds={[round(v, 3) for v in pair_tests]}",
        flush=True,
    )
    scaled_dev = float(np.mean([f["scaled_dev"] for f in folds]))
    scaled_test = float(np.mean([f["scaled_test"] for f in folds]))
    print(
        f"scaled-input dev={scaled_dev:.3f} test={scaled_test:.3f} "
        f"folds={[round(f['scaled_test'], 3) for f in folds]}",
        flush=True,
    )
    clinical_name = max(results, key=lambda n: results[n]["mean_dev"])
    clinical = results[clinical_name]
    print(f"clinical choice={clinical_name}", flush=True)

    best_a, best_dev = 0.0, -1.0
    for a in A_GRID:
        dev = mean_auc(
            folds,
            lambda f, a=a: (1 - a) * sigmoid(f["mtm_dv"] + clinical["w"] * f[clinical_name + "_dv"])
            + a * sigmoid(f["stm_dv"]),
        )
        if dev > best_dev:
            best_dev, best_a = dev, float(a)

    def mix(f, test=False, a=best_a):
        tag = "_te" if test else "_dv"
        return (1 - a) * sigmoid(f["mtm" + tag] + clinical["w"] * f[clinical_name + tag]) + a * sigmoid(f["stm" + tag])

    mix_tests = test_aucs(folds, mix)
    summary = {
        "clinical_options": results,
        "clinical_choice": clinical_name,
        "stm_mix_a": best_a,
        "stm_mix_mean_dev": best_dev,
        "stm_mix_test_aucs": mix_tests,
        "stm_mix_mean_test": float(np.mean(mix_tests)),
        "phq_gad": {"w": best_pair, "mean_dev": best_pair_dev, "test_aucs": pair_tests, "mean_test": float(np.mean(pair_tests))},
        "scaled_input": {
            "mean_dev": scaled_dev,
            "mean_test": scaled_test,
            "test_aucs": [f["scaled_test"] for f in folds],
            "dev_aucs": [f["scaled_dev"] for f in folds],
        },
        "previous_phq_mean_test": 0.7344465964006194,
        "fold_train_corrs": [f["corrs"] for f in folds],
    }
    (OUT_DIR / "blend_v2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"stm mix a={best_a:.2f} dev={best_dev:.3f} test={summary['stm_mix_mean_test']:.3f} "
        f"folds={[round(v, 3) for v in mix_tests]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
