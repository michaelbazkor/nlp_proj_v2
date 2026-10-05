"""Draw poster charts into this folder. Result checkpoints are read only."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
POSTER = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

INK = "#1e293b"
MUTED = "#64748b"
STM_C = "#1d4ed8"
MTM_C = "#0f766e"
PHQ_C = "#b45309"
PAPER_C = "#94a3b8"

STM_FOLDS = [0.7712087912087912, 0.6571945508727117, 0.7791613452532993, 0.7397214854111406, 0.6339522546419099]
MTM_FOLDS = [0.7286813186813187, 0.7130693912303107, 0.7674542358450405, 0.7317639257294429, 0.6896551724137931]
PHQ_FOLDS = [0.7351648351648351, 0.7120051085568327, 0.7690506598552576, 0.7416003536693192, 0.7144120247568524]


def cohens_d(auc: float) -> float:
    return float(np.sqrt(2.0) * norm.ppf(auc))


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK, labelsize=13)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK)


def save(fig, name: str):
    fig.savefig(POSTER / name, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bars():
    labels = ["Paper\nSTM", "Paper\nMTM", "This work\nSTM", "This work\nMTM + PHQ-9"]
    values = [0.629, 0.697, float(np.mean(STM_FOLDS)), float(np.mean(PHQ_FOLDS))]
    colors = [PAPER_C, PAPER_C, STM_C, PHQ_C]
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    x = np.arange(len(values))
    ax.bar(x, values, color=colors, width=0.72, zorder=2)
    ax.errorbar([1], [0.697], yerr=[[0.697 - 0.690], [0.707 - 0.697]], fmt="none", ecolor=INK, capsize=6, lw=1.4, zorder=3)
    d_paper, d_ours = 0.729, cohens_d(values[3])
    notes = {1: f"d = {d_paper:.3f}", 3: f"d = {d_ours:.3f}"}
    for i, v in enumerate(values):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=14, color=INK, fontweight="bold")
        if i in notes:
            ax.text(i, v + 0.042, notes[i], ha="center", va="bottom", fontsize=12, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(0.5, 0.90)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("High suicide risk", fontsize=20, pad=12, fontweight="bold")
    ax.axhline(0.5, color="#e2e8f0", lw=1, zorder=0)
    style_ax(ax)
    fig.tight_layout()
    save(fig, "05_high_risk_bars.svg")
    return d_ours


def folds():
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    x = np.arange(5)
    w = 0.24
    series = [
        (STM_FOLDS, STM_C, "STM", -w),
        (MTM_FOLDS, MTM_C, "MTM", 0),
        (PHQ_FOLDS, PHQ_C, "MTM + PHQ-9", w),
    ]
    for vals, color, label, shift in series:
        ax.bar(x + shift, vals, width=w, color=color, label=label, zorder=2)
        ax.axhline(float(np.mean(vals)), color=color, ls="--", lw=1.2, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in range(5)])
    ax.set_ylim(0.55, 0.88)
    ax.set_ylabel("Test AUC-ROC")
    ax.set_title("High suicide risk, by fold", fontsize=20, pad=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=12, loc="upper right")
    style_ax(ax)
    fig.tight_layout()
    save(fig, "06_folds.svg")


def roc():
    import torch

    import train_pw15_mtm as saved
    from reselect_mtm import UngatedMTM
    from ssr.config import load_config
    from ssr.data.cohort import build_cohort
    from ssr.fusion.project import collect_user_blocks
    from ssr.models.stm import STM
    from ssr.train.cv import PSYCHIATRIC, _make_splits, _zscore_fit

    cfg = load_config(ROOT / "eli_matrix.yaml")
    cohort = build_cohort(cfg, assert_paper=False)
    rep_roots = {m["name"]: (ROOT / "more stuff" / "reps" / m["name"]).resolve() for m in cfg.represent["models"]}
    user_ids = cohort["UserId"].tolist()
    print(f"users={len(user_ids)} high={int(cohort['y_high'].sum())}", flush=True)
    blocks = {uid: collect_user_blocks(rep_roots, uid) for uid in user_ids}
    splits = list(_make_splits(cohort["y_high"].to_numpy(), 5, cfg.seed, float(cfg.train["train_frac"]), float(cfg.train["dev_frac"])))
    idx = cohort.set_index("UserId")
    ys, stm_scores, phq_scores = [], [], []
    for fold_i, tr, dv, te in splits:
        def ids(ix):
            return [user_ids[i] for i in ix]

        fusion = saved.load_fusion(ROOT / "artifacts" / "pw_1.5" / f"fusion_fold{fold_i}.pt")

        def pack(ix):
            return fusion.transform_many([blocks[u] for u in ids(ix)]), idx.loc[ids(ix)]

        Xtr, frame_tr = pack(tr)
        Xte, frame_te = pack(te)
        mu = _zscore_fit(frame_tr[PSYCHIATRIC].to_numpy().astype(np.float32))
        ytr = (frame_tr[PSYCHIATRIC].to_numpy().astype(np.float32) - mu[0]) / mu[1]
        xmu, xsd = Xtr.mean(0), np.where(Xtr.std(0) < 1e-6, 1.0, Xtr.std(0))

        def norm(X):
            return (X - xmu) / xsd

        ridge = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(norm(Xtr), ytr[:, 0])
        raw_tr = ridge.predict(norm(Xtr)).astype(np.float32)
        raw_te = ridge.predict(norm(Xte)).astype(np.float32)
        psd = float(raw_tr.std()) or 1.0
        phq_te = (raw_te - float(raw_tr.mean())) / psd

        mtm = UngatedMTM(Xtr.shape[1], 1, 512, "tanh")
        mtm.load_state_dict(torch.load(ROOT / "artifacts" / "pw_1.5_mtm_shared" / f"mtm_high_fold{fold_i}.pt", map_location="cpu", weights_only=False)["state_dict"])
        stm_ck = torch.load(ROOT / "artifacts" / "pw_1.5" / f"stm_high_fold{fold_i}.pt", map_location="cpu", weights_only=False)
        p = stm_ck["params"]
        stm = STM(Xtr.shape[1], int(p["n_layers"]), int(p["n_neurons"]), p["activation"])
        stm.load_state_dict(stm_ck["state_dict"])
        mtm.eval()
        stm.eval()
        with torch.no_grad():
            xt = torch.tensor(Xte, dtype=torch.float32)
            mtm_logit = mtm(xt)["suicide_logit"].numpy()
            stm_logit = stm(xt)["suicide_logit"].numpy()
        y = frame_te["y_high"].to_numpy().astype(int)
        ys.append(y)
        stm_scores.append(1 / (1 + np.exp(-np.clip(stm_logit, -60, 60))))
        phq_scores.append(1 / (1 + np.exp(-np.clip(mtm_logit + 0.1 * phq_te, -60, 60))))
        print(f"fold {fold_i} n={len(y)}", flush=True)

    np.savez(
        POSTER / "roc_scores.npz",
        **{f"y{i}": ys[i] for i in range(5)},
        **{f"stm{i}": stm_scores[i] for i in range(5)},
        **{f"phq{i}": phq_scores[i] for i in range(5)},
    )
    fig, ax = plt.subplots(figsize=(7.4, 7.2))
    ax.plot([0, 1], [0, 1], color="#e2e8f0", lw=1.5, zorder=0)
    for i in range(5):
        for scores, color in ((stm_scores[i], STM_C), (phq_scores[i], PHQ_C)):
            fpr, tpr, _ = roc_curve(ys[i], scores)
            ax.plot(fpr, tpr, color=color, alpha=0.28, lw=1.2)
    ax.plot([], [], color=STM_C, lw=2.5, label=f"STM  {np.mean(STM_FOLDS):.3f}")
    ax.plot([], [], color=PHQ_C, lw=2.5, label=f"MTM + PHQ-9  {np.mean(PHQ_FOLDS):.3f}")
    # Bold mean curve: average TPR on a shared FPR grid.
    grid = np.linspace(0, 1, 201)
    for scores, color in ((stm_scores, STM_C), (phq_scores, PHQ_C)):
        tprs = []
        for i in range(5):
            fpr, tpr, _ = roc_curve(ys[i], scores[i])
            tprs.append(np.interp(grid, fpr, tpr))
        ax.plot(grid, np.mean(tprs, axis=0), color=color, lw=2.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("High suicide risk", fontsize=20, pad=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=13, loc="lower right")
    style_ax(ax)
    fig.tight_layout()
    save(fig, "07_roc.svg")


def captions(d_ours: float):
    text = f"""# Poster captions

Claim: LLM hidden states plus a multi-task network predict high suicide risk better than Ophir et al. (2020).

This is a research reproduction. It is not a screening tool. Scores use held-out posts. Clinical scales are used only while training.

The paper's general-risk multi-task result, 0.746, is a different label (about 36% positive) and is not plotted.

## 1. Pipeline

Facebook posts from one user pass through four language models. Each model contributes layer vectors at four positions. Attention pooling, fit on the training fold, compresses those vectors to 1024 features used by both classifiers.

## 2. Attention fusion

Inside one model and one layer, four position vectors receive scores, become weights that sum to one, and are added together. Every block is concatenated and mapped to 1024 dimensions.

## 3. Classifiers

The single-task network maps the features to a suicide score. The multi-task network predicts personality, psychosocial scores, and PHQ-9 with GAD before the suicide score. The reported multi-task score adds one tenth of a PHQ-9 value predicted from the training fold.

## 4. Cohort

1003 users. 132 are high risk, a suicide score of at least 3, about 13%. Five stratified folds, seed 42. Development chooses the model. The test fold is scored once.

## 5. High suicide risk

Paper single-task 0.629. Paper multi-task 0.697 (95% CI 0.690–0.707, Cohen's d 0.729). This single-task model 0.716. This multi-task model plus PHQ-9 0.734 (Cohen's d {d_ours:.3f}, same conversion as the paper).

## 6. Folds

Single-task folds 0.771, 0.657, 0.779, 0.740, 0.634 (mean 0.716). Multi-task alone 0.729, 0.713, 0.767, 0.732, 0.690 (mean 0.726). Multi-task plus PHQ-9 0.735, 0.712, 0.769, 0.742, 0.714 (mean 0.734). Dashed lines are those means. The gain is largest on folds 1 and 4. Folds 0 and 2 remain higher for the single-task model.

The plotted network is the one configuration with the best average development AUC. The PHQ-9 weight is the single shared value that raised that development score.

## 7. ROC

Each thin line is one test fold. Each thick line is the average of those five curves. The legend is the mean test AUC, 0.716 and 0.734. The gray diagonal is chance.
"""
    (POSTER / "captions.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    d = bars()
    print(f"d={d:.4f}", flush=True)
    folds()
    captions(d)
    roc()
    print("done", flush=True)
