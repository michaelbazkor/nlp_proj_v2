"""Poster panels that need the cohort and the saved models. Writes only under poster/.

python poster/make_panels.py          score the held-out users, then draw
python poster/make_panels.py redraw   draw again from panel_stats.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
POSTER = Path(__file__).resolve().parent
A0 = POSTER / "a0"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(POSTER))

from svgkit import (  # noqa: E402
    EDGE, INK, LINE, MTM_C, MUTED, PAPER_C, PERS_C, PHQ_C, PSY_C, PSYCH_C, RIGHT_C, STM_C, WRONG_C, Svg, text_width,
    wrap,
)

plt.rcParams.update(
    {
        "font.family": "Arial",
        "svg.fonttype": "none",
        "axes.edgecolor": EDGE,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.titlecolor": INK,
    }
)

AUX_TARGETS = ["PHQ9", "GAD", "Brooding", "Worry", "SWL", "Lonely", "BFI_O", "BFI_C", "BFI_E", "BFI_A", "BFI_N"]

SCALE_NAMES = {
    "PHQ9": "PHQ-9",
    "GAD": "GAD",
    "Brooding": "Brooding",
    "Worry": "Worry",
    "SWL": "Life satisfaction",
    "Lonely": "Loneliness",
    "BFI_O": "Openness",
    "BFI_C": "Conscientiousness",
    "BFI_E": "Extraversion",
    "BFI_A": "Agreeableness",
    "BFI_N": "Neuroticism",
    "suicide": "Suicide score",
}

STAGES = [
    ("Personality", PERS_C, ["BFI_O", "BFI_C", "BFI_E", "BFI_A", "BFI_N"]),
    ("Psychosocial", PSY_C, ["Brooding", "Worry", "Lonely", "SWL"]),
    ("Psychiatric", PSYCH_C, ["PHQ9", "GAD"]),
]

# Verbatim excerpts from each user's own posts, picked by hand as the passages most related to
# distress. Spelling is the user's. Names are replaced and the rest of each post is cut.
EXAMPLE_TEXT = {
    "Correct, high risk": [
        "i think a Lot of ppl who see me constantly talking abt suicide think im joking? but honestly the "
        "reality is im not ever joking and i am willing and ready to die at a moments notice",
        "Mental illness is a series of questions like when have I eaten last? Why am I crying in the "
        "bathroom at 5:16 in the pm? Was I always this sad for no reason???",
    ],
    "Missed high risk": [
        "I saw my dad have a stroke that night and I would never wish that upon my worst enemy.",
        "Never let yourself get so far down a hole to where you feel you can't see the top.",
        "Brooding, loving, and hopeless.... Lol hmmmm",
    ],
    "False alarm": [
        "My kids are growing up around me and I still haven't given them the life I wanted to. "
        "I can't help but feel like I'm failing. Too little, too late.",
        "I truly feel like a caged bird, unable to express my potpourri of emotions and thoughts using a "
        "creative medium.",
    ],
    "Correct, no risk": [
        "Someone just asked me how my dad was, and I said \"Dead, so not great\". I didn't do that right, did I?",
        "Sometimes I wonder if I missed the day in kindergarten when all the other kids learned how to deal "
        "with their feelings.",
    ],
}

EXAMPLE_NOTE = {
    "Correct, high risk": "Open, repeated talk of suicide across 933 posts.",
    "Missed high risk": "Rare distress, buried among 189 mostly cheerful posts.",
    "False alarm": "Self-critical, low-mood posts, but no reported ideation.",
    "Correct, no risk": "Death and feelings come up only as jokes.",
}


def sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))


def best_f1_threshold(y, scores):
    from sklearn.metrics import f1_score

    best_t, best_f = 0.5, -1.0
    for t in np.quantile(scores, np.linspace(0.05, 0.95, 37)):
        f = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = float(f), float(t)
    return best_t


def ridge_phq(Xtr, ytr, Xdv, Xte):
    from sklearn.linear_model import RidgeCV

    xmu, xsd = Xtr.mean(0), np.where(Xtr.std(0) < 1e-6, 1.0, Xtr.std(0))

    def norm(X):
        return (X - xmu) / xsd

    model = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(norm(Xtr), ytr)
    raw = [model.predict(norm(X)).astype(np.float32) for X in (Xtr, Xdv, Xte)]
    sd = float(raw[0].std()) or 1.0
    mu = float(raw[0].mean())
    return [(v - mu) / sd for v in raw]


def score():
    import torch

    import train_pw15_mtm as saved
    from reselect_mtm import UngatedMTM
    from ssr.config import load_config
    from ssr.data.cohort import build_cohort
    from ssr.fusion.project import collect_user_blocks
    from ssr.train.cv import PERSONALITY, PSYCHIATRIC, PSYCHOSOCIAL, _make_splits, _zscore_fit

    cfg = load_config(ROOT / "eli_matrix.yaml")
    cohort = build_cohort(cfg, assert_paper=False)
    y_high = cohort["y_high"].to_numpy()
    suicide = cohort["suicide"].to_numpy().astype(float)
    corr_rows = []
    for col in AUX_TARGETS:
        x = cohort[col].to_numpy().astype(float)
        corr_rows.append(
            {
                "scale": col,
                "r_suicide": float(np.corrcoef(x, suicide)[0, 1]),
                "r_high": float(np.corrcoef(x, y_high)[0, 1]),
            }
        )

    rep_roots = {m["name"]: (ROOT / "more stuff" / "reps" / m["name"]).resolve() for m in cfg.represent["models"]}
    user_ids = cohort["UserId"].tolist()
    print("loading representations", flush=True)
    blocks = {uid: collect_user_blocks(rep_roots, uid) for uid in user_ids}
    splits = list(_make_splits(y_high, 5, cfg.seed, float(cfg.train["train_frac"]), float(cfg.train["dev_frac"])))
    idx = cohort.set_index("UserId")

    records = []
    aux_fold = []
    for fold_i, tr, dv, te in splits:
        def ids(ix):
            return [user_ids[i] for i in ix]

        fusion = saved.load_fusion(ROOT / "artifacts" / "pw_1.5" / f"fusion_fold{fold_i}.pt")

        def pack(ix):
            return fusion.transform_many([blocks[u] for u in ids(ix)]), idx.loc[ids(ix)]

        Xtr, ftr = pack(tr)
        Xdv, fdv = pack(dv)
        Xte, fte = pack(te)
        mu = _zscore_fit(ftr[PSYCHIATRIC].to_numpy().astype(np.float32))
        y_psych = (ftr[PSYCHIATRIC].to_numpy().astype(np.float32) - mu[0]) / mu[1]
        _, phq_dv, phq_te = ridge_phq(Xtr, y_psych[:, 0], Xdv, Xte)
        groups = {"pers": PERSONALITY, "psy": PSYCHOSOCIAL, "psych": PSYCHIATRIC}
        zmu = {k: _zscore_fit(ftr[cols].to_numpy().astype(np.float32)) for k, cols in groups.items()}

        model = UngatedMTM(Xtr.shape[1], 1, 512, "tanh")
        ck = torch.load(ROOT / f"artifacts/pw_1.5_mtm_shared/mtm_high_fold{fold_i}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        model.eval()

        def run(X):
            with torch.no_grad():
                return model(torch.tensor(X, dtype=torch.float32))

        out_dv, out_te = run(Xdv), run(Xte)
        prob_dv = sigmoid(out_dv["suicide_logit"].numpy() + 0.1 * phq_dv)
        prob_te = sigmoid(out_te["suicide_logit"].numpy() + 0.1 * phq_te)
        thr = best_f1_threshold(fdv["y_high"].to_numpy().astype(int), prob_dv)
        yte = fte["y_high"].to_numpy().astype(int)
        pred = (prob_te >= thr).astype(int)
        for i, uid in enumerate(ids(te)):
            records.append(
                {
                    "fold": fold_i,
                    "uid": uid,
                    "suicide": int(fte.iloc[i]["suicide"]),
                    "y": int(yte[i]),
                    "prob": float(prob_te[i]),
                    "pred": int(pred[i]),
                    "threshold": thr,
                }
            )
        head = {"pers": "personality", "psy": "psychosocial", "psych": "psychiatric"}
        fold_r = {"fold": fold_i}
        for key, cols in groups.items():
            pred_z = out_te[head[key]].numpy()
            true_z = (fte[cols].to_numpy().astype(np.float32) - zmu[key][0]) / zmu[key][1]
            for j, col in enumerate(cols):
                a, b = true_z[:, j], pred_z[:, j]
                fold_r[col] = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-8 and np.std(b) > 1e-8 else float("nan")
        aux_fold.append(fold_r)
        print(f"fold {fold_i} thr={thr:.3f}", flush=True)

    rec = pd.DataFrame(records)
    rec["correct"] = (rec["pred"] == rec["y"]).astype(int)
    return rec, aux_fold, corr_rows


def pick_examples(rec: pd.DataFrame) -> dict[str, pd.Series]:
    """The most extreme user in each confusion cell."""
    frames = {
        "Correct, high risk": rec[(rec.suicide >= 5) & (rec.pred == 1)].sort_values("prob", ascending=False),
        "Missed high risk": rec[(rec.suicide >= 5) & (rec.pred == 0)].sort_values("prob"),
        "False alarm": rec[(rec.suicide == 0) & (rec.pred == 1)].sort_values("prob", ascending=False),
        "Correct, no risk": rec[(rec.suicide == 0) & (rec.pred == 0)].sort_values("prob"),
    }
    return {label: frame.iloc[0] for label, frame in frames.items()}


def main() -> None:
    rec, aux_fold, corr_rows = score()
    counts = []
    for score in range(7):
        sub = rec[rec["suicide"] == score]
        counts.append(
            {
                "suicide": score,
                "correct": int(sub["correct"].sum()),
                "incorrect": int((1 - sub["correct"]).sum()),
                "n": int(len(sub)),
            }
        )
    aux = pd.DataFrame(aux_fold)
    aux_mean = {col: float(aux[col].mean()) for col in AUX_TARGETS}
    aux_std = {col: float(aux[col].std(ddof=0)) for col in AUX_TARGETS}

    # No user ids are written.
    examples = []
    for label, row in pick_examples(rec).items():
        examples.append(
            {
                "label": label,
                "suicide": int(row.suicide),
                "prob": float(row.prob),
                "pred": int(row.pred),
                "threshold": float(row.threshold),
            }
        )

    stats = {
        "counts": counts,
        "correlations": corr_rows,
        "aux_mean_r": aux_mean,
        "aux_std_r": aux_std,
        "aux_fold_r": aux_fold,
        "examples": examples,
        "n_test": int(len(rec)),
        "accuracy": float(rec.correct.mean()),
        "thresholds": [float(t) for t in rec.groupby("fold")["threshold"].first()],
    }
    (POSTER / "panel_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    draw_panels(stats)
    print("accuracy", round(stats["accuracy"], 3), flush=True)
    print("aux", {k: round(v, 3) for k, v in aux_mean.items()}, flush=True)


def headline(fig, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.985, title, fontsize=21, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(0.012, 0.928, subtitle, fontsize=12.5, color=MUTED, va="top", ha="left")


def clean(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, labelsize=12.5)


def save(fig, name: str, a0: bool = False) -> None:
    out = A0 / name if a0 else POSTER / name
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight", pad_inches=0.12 if a0 else 0.25, facecolor="white")
    plt.close(fig)


def confusion(counts):
    tp = sum(c["correct"] for c in counts if c["suicide"] >= 3)
    pos = sum(c["n"] for c in counts if c["suicide"] >= 3)
    tn = sum(c["correct"] for c in counts if c["suicide"] < 3)
    neg = sum(c["n"] for c in counts if c["suicide"] < 3)
    return {"tp": tp, "fn": pos - tp, "tn": tn, "fp": neg - tn, "pos": pos, "neg": neg}


def draw_errors(stats: dict, a0: bool = False) -> None:
    counts = stats["counts"]
    cm = confusion(counts)
    n = np.array([c["n"] for c in counts])
    wrong = np.array([c["incorrect"] for c in counts])
    right = np.array([c["correct"] for c in counts])
    pw, pr = wrong / n * 100, right / n * 100
    xs = np.arange(7)

    fig, ax = plt.subplots(figsize=(9.4, 6.3) if a0 else (11.6, 6.9))
    if a0:
        fig.subplots_adjust(top=0.91, bottom=0.14, left=0.09, right=0.98)
    else:
        fig.subplots_adjust(top=0.86, bottom=0.14, left=0.08, right=0.98)
        headline(
            fig,
            "Correct and incorrect calls by true suicide score",
            "1003 held-out users, MTM + PHQ-9. Each fold's threshold maximizes development F1. "
            "Numbers in bars are users.",
        )
    ax.axvspan(-0.5, 2.5, color="#f8fafc", zorder=0)
    ax.axvspan(2.5, 6.5, color="#fff7ed", zorder=0)
    ax.bar(xs, pw, width=0.66, color=WRONG_C, zorder=2, label="Incorrect")
    ax.bar(xs, pr, bottom=pw, width=0.66, color=RIGHT_C, zorder=2, label="Correct")
    for i in xs:
        if pw[i] >= 8:
            ax.text(i, pw[i] / 2, f"{wrong[i]}", ha="center", va="center", color="white", fontsize=13, fontweight="bold")
        else:
            ax.text(i, pw[i] + 2.5, f"{wrong[i]}", ha="center", va="bottom", color=WRONG_C, fontsize=13, fontweight="bold")
        ax.text(i, pw[i] + pr[i] / 2 + (3 if pw[i] < 8 else 0), f"{right[i]}", ha="center", va="center",
                color="white", fontsize=13, fontweight="bold")
        ax.text(i, 102.5, f"{pw[i]:.0f}% wrong", ha="center", va="bottom", fontsize=12, fontweight="bold",
                color=WRONG_C if i >= 3 else MUTED)
    ax.text(1, 121, "Not high risk (scores 0–2)", ha="center", fontsize=14, fontweight="bold", color=INK)
    ax.text(1, 114.5, f"{cm['tn']} of {cm['neg']} correctly called not high risk", ha="center", fontsize=11.5, color=MUTED)
    ax.text(4.5, 121, "High risk (scores 3–6)", ha="center", fontsize=14, fontweight="bold", color="#9a3412")
    ax.text(4.5, 114.5, f"{cm['tp']} of {cm['pos']} correctly called high risk", ha="center", fontsize=11.5, color=MUTED)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}\nn = {n[s]}" for s in range(7)])
    for lab, s in zip(ax.get_xticklabels(), range(7)):
        lab.set_fontweight("bold" if s >= 3 else "normal")
    ax.set_xlabel("True suicide score", fontsize=13, labelpad=8)
    ax.set_ylabel("Share of users with that score", fontsize=13)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(EDGE)
    ax.grid(axis="y", color=LINE, lw=0.8, zorder=1)
    clean(ax)
    handles = [Patch(color=RIGHT_C, label="Correct"), Patch(color=WRONG_C, label="Incorrect")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 1.0 if a0 else 0.985), ncol=2,
               frameon=False, fontsize=12.5)
    save(fig, "errors_by_score.svg" if a0 else "10_errors_by_score.svg", a0)


def draw_correlations(a0: bool = False) -> None:
    cohort = pd.read_parquet(ROOT / "artifacts" / "cohort_full.parquet")
    cols = ["BFI_O", "BFI_C", "BFI_E", "BFI_A", "BFI_N", "Brooding", "Worry", "Lonely", "PHQ9", "GAD", "suicide"]
    colors = [PERS_C] * 5 + [PSY_C] * 3 + [PSYCH_C] * 2 + [INK]
    R = cohort[cols].astype(float).corr().to_numpy()
    k = len(cols)
    lim = max(0.8, float(np.ceil(np.abs(R[np.tril_indices(k, -1)]).max() * 10) / 10))
    cmap = LinearSegmentedColormap.from_list("corr", ["#1e40af", "#93c5fd", "#f8fafc", "#fca5a5", "#b91c1c"])
    norm = Normalize(-lim, lim)

    fig, ax = plt.subplots(figsize=(9.2, 8.4) if a0 else (10.8, 9.6))
    if a0:
        fig.subplots_adjust(top=0.99, bottom=0.17, left=0.19, right=0.99)
    else:
        fig.subplots_adjust(top=0.90, bottom=0.17, left=0.17, right=0.98)
        headline(
            fig,
            "How the questionnaires relate to each other and to suicide risk",
            f"Pearson r, all {len(cohort)} users. The bottom row is the 0–6 suicide score.",
        )
    for i in range(1, k):
        for j in range(i):
            r = R[i, j]
            ax.add_patch(
                FancyBboxPatch((j - 0.44, i - 0.44), 0.88, 0.88, boxstyle="round,pad=0,rounding_size=0.14",
                               fc=cmap(norm(r)), ec="none")
            )
            label = f"{r:.2f}".replace("-", "−")
            ax.text(j, i, label, ha="center", va="center", fontsize=11.5,
                    color="white" if abs(r) > 0.45 else INK, fontweight="bold" if i == k - 1 else "normal")
    ax.add_patch(
        FancyBboxPatch((-0.5, k - 1.5), k - 1, 1.0, boxstyle="round,pad=0,rounding_size=0.18", fc="none", ec=INK, lw=1.8)
    )
    ax.set_xlim(-0.6, k - 1.4)
    ax.set_ylim(k - 0.4, 0.4)
    ax.set_aspect("equal")
    ax.set_xticks(range(k - 1))
    ax.set_xticklabels([SCALE_NAMES[c] for c in cols[:-1]], rotation=40, ha="right", fontsize=12.5)
    ax.set_yticks(range(1, k))
    ax.set_yticklabels([SCALE_NAMES[c] for c in cols[1:]], fontsize=12.5)
    for lab, c in zip(ax.get_xticklabels(), colors[:-1]):
        lab.set_color(c)
    for lab, c in zip(ax.get_yticklabels(), colors[1:]):
        lab.set_color(c)
        if c == INK:
            lab.set_fontweight("bold")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)

    cax = ax.inset_axes([0.56, 0.93, 0.40, 0.028])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=10.5, length=0, colors=INK)
    cb.set_ticks([-lim, -lim / 2, 0, lim / 2, lim])
    cb.set_ticklabels([f"{v:.1f}".replace("-", "−") for v in [-lim, -lim / 2, 0, lim / 2, lim]])
    cax.set_title("Pearson r", fontsize=11, color=MUTED, pad=4)
    for row, (name, c) in enumerate([("Personality", PERS_C), ("Psychosocial", PSY_C), ("Psychiatric", PSYCH_C)]):
        y = 0.80 - row * 0.05
        ax.add_patch(FancyBboxPatch((0.58, y - 0.012), 0.022, 0.024, boxstyle="round,pad=0,rounding_size=0.004",
                                    transform=ax.transAxes, fc=c, ec="none"))
        ax.text(0.615, y, name, transform=ax.transAxes, fontsize=12.5, color=c, va="center", fontweight="bold")
    ax.text(0.58, 0.63, "Life satisfaction is not shown. Its\ncolumn in the data file repeats the\n"
            "extraversion scores.", transform=ax.transAxes, fontsize=11, color=MUTED, va="top")
    save(fig, "correlations.svg" if a0 else "11_correlations.svg", a0)


def draw_aux(stats: dict, a0: bool = False) -> None:
    folds = pd.DataFrame(stats["aux_fold_r"])
    mean, sd = stats["aux_mean_r"], stats["aux_std_r"]
    vals = folds[AUX_TARGETS].to_numpy()
    lo = float(np.floor((np.nanmin(vals) - 0.03) * 20) / 20)
    hi = float(np.ceil((np.nanmax(vals) + 0.03) * 20) / 20)
    value_x = hi + 0.04

    fig, ax = plt.subplots(figsize=(9.4, 6.3) if a0 else (11.2, 8.0))
    if a0:
        fig.subplots_adjust(top=0.99, bottom=0.16, left=0.22, right=0.97)
    else:
        fig.subplots_adjust(top=0.87, bottom=0.13, left=0.20, right=0.97)
        headline(
            fig,
            "How well the MTM's middle stages predict each questionnaire",
            "Pearson r on held-out users. Light dots are the five folds. The diamond and bar are the mean ± SD.",
        )
    y, ticks, labels, label_colors = 0.0, [], [], []
    for stage_i, (name, color, cols) in enumerate(STAGES):
        top = y
        ax.text(lo + 0.008, y, f"Stage {stage_i + 1}: {name}", fontsize=13, fontweight="bold", color=color, va="center")
        y += 1
        for col in cols:
            f = folds[col].to_numpy()
            m, s_ = mean[col], sd[col]
            ax.plot([m - s_, m + s_], [y, y], color=color, lw=2.4, alpha=0.75, zorder=3, solid_capstyle="round")
            ax.scatter(f, [y] * len(f), s=38, color=color, alpha=0.30, edgecolor="none", zorder=2)
            ax.scatter([m], [y], marker="D", s=90, color=color, edgecolor="white", linewidth=1.3, zorder=4)
            ax.text(value_x, y, f"{m:+.2f}".replace("-", "−"), fontsize=12.5, fontweight="bold", color=color,
                    va="center", ha="left")
            ticks.append(y)
            labels.append(SCALE_NAMES[col] + ("*" if col == "SWL" else ""))
            label_colors.append(INK)
            y += 1
        ax.axhspan(top - 0.55, y - 0.45, color=color, alpha=0.05, zorder=0, lw=0)
        y += 0.5
    ax.text(value_x, -1.0, "mean r", fontsize=11.5, color=MUTED, ha="left", va="center")
    ax.axvline(0, color="#94a3b8", lw=1.3, zorder=1)
    ax.set_xlim(lo, value_x + 0.07)
    ax.set_ylim(y - 0.6, -1.5)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=12.5)
    xt = np.round(np.arange(np.ceil(lo * 10) / 10, hi + 1e-9, 0.1), 1)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{v:.1f}".replace("-", "−") for v in xt])
    ax.grid(axis="x", color=LINE, lw=0.8, zorder=0)
    ax.set_xlabel("Correlation between the head's prediction and the true score", fontsize=13, labelpad=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, labelsize=12.5)
    fig.text(0.22 if a0 else 0.20, 0.035, "* The life-satisfaction column in the data file repeats the extraversion "
             "scores, so this head also predicts extraversion.", fontsize=10.5, color=MUTED)
    save(fig, "aux_scores.svg" if a0 else "12_aux_scores.svg", a0)


ROC_CURVES = (("stm", STM_C, "STM"), ("phq", PHQ_C, "MTM + PHQ-9"))


def draw_roc(a0: bool = False, scores: Path = POSTER / "roc_scores.npz", curves=ROC_CURVES) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    data = np.load(scores)
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(7.8, 7.9) if a0 else (9.0, 9.2))
    if a0:
        fig.subplots_adjust(top=0.99, bottom=0.10, left=0.12, right=0.98)
    else:
        fig.subplots_adjust(top=0.86, bottom=0.09, left=0.11, right=0.97)
        headline(fig, "ROC curves for high suicide risk",
                 "Five held-out test folds. Thin lines are folds; the thick line and band are the mean ± SD.")
    ax.fill_between([0, 1], [0, 1], 0, color="#f8fafc", zorder=0)
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.3, ls=(0, (4, 4)), zorder=1)
    ax.text(0.60, 0.555, "chance", rotation=45, rotation_mode="anchor", color=MUTED, fontsize=11.5,
            ha="center", va="center")
    legend = []
    for key, color, name in curves:
        tprs, aucs = [], []
        for i in range(5):
            y, sc = data[f"y{i}"], data[f"{key}{i}"]
            fpr, tpr, _ = roc_curve(y, sc)
            ax.plot(fpr, tpr, color=color, alpha=0.22, lw=1.1, zorder=2)
            t = np.interp(grid, fpr, tpr)
            t[0] = 0.0
            tprs.append(t)
            aucs.append(roc_auc_score(y, sc))
        m, s_ = np.mean(tprs, axis=0), np.std(tprs, axis=0)
        ax.fill_between(grid, np.clip(m - s_, 0, 1), np.clip(m + s_, 0, 1), color=color, alpha=0.10, lw=0, zorder=2)
        ax.plot(grid, m, color=color, lw=3.0, zorder=4, solid_capstyle="round")
        legend.append((name, color, float(np.mean(aucs)), float(np.std(aucs))))
    for k, (name, color, auc, sd) in enumerate(reversed(legend)):
        y0 = 0.10 + k * 0.115
        ax.add_patch(FancyBboxPatch((0.50, y0 - 0.04), 0.47, 0.085, boxstyle="round,pad=0,rounding_size=0.015",
                                    fc="white", ec=LINE, lw=1.2, zorder=5))
        ax.plot([0.53, 0.60], [y0, y0], color=color, lw=3.0, zorder=6, solid_capstyle="round")
        ax.text(0.63, y0 + 0.012, name, fontsize=13, fontweight="bold", color=INK, va="center", zorder=6)
        ax.text(0.63, y0 - 0.020, f"AUC {auc:.3f} ± {sd:.3f}", fontsize=11.5, color=MUTED, va="center", zorder=6)
        ax.text(0.95, y0, f"{auc:.3f}", fontsize=19, fontweight="bold", color=color, va="center", ha="right", zorder=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.grid(color=LINE, lw=0.8, zorder=0)
    ax.set_xlabel("False positive rate (share of other users flagged)", fontsize=13, labelpad=8)
    ax.set_ylabel("True positive rate (share of high-risk users caught)", fontsize=13, labelpad=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, labelsize=12.5)
    save(fig, "roc.svg" if a0 else "07_roc.svg", a0)


def gauge(s: Svg, x, y, w, prob, thr, color):
    s.rect(x, y, w, 10, fill="#e2e8f0", rx=5)
    s.rect(x, y, max(10, round(prob * w, 1)), 10, fill=color, rx=5, opacity=0.85)
    tx = x + thr * w
    s.line(tx, y - 6, tx, y + 16, stroke=INK, sw=1.6)
    s.circle(x + prob * w, y + 5, 7, fill="#ffffff", stroke=color, sw=2.5)
    s.text(x, y + 32, "0", 11, MUTED)
    s.text(x + w, y + 32, "1", 11, MUTED, anchor="end")
    s.text(tx, y - 10, f"threshold {thr:.2f}", 11, INK, anchor="start" if thr < 0.6 else "end")


def draw_examples(stats: dict) -> None:
    cm = confusion(stats["counts"])
    ex = {e["label"]: e for e in stats["examples"]}
    cells = [
        ("Correct, high risk", 0, 0, True, "Correct", cm["tp"], cm["pos"], "of high-risk users flagged"),
        ("Missed high risk", 0, 1, False, "Missed", cm["fn"], cm["pos"], "of high-risk users missed"),
        ("False alarm", 1, 0, False, "False alarm", cm["fp"], cm["neg"], "of other users flagged"),
        ("Correct, no risk", 1, 1, True, "Correct", cm["tn"], cm["neg"], "of other users cleared"),
    ]
    gx, gy, cw, ch, gap, head_w = 232, 150, 620, 430, 16, 184
    s = Svg(1500, gy + 2 * ch + gap + 28)
    s.text(32, 46, "Where the reported model was right and wrong", 28, weight=700)
    s.text(32, 76, "Held-out users at each fold's development-F1 threshold, the most extreme score in each cell. "
                   "Quotes are verbatim excerpts from that user's posts; names are removed.", 14, MUTED)
    for j, label in enumerate(["Model: high risk", "Model: not high risk"]):
        x = gx + j * (cw + gap)
        s.rect(x, 102, cw, 36, fill="#f1f5f9", rx=18)
        s.text(x + cw / 2, 126, label, 15, INK, 700, "middle")
    for i, (label, sub, n) in enumerate([("High risk", "true score 3–6", cm["pos"]),
                                         ("Not high risk", "true score 0–2", cm["neg"])]):
        y = gy + i * (ch + gap)
        s.rect(32, y, head_w, ch, fill="#f1f5f9", rx=14)
        s.text(32 + head_w / 2, y + ch / 2 - 22, "True label", 12, MUTED, anchor="middle")
        s.text(32 + head_w / 2, y + ch / 2 + 4, label, 18, INK, 700, "middle")
        s.text(32 + head_w / 2, y + ch / 2 + 26, sub, 12.5, MUTED, anchor="middle")
        s.text(32 + head_w / 2, y + ch / 2 + 46, f"{n} users", 12.5, MUTED, anchor="middle")
    for key, i, j, ok, pill, count, total, what in cells:
        e = ex[key]
        x, y = gx + j * (cw + gap), gy + i * (ch + gap)
        color = RIGHT_C if ok else WRONG_C
        s.rect(x, y, cw, ch, fill="#f0fdfa" if ok else "#fff1f2", stroke=color, sw=1.6, rx=14)
        pw = len(pill) * 7.6 + 28
        s.rect(x + 24, y + 22, pw, 26, fill=color, rx=13)
        s.text(x + 24 + pw / 2, y + 40, pill, 13, "#ffffff", 700, "middle")
        s.text(x + 24, y + 98, f"{count}", 44, color, 700)
        nx = x + 24 + len(str(count)) * 26 + 14
        s.text(nx, y + 78, f"{count / total:.0%}", 16, INK, 700)
        s.text(nx, y + 98, what, 13, MUTED)
        s.line(x + 24, y + 118, x + cw - 24, y + 118, stroke="#e2e8f0", sw=1)
        s.text(x + 24, y + 142, f"Example: true score {e['suicide']}", 13, INK, 700)
        s.text(x + cw - 24, y + 142, f"model score {e['prob']:.2f}", 13, color, 700, "end")
        gauge(s, x + 24, y + 166, cw - 48, e["prob"], e["threshold"], color)
        qy = y + 238
        for quote in EXAMPLE_TEXT[key]:
            rows = wrap("“" + quote + "”", cw - 72, 15)
            s.rect(x + 24, qy - 15, 4, len(rows) * 21 - 2, fill=color, rx=2, opacity=0.5)
            qy = s.lines(x + 40, qy, rows, 15, "#334155", italic=True, leading=1.4) + 12
        s.text(x + 24, y + ch - 20, EXAMPLE_NOTE[key], 13, color, 700)
    s.save(POSTER / "13_examples.svg")

    lines = [
        "# Test-set examples",
        "",
        "Each card is one held-out user, chosen for the most extreme model score in its cell.",
        "Quotes are verbatim excerpts from that user's own posts, picked by hand as the passages most related to",
        "distress. Spelling is the user's; names are removed.",
        "The decision uses the MTM + PHQ-9 score and a threshold that maximizes F1 on that fold's development users.",
        "",
    ]
    for key, *_ in cells:
        e = ex[key]
        call = "called high risk" if e["pred"] == 1 else "called not high risk"
        lines += [f"## {key} · true score {e['suicide']} · {call} · score {e['prob']:.2f} · threshold {e['threshold']:.3f}",
                  "", EXAMPLE_NOTE[key], ""] + [f"> {q}\n" for q in EXAMPLE_TEXT[key]]
    (POSTER / "13_examples.md").write_text("\n".join(lines), encoding="utf-8")


def bullet_list(s: Svg, x, y, width, items, color, symbol, size=15.5):
    for k, item in enumerate(items):
        s.circle(x + 13, y - 5, 13, fill=color)
        s.text(x + 13, y, symbol if symbol else str(k + 1), 13, "#ffffff", 700, "middle")
        rows = wrap(item, width - 44, size)
        y = s.lines(x + 40, y, rows, size, INK, leading=1.4) + 16
    return y


def draw_conclusions(stats: dict) -> None:
    cm = confusion(stats["counts"])
    aux = stats["aux_mean_r"]
    acc = (cm["tp"] + cm["tn"]) / (cm["pos"] + cm["neg"])
    majority = cm["neg"] / (cm["pos"] + cm["neg"])
    top_r = max(aux.values())
    lx, cw, cy = 32, 704, 74
    rx_ = lx + cw + 28
    found = [
        "Hidden states from four language models, pooled by attention fusion, beat both of the paper's "
        "high-risk models: single-task 0.716 vs 0.629, multi-task 0.734 vs 0.697.",
        "One multi-task setting shared by every fold generalizes better than a new setting per fold: 0.726 vs 0.702.",
        "A predicted PHQ-9 term with weight 0.1, chosen on development AUC, adds 0.008. "
        "Cohen's d is 0.886, against 0.729 in the paper.",
        f"The multi-task model ranks users better even though its middle stages barely predict the "
        f"questionnaires (best mean r {top_r:.2f}).",
    ]
    limits = [
        "Research reproduction only. It is not a screening tool.",
        f"Yes/no calls are weak. Accuracy is {acc:.0%}, while calling every user low risk scores {majority:.0%}. "
        f"Only {cm['tp']} of {cm['pos']} high-risk users are flagged.",
        "Gains are uneven across folds. About 19 high-risk development users choose each model, so selection is noisy.",
        "The paper's 0.746 is general risk (any ideation, 36% of users). That is a different label from the one here.",
        "Questionnaires are training targets only. The PHQ-9 term is predicted from posts, not taken from the person.",
        "The data file's life-satisfaction column repeats the extraversion scores, so real life satisfaction is missing.",
        "The sample has more distress than the general population.",
    ]
    end = max(
        bullet_list(Svg(1, 1), lx + 24, cy + 222, cw - 48, found, MTM_C, None),
        bullet_list(Svg(1, 1), rx_ + 24, cy + 96, cw - 48, limits, WRONG_C, "!"),
    )
    card_h = round(end - cy + 8)
    s = Svg(1500, cy + card_h + 32)
    s.text(32, 46, "Conclusions and limitations", 28, weight=700)

    s.rect(lx, cy, cw, card_h, fill="#ffffff", stroke=EDGE, rx=16, shadow=True)
    s.rect(lx, cy, cw, 52, fill=MTM_C, rx=16)
    s.rect(lx, cy + 30, cw, 22, fill=MTM_C, rx=0)
    s.text(lx + 24, cy + 34, "What we found", 19, "#ffffff", 700)
    tiles = [
        ("0.734", "This work, MTM + PHQ-9", PHQ_C),
        ("0.716", "This work, STM", STM_C),
        ("0.697", "Paper, best high-risk model", PAPER_C),
    ]
    tw = (cw - 48 - 2 * 14) / 3
    for k, (value, label, color) in enumerate(tiles):
        tx = lx + 24 + k * (tw + 14)
        s.rect(tx, cy + 74, tw, 104, fill="#f8fafc", stroke=LINE, sw=1, rx=12)
        s.rect(tx, cy + 74, 6, 104, fill=color, rx=3)
        s.text(tx + 22, cy + 124, value, 38, color, 700)
        s.text(tx + 22, cy + 150, label, 13, INK, 700)
        s.text(tx + 22, cy + 168, "high-risk test AUC", 12, MUTED)
    bullet_list(s, lx + 24, cy + 222, cw - 48, found, MTM_C, None)

    s.rect(rx_, cy, cw, card_h, fill="#ffffff", stroke=EDGE, rx=16, shadow=True)
    s.rect(rx_, cy, cw, 52, fill=WRONG_C, rx=16)
    s.rect(rx_, cy + 30, cw, 22, fill=WRONG_C, rx=0)
    s.text(rx_ + 24, cy + 34, "Limitations", 19, "#ffffff", 700)
    bullet_list(s, rx_ + 24, cy + 96, cw - 48, limits, WRONG_C, "!")
    s.save(POSTER / "14_conclusions.svg")


def fold_ci(folds):
    f = np.asarray(folds)
    h = 1.96 * f.std(ddof=1) / np.sqrt(len(f))
    return float(f.mean()), float(f.mean() - h), float(f.mean() + h)


def draw_headline(ours=None) -> None:
    """`ours`: (label, fold AUCs, color, paper reference) per bar of this work."""
    from scipy.stats import norm

    from make_figures import MTM_FOLDS, PHQ_FOLDS, STM_FOLDS

    if ours is None:
        ours = [("STM", STM_FOLDS, STM_C, 0.629), ("MTM", MTM_FOLDS, MTM_C, 0.697),
                ("MTM +\nPHQ-9", PHQ_FOLDS, PHQ_C, 0.697)]
    bars = [
        ("STM", (0.629, 0.606, 0.660), "#cbd5e1", None),
        ("MTM", (0.697, 0.690, 0.707), PAPER_C, None),
    ] + [(name, fold_ci(folds), color, ref) for name, folds, color, ref in ours]
    xs = [0, 1] + [2.5 + k for k in range(len(ours))]
    right = xs[-1] + 0.6
    base = 0.5
    fig, ax = plt.subplots(figsize=(8.8, 6.9))
    fig.subplots_adjust(top=0.99, bottom=0.12, left=0.11, right=0.99)
    ax.axvspan(-0.6, 1.6, color="#f8fafc", zorder=0)
    ax.axvspan(1.9, right, color="#f0fdfa", zorder=0)
    ax.plot([1.35, right], [0.697, 0.697], color=PAPER_C, lw=1.6, ls=(0, (5, 4)), zorder=1)
    for x, (name, (m, lo, hi), color, ref) in zip(xs, bars):
        ax.bar(x, m - base, bottom=base, width=0.7, color=color, zorder=2)
        ax.plot([x, x], [lo, hi], color=INK, lw=1.6, zorder=3)
        for yy in (lo, hi):
            ax.plot([x - 0.09, x + 0.09], [yy, yy], color=INK, lw=1.6, zorder=3)
        ax.text(x, hi + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=17, fontweight="bold",
                color=INK if ref is None else color)
        d = float(np.sqrt(2) * norm.ppf(m))
        text_color = INK if ref is None else "white"
        ax.text(x, base + 0.010, f"d = {d:.2f}", ha="center", va="bottom", fontsize=11.5, color=text_color,
                fontweight="bold", zorder=4)
        if ref is not None:
            ax.text(x, lo - 0.010, f"+{m - ref:.3f}", ha="center", va="top", fontsize=13, color="white",
                    fontweight="bold", zorder=4)
    ax.text(0.5, 0.828, "Ophir et al. 2020", ha="center", fontsize=14.5, fontweight="bold", color=INK)
    ax.text(0.5, 0.814, "ELMo embeddings", ha="center", fontsize=12, color=MUTED)
    mid = float(np.mean(xs[2:]))
    ax.text(mid, 0.828, "This work", ha="center", fontsize=14.5, fontweight="bold", color=MTM_C)
    ax.text(mid, 0.814, "hidden states of four LLMs", ha="center", fontsize=12, color=MUTED)
    ax.set_xlim(-0.6, right)
    ax.set_ylim(base, 0.845)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=13.5, fontweight="bold")
    yt = np.round(np.arange(0.5, 0.801, 0.05), 2)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:.2f}" for v in yt])
    ax.set_ylabel("Test AUC, high suicide risk", fontsize=13, labelpad=8)
    ax.grid(axis="y", color=LINE, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(EDGE)
    ax.tick_params(length=0, labelsize=12.5)
    save(fig, "headline.svg", True)


def draw_corr_bars() -> None:
    cohort = pd.read_parquet(ROOT / "artifacts" / "cohort_full.parquet")
    group = {c: (name, color) for name, color, cols in STAGES for c in cols}
    cols = [c for c in AUX_TARGETS if c != "SWL"]
    r = cohort[cols + ["suicide"]].astype(float).corr()["suicide"].drop("suicide").sort_values()
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    fig.subplots_adjust(top=0.99, bottom=0.16, left=0.19, right=0.97)
    ys = np.arange(len(r))
    colors = [group[c][1] for c in r.index]
    ax.barh(ys, r.to_numpy(), color=colors, height=0.68, zorder=2)
    for y, (c, v) in zip(ys, r.items()):
        ax.text(v + (0.008 if v >= 0 else -0.008), y, f"{v:+.2f}".replace("-", "−"), va="center",
                ha="left" if v >= 0 else "right", fontsize=12.5, fontweight="bold", color=group[c][1])
    ax.axvline(0, color="#94a3b8", lw=1.3, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([SCALE_NAMES[c] for c in r.index], fontsize=13)
    ax.set_xlim(-0.27, 0.53)
    xt = [-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{v:.1f}".replace("-", "−") for v in xt])
    ax.set_xlabel("Pearson r with the 0–6 suicide score", fontsize=13, labelpad=6)
    ax.grid(axis="x", color=LINE, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, labelsize=12.5)
    handles = [Patch(color=color, label=name) for name, color, _ in STAGES]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=12.5, handlelength=1.1)
    save(fig, "corr_bars.svg", True)


def draw_examples_a0(stats: dict, texts: dict | None = None) -> None:
    texts = {**EXAMPLE_TEXT, **(texts or {})}
    cm = confusion(stats["counts"])
    ex = {e["label"]: e for e in stats["examples"]}
    picks = {"Correct, high risk": [0], "Missed high risk": [0, 2], "False alarm": [0], "Correct, no risk": [0, 1]}
    cells = [
        ("Correct, high risk", 0, 0, True, "Correct", cm["tp"], cm["pos"], "of high-risk users flagged"),
        ("Missed high risk", 0, 1, False, "Missed", cm["fn"], cm["pos"], "of high-risk users missed"),
        ("False alarm", 1, 0, False, "False alarm", cm["fp"], cm["neg"], "of other users flagged"),
        ("Correct, no risk", 1, 1, True, "Correct", cm["tn"], cm["neg"], "of other users cleared"),
    ]
    W, gx, gy, gap, ch = 1150, 58, 46, 14, 252
    cw = (W - gx - gap) / 2
    s = Svg(W, gy + 2 * ch + gap)
    for j, label in enumerate(["Model: high risk", "Model: not high risk"]):
        x = gx + j * (cw + gap)
        s.rect(x, 0, cw, 34, fill="#f1f5f9", rx=17)
        s.text(x + cw / 2, 23, label, 16, INK, 700, "middle")
    for i, label in enumerate(["True: high risk (3–6)", "True: not high risk (0–2)"]):
        y = gy + i * (ch + gap)
        s.rect(0, y, 44, ch, fill="#f1f5f9", rx=12)
        cx, cy = 22, y + ch / 2
        s.raw(f'<text x="{cx}" y="{cy + 6}" font-size="15" font-weight="700" fill="{INK}" text-anchor="middle" '
              f'transform="rotate(-90 {cx} {cy})">{label}</text>')
    for key, i, j, ok, pill, count, total, what in cells:
        e = ex[key]
        x, y = gx + j * (cw + gap), gy + i * (ch + gap)
        color = RIGHT_C if ok else WRONG_C
        s.rect(x, y, cw, ch, fill="#f0fdfa" if ok else "#fff1f2", stroke=color, sw=1.6, rx=14)
        pw = text_width(pill, 14, True) + 26
        s.rect(x + 20, y + 16, pw, 28, fill=color, rx=14)
        s.text(x + 20 + pw / 2, y + 35, pill, 14, "#ffffff", 700, "middle")
        nx = x + 20 + pw + 14
        s.text(nx, y + 40, f"{count}", 28, color, 700)
        s.text(nx + text_width(str(count), 28, True) + 8, y + 38, f"users, {count / total:.0%} {what}", 14.5, MUTED)
        thr = e["threshold"]
        s.text(x + 20, y + 70, f"Example: true score {e['suicide']}, model score {e['prob']:.2f} "
                               f"(threshold {thr:.2f})" if thr >= 0.01 else
               f"Example: true score {e['suicide']}, model score {e['prob']:.3f} (threshold {thr:.3f})",
               13.5, INK, 700)
        qy = y + 100
        for q in picks[key]:
            rows = wrap("“" + texts[key][q] + "”", cw - 64, 16)
            s.rect(x + 20, qy - 15, 4, len(rows) * 21.6 - 3, fill=color, rx=2, opacity=0.5)
            qy = s.lines(x + 36, qy, rows, 16, "#334155", italic=True, leading=1.35) + 8
        s.text(x + 20, y + ch - 16, EXAMPLE_NOTE[key], 14.5, color, 700)
    A0.mkdir(exist_ok=True)
    s.save(A0 / "examples.svg")


def draw_a0(stats: dict) -> None:
    draw_roc(True)
    draw_headline()
    draw_errors(stats, True)
    draw_correlations(True)
    draw_corr_bars()
    draw_aux(stats, True)
    draw_examples_a0(stats)


def draw_panels(stats: dict) -> None:
    draw_roc()
    draw_errors(stats)
    draw_correlations()
    draw_aux(stats)
    draw_examples(stats)
    draw_conclusions(stats)
    draw_a0(stats)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("redraw", "a0"):
        saved = json.loads((POSTER / "panel_stats.json").read_text(encoding="utf-8"))
        draw_a0(saved) if sys.argv[1] == "a0" else draw_panels(saved)
    else:
        main()
