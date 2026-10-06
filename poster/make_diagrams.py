"""Poster diagrams that need no data: the full system (08) and the hyperparameter table (09)."""
from __future__ import annotations

from pathlib import Path

from svgkit import EDGE, INK, LINE, MTM_C, MUTED, PERS_C, PHQ_C, PSY_C, PSYCH_C, STM_C, Svg, text_width

POSTER = Path(__file__).resolve().parent

LLMS = ["Qwen3-32B", "DeepSeek-R1-Distill-32B", "Gemma-4-26B (MoE)", "Llama-3.3-70B"]
LLM_COLORS = ["#6366f1", "#0ea5e9", "#10b981", "#f43f5e"]
POS_NAMES = ["Input only", "Last prompt token", "Chain of thought", "Final prediction"]
POS_COLORS = ["#94a3b8", "#60a5fa", "#a78bfa", "#f59e0b"]

STM_FOLDS = [0.7712087912087912, 0.6571945508727117, 0.7791613452532993, 0.7397214854111406, 0.6339522546419099]
PHQ_FOLDS = [0.7351648351648351, 0.7120051085568327, 0.7690506598552576, 0.7416003536693192, 0.7144120247568524]
STM_SETTINGS = [
    (3, 1024, 0.01, 1000),
    (1, 64, 0.005, 1000),
    (3, 1024, 0.005, 5000),
    (1, 512, 0.001, 5000),
    (1, 512, 0.05, 2500),
]


def vec(s: Svg, x, y, color, n=5, size=9, gap=2):
    for k in range(n):
        s.rect(x + k * (size + gap), y, size, size, fill=color, rx=2)


def system() -> None:
    s = Svg(1680, 950)
    s.text(36, 46, "From a user's posts to a high-risk score", 30, weight=700)
    s.text(
        36,
        76,
        "Hidden states from four language models are pooled, compressed to 1024 features, and passed to two classifiers.",
        16,
        MUTED,
    )
    top, h = 104, 196

    s.rect(36, top, 164, h, fill="#ffffff", stroke=EDGE, rx=14, shadow=True)
    for k in range(3):
        yy = top + 18 + k * 44
        s.rect(52, yy, 132, 36, fill="#f1f5f9", rx=8)
        s.circle(67, yy + 18, 8, fill="#cbd5e1")
        s.rect(82, yy + 10, 88 - 18 * (k % 2), 6, fill="#cbd5e1", rx=3)
        s.rect(82, yy + 21, 64 + 16 * (k % 2), 6, fill="#e2e8f0", rx=3)
    s.text(118, top + 168, "Facebook posts", 16, weight=700, anchor="middle")
    s.text(118, top + 186, "10 or more per user", 12, MUTED, anchor="middle")
    s.line(206, top + h / 2, 228, top + h / 2, arrow=True, sw=2)

    s.rect(236, top, 254, h, fill="#ffffff", stroke=EDGE, rx=14, shadow=True)
    s.text(363, top + 32, "4 language models", 16, weight=700, anchor="middle")
    for k, (name, color) in enumerate(zip(LLMS, LLM_COLORS)):
        yy = top + 48 + k * 31
        s.rect(254, yy, 218, 25, fill="#f8fafc", stroke=LINE, sw=1, rx=12.5)
        s.circle(270, yy + 12.5, 5, fill=color)
        s.text(283, yy + 17, name, 13)
    s.text(363, top + 186, "hidden states from several layers", 12, MUTED, anchor="middle")
    s.line(496, top + h / 2, 518, top + h / 2, arrow=True, sw=2)

    s.rect(526, top, 244, h, fill="#ffffff", stroke=EDGE, rx=14, shadow=True)
    s.text(648, top + 32, "4 positions per layer", 16, weight=700, anchor="middle")
    for k in range(4):
        yy = top + 54 + k * 30
        vec(s, 546, yy, POS_COLORS[k])
        s.text(610, yy + 9, POS_NAMES[k], 13)
    s.text(648, top + 186, "56 vectors per user", 12, MUTED, anchor="middle")
    s.line(776, top + h / 2, 796, top + h / 2, arrow=True, sw=2)

    s.rect(802, top, 842, h, fill="#f0fdfa", stroke=MTM_C, sw=2, rx=14, shadow=True)
    s.text(826, top + 32, "Attention fusion", 18, MTM_C, 700)
    s.text(826 + text_width("Attention fusion", 18, True) + 14, top + 32, "fit on each training fold only", 13, MUTED)
    sy, sh = top + 50, 108
    mid = sy + sh / 2

    s.rect(826, sy, 120, sh, fill="#ffffff", stroke=EDGE, rx=10)
    for k in range(4):
        vec(s, 859, sy + 16 + k * 15, POS_COLORS[k], size=9, gap=2)
    s.text(886, sy + 88, "4 vectors", 13, weight=700, anchor="middle")
    s.text(886, sy + 102, "one model and layer", 10.5, MUTED, anchor="middle")
    s.line(950, mid, 968, mid, arrow=True)

    s.rect(972, sy, 130, sh, fill="#ffffff", stroke=EDGE, rx=10)
    s.text(1037, sy + 34, "Score", 14, weight=700, anchor="middle")
    s.text(1037, sy + 60, "s = w · h", 15, anchor="middle")
    s.text(1037, sy + 86, "linear, no bias", 11.5, MUTED, anchor="middle")
    s.line(1106, mid, 1124, mid, arrow=True)

    s.rect(1128, sy, 140, sh, fill="#ffffff", stroke=EDGE, rx=10)
    s.text(1198, sy + 24, "Softmax", 14, weight=700, anchor="middle")
    base = sy + 82
    for k, hh in enumerate([20, 44, 30, 14]):
        s.rect(1151 + k * 25, base - hh, 18, hh, fill=POS_COLORS[k], rx=3)
    s.line(1146, base, 1250, base, stroke=EDGE, sw=1)
    s.text(1198, sy + 100, "weights sum to 1", 11.5, MUTED, anchor="middle")
    s.line(1272, mid, 1290, mid, arrow=True)

    s.rect(1294, sy, 140, sh, fill="#ffffff", stroke=EDGE, rx=10)
    s.text(1364, sy + 24, "Weighted sum", 14, weight=700, anchor="middle")
    for k, c in enumerate(["#5eead4", "#2dd4bf", "#14b8a6", "#0d9488", "#0f766e"]):
        s.rect(1330 + k * 14, sy + 46, 12, 12, fill=c, rx=2)
    s.text(1364, sy + 88, "one vector per", 11.5, MUTED, anchor="middle")
    s.text(1364, sy + 102, "model and layer", 11.5, MUTED, anchor="middle")
    s.line(1438, mid, 1456, mid, arrow=True)

    s.rect(1460, sy, 164, sh, fill=INK, rx=10)
    s.text(1542, sy + 32, "Concatenate all 14", 13.5, "#ffffff", 700, "middle")
    s.text(1542, sy + 60, "Linear to 1024", 16, "#99f6e4", 700, "middle")
    s.text(1542, sy + 86, "about 79k inputs", 11.5, "#cbd5e1", anchor="middle")
    s.text(
        826,
        top + 184,
        "The weights choose among the 4 positions inside each model and layer. This is not attention over tokens.",
        12,
        MUTED,
    )

    fork_y = 328
    s.path(f"M1542,{top + h} V{fork_y}", sw=2)
    s.line(418, fork_y, 1542, fork_y, sw=2)
    s.text(860, fork_y - 9, "1024 features, shared by both classifiers", 13, MUTED, anchor="middle")
    s.line(418, fork_y, 418, 406, arrow=True, sw=2)
    s.line(1290, fork_y, 1290, 406, arrow=True, sw=2)

    p_top, p_h = 352, 578
    s.rect(36, p_top, 764, p_h, fill="#f8fafc", stroke=STM_C, sw=2, rx=16)
    s.rect(830, p_top, 814, p_h, fill="#f8fafc", stroke=MTM_C, sw=2, rx=16)
    s.text(60, 390, "STM", 24, STM_C, 700)
    s.text(60, 414, "single task: suicide only", 14, MUTED)
    s.text(854, 390, "MTM", 24, MTM_C, 700)
    s.text(854, 414, "multi-task: clinical stages, then suicide", 14, MUTED)

    # STM network
    s.rect(308, 410, 220, 40, fill=INK, rx=10)
    s.text(418, 436, "1024 features", 15, "#ffffff", 700, "middle")
    rows = [500, 576, 652]
    xs = [418 + (i - 4) * 42 for i in range(9)]
    out_y = 736
    for px in [318 + i * 25 for i in range(9)]:
        for x in xs:
            s.line(px, 450, x, rows[0], stroke="#bfdbfe", sw=0.7)
    for a, b in zip(rows[:-1], rows[1:]):
        for x1 in xs:
            for x2 in xs:
                s.line(x1, a, x2, b, stroke=EDGE, sw=0.6, opacity=0.8)
    for x in xs:
        s.line(x, rows[-1], 418, out_y, stroke=EDGE, sw=0.8)
    for j, y in enumerate(rows):
        for x in xs:
            if j == 0:
                s.circle(x, y, 11, fill="#dbeafe", stroke=STM_C, sw=1.5)
            else:
                s.circle(x, y, 11, fill="#ffffff", stroke=STM_C, sw=1.5, dash="4 3")
    s.circle(418, out_y, 20, fill=STM_C)
    for label, y in zip(["Layer 1", "Layer 2", "Layer 3", "Output"], rows + [out_y]):
        s.text(60, y + 5, label, 13, MUTED, 700)
    for label, y in zip(
        ["64 to 1024 tanh units", "folds 0 and 2 only", "folds 0 and 2 only", "suicide logit, BCE loss"], rows + [out_y]
    ):
        s.text(614, y + 5, label, 13, MUTED)
    s.line(418, out_y + 22, 418, 834, arrow=True, sw=2)
    s.rect(128, 840, 580, 44, fill=STM_C, rx=12)
    s.text(418, 868, "Mean test AUC 0.716", 17, "#ffffff", 700, "middle")
    s.text(418, 912, "Depth, width, learning rate, and epochs are chosen on each fold's development set.", 12, MUTED, anchor="middle")

    # MTM cascade
    cx = 1290
    s.rect(1180, 410, 220, 40, fill=INK, rx=10)
    s.text(cx, 436, "1024 features", 15, "#ffffff", 700, "middle")
    s.line(cx, 450, cx, 462, arrow=True, sw=2)
    s.rect(1110, 466, 360, 42, fill="#334155", rx=10)
    s.text(cx, 493, "Shared layer: 512 tanh units", 15, "#ffffff", 700, "middle")
    stages = [
        ("Personality: 5 scores", "Big Five: O, C, E, A, N", PERS_C, "MSE"),
        ("Psychosocial: 4 scores", "brooding, worry, loneliness, life satisfaction", PSY_C, "MSE"),
        ("Psychiatric: 2 scores", "PHQ-9 depression, GAD anxiety", PSYCH_C, "MSE"),
        ("Suicide: 1 logit", "high risk means a score of 3 or more", MTM_C, "BCE"),
    ]
    ys = [532, 608, 684, 760]
    bh = 54
    rail_x = 1082
    s.path(f"M1110,487 H{rail_x} V{ys[-1] + bh / 2}", stroke="#94a3b8", sw=2)
    for k, ((title, sub, color, loss), y) in enumerate(zip(stages, ys)):
        last = k == len(stages) - 1
        s.rect(1110, y, 360, bh, fill=color if last else "#ffffff", stroke=color, sw=1.6, rx=10)
        if not last:
            s.rect(1110, y, 10, bh, fill=color, rx=4)
        s.text(1134, y + 23, title, 15, "#ffffff" if last else INK, 700)
        s.text(1134, y + 42, sub, 12, "#ccfbf1" if last else MUTED)
        if k > 0:
            s.line(rail_x, y + bh / 2, 1106, y + bh / 2, stroke="#94a3b8", sw=2, arrow=True)
        if loss == "BCE":
            s.rect(1482, y + 15, 58, 24, fill=color, rx=12)
            s.text(1511, y + 32, loss, 12, "#ffffff", 700, "middle")
        else:
            s.rect(1482, y + 15, 58, 24, fill="#ffffff", stroke=color, sw=1.4, rx=12)
            s.text(1511, y + 32, loss, 12, color, 700, "middle")
    s.line(cx, 508, cx, ys[0] - 3, arrow=True, sw=2)
    for a, b in zip(ys[:-1], ys[1:]):
        s.line(cx, a + bh, cx, b - 3, arrow=True, sw=2)
    s.line(cx, ys[-1] + bh, cx, 836, arrow=True, sw=2)
    s.rect(1040, 840, 500, 44, fill=PHQ_C, rx=12)
    s.text(cx, 868, "Logit + 0.1 × predicted PHQ-9: mean test AUC 0.734", 16, "#ffffff", 700, "middle")
    s.text(cx, 912, "The PHQ-9 term is a ridge prediction from the 1024 features, fit on the training fold.", 12, MUTED, anchor="middle")

    lx, ly = 856, 474
    s.line(lx, ly, lx + 34, ly, arrow=True, sw=2)
    s.text(lx + 44, ly + 4, "previous stage's prediction", 12.5)
    s.line(lx, ly + 30, lx + 34, ly + 30, stroke="#94a3b8", sw=2, arrow=True)
    s.text(lx + 44, ly + 34, "shared layer, passed again", 12.5)
    s.rect(lx, ly + 50, 40, 20, fill="#ffffff", stroke=PSY_C, sw=1.4, rx=10)
    s.text(lx + 20, ly + 64, "MSE", 10.5, PSY_C, 700, "middle")
    s.text(lx + 50, ly + 64, "z-scored questionnaire", 12.5)
    s.rect(lx, ly + 80, 40, 20, fill=MTM_C, rx=10)
    s.text(lx + 20, ly + 94, "BCE", 10.5, "#ffffff", 700, "middle")
    s.text(lx + 50, ly + 94, "high-risk label", 12.5)

    s.save(POSTER / "08_system.svg")


def bar(s: Svg, x, y, value, color, bold, lo=0.6, hi=0.8, width=120):
    s.rect(x, y - 9, width, 18, fill="#f1f5f9", rx=4)
    w = max(0.0, min(1.0, (value - lo) / (hi - lo))) * width
    s.rect(x, y - 9, round(w, 1), 18, fill=color, rx=4)
    s.text(x + width + 10, y + 5, f"{value:.3f}", 15, INK, 700 if bold else 400)


def hyperparams() -> None:
    s = Svg(1500, 650)
    s.text(32, 46, "Hyperparameters of the reported models", 28, weight=700)
    s.text(
        32,
        76,
        "The STM is picked separately on each fold. The MTM uses one setting on every fold, chosen by mean development AUC.",
        15,
        MUTED,
    )
    x0, x1 = 32, 1468
    stm_end, mtm_start = 924, 940
    cols = {"fold": 32, "net": 104, "size": 230, "lr": 430, "ep": 560, "auc": 680}
    s.rect(x0, 100, stm_end - x0, 38, fill="#dbeafe", rx=8)
    s.text(48, 125, "STM, chosen per fold", 15, STM_C, 700)
    s.rect(mtm_start, 100, x1 - mtm_start, 38, fill="#ccfbf1", rx=8)
    s.text(mtm_start + 16, 125, "MTM + PHQ-9, one shared setting", 15, MTM_C, 700)

    hy = 164
    for label, x in [("Fold", cols["fold"] + 16), ("Network", cols["net"]), ("Layers × units", cols["size"]),
                     ("Learning rate", cols["lr"]), ("Epochs", cols["ep"]), ("Test AUC", cols["auc"])]:
        s.text(x, hy, label, 13, MUTED, 700)
    s.text(mtm_start + 16, hy, "Setting", 13, MUTED, 700)
    s.text(1210, hy, "Test AUC", 13, MUTED, 700)
    s.line(x0, 176, x1, 176, stroke=EDGE, sw=1.2)

    row_h, ry = 54, 178
    for i, ((layers, units, lr, ep), a, b) in enumerate(zip(STM_SETTINGS, STM_FOLDS, PHQ_FOLDS)):
        y = ry + i * row_h
        if i % 2 == 0:
            s.rect(x0, y, stm_end - x0, row_h, fill="#f8fafc", rx=0)
            s.rect(1196, y, x1 - 1196, row_h, fill="#f8fafc", rx=0)
        cy = y + row_h / 2
        s.text(cols["fold"] + 26, cy + 5, str(i), 16, INK, 700, "middle")
        for k in range(layers):
            hh = 10 + 3.4 * (units.bit_length() - 5)
            s.rect(cols["net"] + 4 + k * 18, cy - hh / 2, 12, round(hh, 1), fill="#93c5fd", stroke=STM_C, sw=1, rx=3)
        s.text(cols["size"], cy + 5, f"{layers} × {units}", 16)
        s.text(cols["lr"], cy + 5, f"{lr:g}", 16)
        s.text(cols["ep"], cy + 5, f"{ep:,}", 16)
        bar(s, cols["auc"], cy, a, STM_C, a > b)
        bar(s, 1210, cy, b, PHQ_C, b > a)
    mean_y = ry + 5 * row_h
    s.line(x0, mean_y, stm_end, mean_y, stroke=INK, sw=1.2)
    s.line(1196, mean_y, x1, mean_y, stroke=INK, sw=1.2)
    cy = mean_y + 26
    s.text(cols["fold"] + 26, cy + 5, "Mean", 15, INK, 700, "middle")
    s.text(cols["size"], cy + 5, "five different settings", 14, MUTED)
    bar(s, cols["auc"], cy, sum(STM_FOLDS) / 5, STM_C, False)
    bar(s, 1210, cy, sum(PHQ_FOLDS) / 5, PHQ_C, True)
    for x in (cols["auc"], 1210):
        px = x + (0.697 - 0.6) / 0.2 * 120
        s.line(px, ry + 4, px, mean_y + 44, stroke=INK, sw=1, dash="3 3", opacity=0.55)
    s.text(cols["auc"] + 58, mean_y + 64, "dashed line: paper MTM, 0.697", 11.5, MUTED, anchor="middle")

    mx, my = mtm_start + 16, ry + 26
    s.rect(mtm_start, ry, 248, 5 * row_h, fill="#f0fdfa", stroke="#99f6e4", sw=1, rx=10)
    s.rect(mx + 4, my + 6, 12, 23, fill="#5eead4", stroke=MTM_C, sw=1, rx=3)
    for k, c in enumerate([PERS_C, PSY_C, PSYCH_C, MTM_C]):
        s.rect(mx + 26 + k * 16, my + 10, 10, 15, fill=c, rx=2, opacity=0.85)
    s.text(mx, my + 66, "1 × 512 shared layer", 16, INK, 700)
    s.text(mx, my + 90, "tanh, learning rate 0.005", 14)
    s.text(mx, my + 112, "1,000 epochs", 14)
    s.text(mx, my + 140, "4 cascade subnets, 512 units", 13, MUTED)
    s.text(mx, my + 160, "plus 0.1 × predicted PHQ-9", 13, PHQ_C, 700)
    s.text(mx, my + 196, "same on every fold", 13, MTM_C, 700)

    chips = [
        "RMSprop, momentum 0.9",
        "Batch 32",
        "Early stopping, patience 200",
        "Seed 42",
        "Suicide loss: unweighted BCE",
        "MTM side losses: MSE on z-scored scales",
        "Fusion: Adam, lr 0.01, patience 50",
        "Fusion positive weight 1.5 × negatives / positives",
        "PHQ-9 term: ridge regression, weight 0.1",
    ]
    s.text(32, 548, "Shared by both", 13, MUTED, 700)
    x, y = 32, 560
    for chip in chips:
        w = text_width(chip, 13) + 26
        if x + w > x1:
            x, y = 32, y + 38
        s.rect(x, y, w, 28, fill="#f1f5f9", stroke=LINE, sw=1, rx=14)
        s.text(x + 13, y + 19, chip, 13)
        x += w + 10
    s.text(
        32,
        y + 60,
        "Search space: 504 settings per model (1 to 3 layers, 16 to 1024 units, tanh or sigmoid, 4 learning rates, 1,000 to 5,000 epochs).",
        12,
        MUTED,
    )
    s.h = y + 76
    s.save(POSTER / "09_hyperparams.svg")


if __name__ == "__main__":
    system()
    hyperparams()
