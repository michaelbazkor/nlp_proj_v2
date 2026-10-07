"""Poster diagrams that need no data: the full system (08) and the hyperparameter table (09)."""
from __future__ import annotations

from pathlib import Path

from svgkit import EDGE, INK, LINE, MTM_C, MUTED, PERS_C, PHQ_C, PSY_C, PSYCH_C, STM_C, Svg, esc, text_width, wrap

POSTER = Path(__file__).resolve().parent
MONO = "Consolas, 'Courier New', monospace"

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
LLM_TAPS = [
    ("Qwen3-32B", 64, [20, 40, 60, 64], 5120, "bf16, reasons first"),
    ("DeepSeek-R1-Distill-Qwen-32B", 64, [20, 40, 60, 64], 5120, "bf16, reasons first"),
    ("Gemma-4-26B-A4B (MoE)", 30, [20, 30], 2816, "bf16, 4B active"),
    ("Llama-3.3-70B-Instruct", 80, [20, 40, 60, 80], 8192, "8-bit weights"),
]
POS_TEXT = ["#475569", "#2563eb", "#7c3aed", "#b45309"]


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


def mono(s: Svg, x, y, parts, size=15):
    spans = []
    for text, color, *style in parts:
        a = f' fill="{color}"'
        if "b" in style:
            a += ' font-weight="700"'
        if "i" in style:
            a += ' font-style="italic"'
        spans.append(f"<tspan{a}>{esc(text)}</tspan>")
    s.raw(f'<text x="{x}" y="{y}" font-size="{size}" font-family="{MONO}">{"".join(spans)}</text>')


def step(s: Svg, y, n, title, sub=""):
    s.circle(19, y - 8, 19, fill=INK)
    s.text(19, y - 1, str(n), 19, "#ffffff", 700, "middle")
    s.text(50, y, title, 24, INK, 700)
    if sub:
        s.text(round(50 + len(title) * 24 * 0.53 + 16), y, sub, 15, MUTED)


def bullets(s: Svg, x, y, width, items, size=13.5, dot=INK, gap=8):
    for item in items:
        s.circle(x + 5, y - size * 0.35, 3.5, fill=dot)
        y = s.lines(x + 18, y, wrap(item, width - 18, size), size, INK, leading=1.38) + gap
    return y


def method(v1_folds: list[float] | None = None, out: Path | None = None) -> None:
    """`v1_folds`: test AUCs of the aux-pretrained, weight-averaged MTM; draws its box instead of MTM + PHQ-9."""
    W = 1200
    s = Svg(W, 1965)
    s.shadows = False

    # 1. corpus
    step(s, 32, 1, "Compile each user's posts into one corpus")
    t, h = 56, 180
    s.rect(0, t, 236, h, fill="#ffffff", stroke=EDGE, rx=14, shadow=True)
    for k in range(2):
        yy = t + 14 + k * 40
        s.rect(14, yy, 208, 32, fill="#f1f5f9", rx=8)
        s.circle(29, yy + 16, 7, fill="#cbd5e1")
        s.rect(44, yy + 9, 150 - 30 * k, 6, fill="#cbd5e1", rx=3)
        s.rect(44, yy + 19, 110 + 20 * k, 6, fill="#e2e8f0", rx=3)
    yy = t + 94
    s.rect(14, yy, 208, 46, fill="#f1f5f9", rx=8)
    s.rect(22, yy + 6, 46, 34, fill="#ddd6fe", rx=4)
    s.path(f"M25,{yy + 38} L38,{yy + 19} L47,{yy + 29} L54,{yy + 22} L65,{yy + 38} Z", stroke=PERS_C, sw=1, fill=PERS_C)
    s.circle(58, yy + 13, 4, fill="#ffffff")
    s.rect(78, yy + 14, 120, 6, fill="#cbd5e1", rx=3)
    s.rect(78, yy + 26, 90, 6, fill="#e2e8f0", rx=3)
    s.text(118, t + 164, "Status posts and photos", 14, INK, 700, "middle")
    s.line(242, t + h / 2, 262, t + h / 2, arrow=True, sw=2)

    cx0 = 268
    s.rect(cx0, t, 212, h, fill="#f5f3ff", stroke=PERS_C, sw=1.6, rx=14)
    s.text(cx0 + 16, t + 30, "Caption each photo", 15.5, PERS_C, 700)
    s.text(cx0 + 16, t + 54, "Qwen2.5-VL-7B-Instruct", 13.5, INK, 700)
    s.lines(cx0 + 16, t + 80, wrap("2–4 objective sentences on content and mood; visible text is copied verbatim",
                                   184, 13), 13, MUTED, leading=1.35)
    mono(s, cx0 + 16, t + 166, [("[image] ", PERS_C, "b"), ("<caption>", INK)], 14)
    s.line(486, t + h / 2, 500, t + h / 2, arrow=True, sw=2)

    bx = 506
    cw8 = 14.5 * 0.55
    s.rect(bx, t, W - bx, h, fill="#f8fafc", stroke=EDGE, rx=14)
    s.text(bx + 20, t + 30, "One corpus per user", 15.5, INK, 700)
    s.text(W - 20, t + 30, "illustrative text, not a real user", 12.5, MUTED, anchor="end", italic=True)
    rows = [
        (0, [("[post] ", STM_C, "b"), ("can't sleep again. third night in a row", INK)]),
        (0, [("[post] ", STM_C, "b"), ("brunch with the girls!! ", INK), ("[image] ", PERS_C, "b"),
             ("Three women smile", INK)]),
        (7, [("at a sunny café table with pancakes and coffee.", INK)]),
        (0, [("[post] ", STM_C, "b"), ("new job starts monday, wish me luck ", INK), ("[post] ", STM_C, "b"),
             ("...", MUTED)]),
    ]
    for k, (indent, parts) in enumerate(rows):
        mono(s, round(bx + 20 + indent * cw8, 1), t + 62 + k * 24, parts, 14.5)
    s.text(bx + 20, t + h - 12, "10 or more posts per user, in shuffled order (seed 42). No profile or demographic fields.",
           13, MUTED)

    # 2. prompt
    step(s, 284, 2, "Insert the corpus into one task prompt", "exact wording; [...] marks shortened text")
    t, h = 306, 276
    s.rect(0, t, 800, h, fill="#0f172a", rx=14)
    white, grey, amber, blue = "#e2e8f0", "#94a3b8", "#fbbf24", "#93c5fd"
    prompt = [
        [("You are an expert supervisor evaluating suicide-risk signals in social media language.", white)],
        [("Input: a collection of Facebook posts authored by a single user. (order is arbitrary)", white)],
        [("[...] isolation, burden, emotional pain, entrapment, hopelessness, death, self-harm [...]", grey, "i")],
        [("Remember that majority of users do not have suicidal ideation, do not overinterpret", white)],
        [("posts that are not overtly suicidal.", white)],
        [("User posts:", white)],
        None,
        [("Format your answer exactly as:", white)],
        [("Rationale: <2-3 sentences>", blue)],
        [("Prediction: RISK=<1 for elevated risk, 0 for low risk>", blue)],
    ]
    for k, row in enumerate(prompt):
        y = t + 36 + k * 24
        if row is None:
            w = round(19 * 15 * 0.55 + 16)
            s.rect(14, y - 18, w, 26, fill=amber, rx=6)
            mono(s, 22, y, [("[insert posts here]", "#0f172a", "b")], 15)
            s.text(22 + w + 8, y - 1, "← the corpus from step 1 goes here", 14, amber, 700)
        else:
            mono(s, 22, y, row, 15)
    rx0 = 820
    s.rect(rx0, t, W - rx0, h, fill="#ffffff", stroke=EDGE, rx=14, shadow=True)
    s.text(rx0 + 20, t + 32, "How each model reads it", 15.5, INK, 700)
    bullets(s, rx0 + 20, t + 62, W - rx0 - 40, [
        "Each model's own chat template",
        "Greedy decoding in one generate() pass, with forward hooks on the tapped layers",
        "Qwen3 and DeepSeek-R1 reason before answering, up to 512 new tokens. Gemma and Llama get 256",
        "32k-token context. Longer corpora are split at post boundaries and the chunks' vectors averaged",
    ])

    # 3. token positions and models
    step(s, 634, 3, "Read hidden states at four token positions", "frozen models, no fine-tuning")
    st, sh = 694, 44
    s.path(f"M0,{st - 6} V{st - 12} H760 V{st - 6}", stroke=MUTED, sw=1.4)
    s.text(380, st - 20, "prompt, read in one forward pass", 13.5, MUTED, 700, "middle")
    s.path(f"M770,{st - 6} V{st - 12} H{W} V{st - 6}", stroke=MUTED, sw=1.4)
    s.text(985, st - 20, "output, generated greedily", 13.5, MUTED, 700, "middle")
    segs = [
        (0, 180, "instructions", "#f1f5f9"),
        (180, 640, "user posts from step 1", "#e2e8f0"),
        (640, 760, "format", "#f1f5f9"),
        (770, 1100, "reasoning and rationale", "#ede9fe"),
        (1100, W, "RISK=1", "#fef3c7"),
    ]
    for x0, x1, label, fill in segs:
        s.rect(x0, st, x1 - x0, sh, fill=fill, stroke="#cbd5e1", sw=1, rx=6)
        for x in range(x0 + 12, x1 - 6, 12):
            s.line(x, st + 8, x, st + sh - 8, stroke="#cbd5e1", sw=1, opacity=0.8)
        lw = text_width(label, 13.5, True) + 18
        cx = (x0 + x1) / 2 - (8 if x1 == W else 0)
        s.rect(round(cx - lw / 2, 1), st + 11, round(lw, 1), 22, fill=fill, rx=6)
        s.text(cx, st + 27, label, 13.5, INK, 700, "middle")
    s.rect(748, st + 4, 9, sh - 8, fill=POS_COLORS[1], rx=2)
    s.rect(W - 12, st + 4, 9, sh - 8, fill=POS_COLORS[3], rx=2)
    by = st + sh + 8
    s.path(f"M190,{by} V{by + 8} H630 V{by}", stroke=POS_COLORS[0], sw=2.5)
    s.path(f"M780,{by} V{by + 8} H1090 V{by}", stroke=POS_COLORS[2], sw=2.5)
    callouts = [
        (410, by + 8, "middle", "Input only", "mean over the post tokens", 0),
        (752, st + sh, "middle", "Last prompt token", "before any output exists", 1),
        (960, by + 8, "middle", "Chain of thought", "mean over generated tokens", 2),
        (W, st + sh, "end", "Final prediction", "last generated token", 3),
    ]
    for x, y0, anchor, name, sub, k in callouts:
        lx = x - 7 if anchor == "end" else x
        s.line(lx, y0 + 2, lx, by + 22, stroke=POS_COLORS[k], sw=2.5)
        s.text(x, by + 42, name, 15.5, POS_TEXT[k], 700, anchor)
        s.text(x, by + 61, sub, 13.5, MUTED, anchor=anchor)

    ty = by + 96
    for label, x, anchor in [("Model", 0, "start"), ("Tapped layers (dots) along the model's depth", 300, "start"),
                             ("Hidden size", 1000, "end"), ("Notes", 1020, "start")]:
        s.text(x, ty, label, 13, MUTED, 700, anchor)
    for k, ((name, layers, taps, width, note), color) in enumerate(zip(LLM_TAPS, LLM_COLORS)):
        y = ty + 12 + k * 40
        if k % 2 == 0:
            s.rect(0, y, W, 36, fill="#f8fafc", rx=8)
        s.circle(14, y + 18, 6, fill=color)
        s.text(30, y + 24, name, 15, INK, 700)
        s.line(300, y + 18, 300 + layers * 5, y + 18, stroke="#cbd5e1", sw=6)
        for tap in taps:
            s.circle(300 + tap * 5, y + 18, 7, fill=color, stroke="#ffffff", sw=2)
        s.text(724, y + 24, ", ".join(map(str, taps)) + f" of {layers}", 14, INK)
        s.text(1000, y + 24, f"{width:,}", 14, INK, 700, "end")
        s.text(1020, y + 24, note, 13.5, MUTED)
    cy = ty + 12 + 4 * 40 + 10
    s.rect(0, cy, W, 40, fill=INK, rx=10)
    s.text(W / 2, cy + 26, "4 + 4 + 2 + 4 = 14 model-layer blocks   ×   4 positions   =   56 vectors per user",
           16.5, "#ffffff", 700, "middle")

    # 4. attention fusion
    y4 = cy + 92
    step(s, y4, 4, "Fuse the 56 vectors into 1024 features", "attention fusion, fit on each training fold")
    t, h, bw, gap = y4 + 24, 116, 178, 26
    mid = t + h / 2
    boxes = [
        ("One block", None, "4 vectors, z-scored"),
        ("Score each", "s = w · h", "one learned w per block"),
        ("Softmax", None, "4 weights that sum to 1"),
        ("Weighted sum", "z = Σ α · h", "one vector per block"),
        ("Concatenate", "14 blocks", "79,360 numbers"),
        ("Linear layer", "→ 1024", "features for STM and MTM"),
    ]
    for k, (title, formula, sub) in enumerate(boxes):
        x = k * (bw + gap)
        dark = k == len(boxes) - 1
        s.rect(x, t, bw, h, fill=INK if dark else "#ffffff", stroke=None if dark else EDGE, rx=12,
               shadow=not dark)
        s.text(x + bw / 2, t + 28, title, 15, "#ffffff" if dark else INK, 700, "middle")
        if k == 0:
            for j in range(4):
                for q in range(6):
                    s.rect(x + 46 + q * 15, t + 40 + j * 13, 12, 10, fill=POS_COLORS[j], rx=2)
        elif k == 2:
            for j, hh in enumerate([16, 38, 26, 10]):
                s.rect(x + 50 + j * 21, t + 84 - hh, 15, hh, fill=POS_COLORS[j], rx=3)
            s.line(x + 44, t + 84, x + 134, t + 84, stroke=EDGE, sw=1)
        else:
            s.text(x + bw / 2, t + 66, formula, 19 if dark else 17, "#5eead4" if dark else INK, 700, "middle")
        s.text(x + bw / 2, t + h - 14, sub, 12.5, "#cbd5e1" if dark else MUTED, anchor="middle")
        if k:
            s.line(x - gap + 3, mid, x - 3, mid, arrow=True, sw=2)
    note = ("Trained through a temporary linear probe that predicts high risk: weighted BCE (positive weight 1.5 × "
            "negatives / positives), full-batch Adam, learning rate 0.01, at most 500 epochs, keeping the epoch with "
            "the best development AUC (patience 50). The probe is then dropped. The weights pick among the 4 "
            "positions, not among tokens.")
    s.lines(0, t + h + 28, wrap(note, W, 13.5), 13.5, MUTED, leading=1.4)

    # 5. classifiers
    y5 = t + h + 104
    step(s, y5, 5, "Classify with two networks on the same 1024 features")
    t, h = y5 + 22, 418
    s.rect(0, t, 500, h, fill="#f8fafc", stroke=STM_C, sw=2, rx=16)
    s.text(20, t + 34, "STM", 22, STM_C, 700)
    s.text(76, t + 34, "single task, best setting per fold", 14, MUTED)
    fy = t + 54
    s.rect(20, fy, 120, 42, fill=INK, rx=10)
    s.text(80, fy + 27, "1024 features", 14, "#ffffff", 700, "middle")
    s.line(144, fy + 21, 160, fy + 21, arrow=True, sw=2)
    s.rect(164, fy, 150, 42, fill="#dbeafe", stroke=STM_C, sw=1.4, rx=10)
    s.text(239, fy + 27, "1–3 tanh layers", 14, INK, 700, "middle")
    s.line(318, fy + 21, 334, fy + 21, arrow=True, sw=2)
    s.rect(338, fy, 142, 42, fill=STM_C, rx=10)
    s.text(409, fy + 27, "P(high risk)", 14, "#ffffff", 700, "middle")
    s.text(409, fy + 62, "sigmoid of 1 logit, BCE", 12, MUTED, anchor="middle")
    hy = fy + 96
    for label, x in [("Fold", 20), ("Layers × units", 72), ("LR", 206), ("Epochs", 270), ("Test AUC", 350)]:
        s.text(x, hy, label, 12.5, MUTED, 700)
    s.line(20, hy + 8, 480, hy + 8, stroke=EDGE, sw=1)
    for i, ((layers, units, lr, ep), auc) in enumerate(zip(STM_SETTINGS, STM_FOLDS)):
        y = hy + 32 + i * 32
        s.text(34, y, str(i), 14.5, INK, 700, "middle")
        for q in range(layers):
            s.rect(72 + q * 11, y - 13, 7, 16, fill="#93c5fd", stroke=STM_C, sw=0.8, rx=2)
        s.text(110, y, f"{layers} × {units}", 14.5)
        s.text(206, y, f"{lr:g}", 14.5)
        s.text(270, y, f"{ep:,}", 14.5)
        bar(s, 350, y - 5, auc, STM_C, False, width=70)
    my = hy + 32 + 5 * 32 - 8
    s.line(20, my - 14, 480, my - 14, stroke=INK, sw=1)
    s.text(20, my + 6, "Mean", 14.5, INK, 700)
    s.text(110, my + 6, "a different setting on each fold", 13, MUTED)
    bar(s, 350, my + 1, sum(STM_FOLDS) / 5, STM_C, True, width=70)

    mx = 520
    s.rect(mx, t, W - mx, h, fill="#f8fafc", stroke=MTM_C, sw=2, rx=16)
    s.text(mx + 20, t + 34, "MTM", 22, MTM_C, 700)
    s.text(mx + 82, t + 34, "multi-task cascade, one setting for all folds", 14, MUTED)
    s.rect(mx + 20, fy, 130, 42, fill=INK, rx=10)
    s.text(mx + 85, fy + 27, "1024 features", 14, "#ffffff", 700, "middle")
    s.line(mx + 154, fy + 21, mx + 170, fy + 21, arrow=True, sw=2)
    s.rect(mx + 174, fy, 290, 42, fill="#334155", rx=10)
    s.text(mx + 319, fy + 27, "Shared layer: 512 tanh units", 14.5, "#ffffff", 700, "middle")
    stages = [("Personality", "5 scores", PERS_C, "MSE"), ("Psychosocial", "4 scores", PSY_C, "MSE"),
              ("Psychiatric", "2 scores", PSYCH_C, "MSE"), ("Suicide", "1 logit", MTM_C, "BCE")]
    sw_, sg = 146, 20
    sy = fy + 84
    centers = [mx + 20 + k * (sw_ + sg) + sw_ / 2 for k in range(4)]
    rail = fy + 60
    s.path(f"M{mx + 319},{fy + 42} V{rail}", stroke="#94a3b8", sw=2)
    s.line(centers[0], rail, centers[-1], rail, stroke="#94a3b8", sw=2)
    for k, ((name, sub, color, loss), c) in enumerate(zip(stages, centers)):
        x = c - sw_ / 2
        last = k == 3
        s.line(c, rail, c, sy - 4, stroke="#94a3b8", sw=2, arrow=True)
        s.rect(x, sy, sw_, 62, fill=color if last else "#ffffff", stroke=color, sw=1.6, rx=10)
        s.text(c, sy + 26, name, 14.5, "#ffffff" if last else color, 700, "middle")
        s.text(c, sy + 46, sub, 12.5, "#ccfbf1" if last else MUTED, anchor="middle")
        tag_fill, tag_text = (color, "#ffffff") if last else ("#ffffff", color)
        s.rect(c - 26, sy + 70, 52, 22, fill=tag_fill, stroke=color, sw=1.2, rx=11)
        s.text(c, sy + 86, loss, 11.5, tag_text, 700, "middle")
        if k:
            s.line(x - sg + 2, sy + 31, x - 3, sy + 31, arrow=True, sw=2)
    ly = sy + 112
    s.line(mx + 20, ly - 4, mx + 44, ly - 4, stroke="#94a3b8", sw=2, arrow=True)
    s.text(mx + 52, ly, "shared layer into every stage", 12.5, MUTED)
    s.line(mx + 250, ly - 4, mx + 274, ly - 4, sw=2, arrow=True)
    s.text(mx + 282, ly, "previous stage's prediction", 12.5, MUTED)
    oy = ly + 16
    s.line(centers[-1], sy + 94, centers[-1], oy - 4, arrow=True, sw=2)
    folds = v1_folds if v1_folds is not None else PHQ_FOLDS
    if v1_folds is not None:
        s.rect(mx + 20, oy, W - mx - 40, 44, fill=MTM_C, rx=12)
        s.text((mx + W) / 2, oy + 28, "P(high risk) from the suicide logit   ·   mean test AUC 0.738", 15.5,
               "#ffffff", 700, "middle")
        s.text(mx + 20, oy + 64, "150 questionnaire-only epochs, then joint training (shared layer at 1/5 the lr). "
               "Scores use a moving average of the weights.", 13, MUTED)
        s.text(mx + 20, oy + 88, "Setting: 1 × 512 shared, tanh, RMSprop lr 1e-4, 512-unit stages. 3 seeds × 5 best "
               "development-AUC epochs, averaged.", 13.5, INK)
        s.text(mx + 20, oy + 112, "Loss: BCE on high risk + 4 × MSE on each z-scored questionnaire score.", 13.5, INK)
        s.text(mx + 20, oy + 138, "Test AUC by fold: " + "  ·  ".join(f"{v:.3f}" for v in folds)
               + f"   mean {sum(folds) / 5:.3f}", 13.5, MTM_C, 700)
    else:
        s.rect(mx + 20, oy, W - mx - 40, 44, fill=PHQ_C, rx=12)
        s.text((mx + W) / 2, oy + 28, "score = suicide logit + 0.1 × predicted PHQ-9   →   P(high risk)", 15.5,
               "#ffffff", 700, "middle")
        s.text(mx + 20, oy + 64, "Predicted PHQ-9: ridge regression on the 1024 features, fit on the training fold.",
               13, MUTED)
        s.text(mx + 20, oy + 88, "Setting: 1 × 512 shared, tanh, learning rate 0.005, 1,000 epochs, 512-unit stages.",
               13.5, INK)
        s.text(mx + 20, oy + 112, "Loss: BCE on high risk + MSE on each z-scored questionnaire score.", 13.5, INK)
        s.text(mx + 20, oy + 138, "Test AUC by fold: " + "  ·  ".join(f"{v:.3f}" for v in folds)
               + f"   mean {sum(folds) / 5:.3f}", 13.5, PHQ_C, 700)

    # 6. protocol
    y6 = t + h + 52
    step(s, y6, 6, "Train, select, and test", "5 stratified folds; every user is a test user exactly once")
    t = y6 + 22
    n, x0, x1 = 1003, 56, 560
    scale = (x1 - x0) / n
    for i in range(5):
        y = t + i * 22
        s.text(0, y + 13, f"Fold {i}", 12.5, MUTED, 700)
        s.rect(x0, y, x1 - x0, 15, fill="#cbd5e1", rx=4)
        tx = x0 + i * 200.6 * scale
        s.rect(round(tx, 1), y, round(200.6 * scale, 1), 15, fill=MTM_C, rx=4)
        dx = tx + 200.6 * scale if i < 4 else x0
        s.rect(round(dx, 1), y, round(142 * scale, 1), 15, fill="#60a5fa", rx=4)
    ly = t + 5 * 22 + 14
    for k, (label, color) in enumerate([("train 660 (87 high risk)", "#cbd5e1"), ("dev 142 (19)", "#60a5fa"),
                                        ("test 200–201 (26–27)", MTM_C)]):
        lx = x0 + [0, 196, 316][k]
        s.rect(lx, ly - 11, 14, 14, fill=color, rx=3)
        s.text(lx + 20, ly, label, 12.5, INK)
    bullets(s, 600, t + 12, W - 600, [
        "504-setting grid: 1–3 layers, 16–1024 units, tanh or sigmoid, learning rate 0.001–0.05, "
        "1,000–5,000 epochs. RMSprop (momentum 0.9), batch 32, best development-AUC epoch kept (patience 200).",
        ("STM: best development AUC on each fold. MTM: aux-pretrained, then joint; pick by typical (median) "
         "development AUC, not a lucky epoch." if v1_folds is not None else
         "STM: best development AUC on each fold. MTM and its PHQ-9 weight: best mean development AUC over the folds."),
        "Test folds are scored once, after selection.",
    ], gap=6)
    s.h = ly + 14
    dest = out or (POSTER / "a0" / "method.svg")
    dest.parent.mkdir(exist_ok=True)
    s.save(dest)


if __name__ == "__main__":
    system()
    hyperparams()
    method()
