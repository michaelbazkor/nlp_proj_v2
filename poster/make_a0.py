"""Assemble the A0 landscape poster as an editable PowerPoint file.

python poster/make_panels.py a0      draw the poster-sized charts into poster/a0/
python poster/make_diagrams.py       draw poster/a0/method.svg
python poster/export_png.py a0       PNG fallbacks for every poster/a0/*.svg
python poster/make_a0.py             build poster/a0/poster_A0.pptx, then PDF and PNG previews

Text boxes, cards, and the table are native PowerPoint shapes. Figures are SVG with a PNG fallback;
right-click a figure and choose "Convert to Shape" to edit it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.opc.package import Part
from pptx.oxml.ns import qn
from pptx.util import Cm, Pt

POSTER = Path(__file__).resolve().parent
A0 = POSTER / "a0"
PNG = A0 / "png"
OUT = A0 / "poster_A0.pptx"
PAPER_DOI = "https://doi.org/10.1038/s41598-020-73917-0"
SVG_EXT = "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"
SVG_NS = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"

NAVY = "13294B"
INK = "1E293B"
MUTED = "64748B"
TEAL = "0F766E"
STM = "1D4ED8"
PHQ = "B45309"
WRONG = "E11D48"
PAGE = "EEF2F7"
FONT = "Arial"

W, H = 118.9, 84.1
MARGIN, GAP = 1.4, 1.2
COL_L = (MARGIN, 32.0)
COL_M = (MARGIN + 32.0 + GAP, 41.6)
COL_R = (COL_M[0] + COL_M[1] + GAP, W - MARGIN - (COL_M[0] + COL_M[1] + GAP))
BODY_TOP, BODY_BOTTOM = 12.3, H - 1.2
PAD = 1.0


def rgb(hex_: str) -> RGBColor:
    return RGBColor.from_string(hex_)


def box(slide, x, y, w, h, fill="FFFFFF", line=None, radius=0.6, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Cm(x), Cm(y), Cm(w), Cm(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = min(0.5, radius / min(w, h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(1.5)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def bullet(p, char="•", color=INK, indent=0.85):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Cm(indent))))
    pPr.set("indent", str(-int(Cm(indent))))
    for tag in ("a:buClr", "a:buFont", "a:buChar", "a:buNone"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    clr = etree.SubElement(pPr, qn("a:buClr"))
    etree.SubElement(clr, qn("a:srgbClr")).set("val", color)
    etree.SubElement(pPr, qn("a:buFont")).set("typeface", FONT)
    etree.SubElement(pPr, qn("a:buChar")).set("char", char)


def text(slide, x, y, w, h, paragraphs, anchor=MSO_ANCHOR.TOP, autofit=False):
    """paragraphs: list of dicts with runs=[(text, {size, bold, color, italic})], align, bullet, space_after."""
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = para.get("align", PP_ALIGN.LEFT)
        p.line_spacing = para.get("line_spacing", 1.08)
        p.space_after = Pt(para.get("space_after", 0))
        p.space_before = Pt(para.get("space_before", 0))
        if para.get("bullet"):
            bullet(p, para.get("char", "•"), para.get("bullet_color", INK), para.get("indent", 0.85))
        for run_text, style in para["runs"]:
            r = p.add_run()
            r.text = run_text
            f = r.font
            f.name = FONT
            f.size = Pt(style.get("size", 24))
            f.bold = style.get("bold", False)
            f.italic = style.get("italic", False)
            f.color.rgb = rgb(style.get("color", INK))
    return tb


def para(s, size=24, color=INK, bold=False, italic=False, **kw):
    return {"runs": [(s, {"size": size, "color": color, "bold": bold, "italic": italic})], **kw}


def rich(parts, size=24, **kw):
    """parts: list of (text, bold) or (text, bold, color)."""
    runs = []
    for part in parts:
        t, b = part[0], part[1]
        c = part[2] if len(part) > 2 else INK
        runs.append((t, {"size": size, "bold": b, "color": c}))
    return {"runs": runs, **kw}


def bullets(items, size=24, color=TEAL, space_after=10, **kw):
    out = []
    for item in items:
        parts = item if isinstance(item, list) else [(item, False)]
        out.append(rich(parts, size, bullet=True, bullet_color=color, space_after=space_after, **kw))
    return out


def picture(slide, name, x, y, w=None, h=None):
    """Place poster/a0/<name>.svg with its PNG fallback; give w or h in cm."""
    png, svg = PNG / f"{name}.png", A0 / f"{name}.svg"
    kw = {"width": Cm(w)} if w else {"height": Cm(h)}
    pic = slide.shapes.add_picture(str(png), Cm(x), Cm(y), **kw)
    if svg.exists():
        attach_svg(slide, pic, svg.read_bytes())
    return pic


def attach_svg(slide, pic, blob: bytes) -> None:
    pkg = slide.part.package
    part = Part(pkg.next_image_partname("svg"), "image/svg+xml", pkg, blob)
    rid = slide.part.relate_to(part, RT.IMAGE)
    blip = pic._element.xpath("./p:blipFill/a:blip")[0]
    ext_lst = blip.find(qn("a:extLst"))
    if ext_lst is None:
        ext_lst = etree.SubElement(blip, qn("a:extLst"))
    ext = etree.SubElement(ext_lst, qn("a:ext"))
    ext.set("uri", SVG_EXT)
    el = etree.SubElement(ext, f"{{{SVG_NS}}}svgBlip", nsmap={"asvg": SVG_NS})
    el.set(qn("r:embed"), rid)


def card(slide, col, y, h, title, accent=TEAL, fill="FFFFFF"):
    x, w = col
    box(slide, x, y, w, h, fill=fill, line="D5DDE8")
    box(slide, x + PAD, y + 0.95, 0.35, 1.55, fill=accent, radius=0.15)
    text(slide, x + PAD + 0.75, y + 0.7, w - 2 * PAD - 0.75, 2.0,
         [para(title, 40, NAVY, True)], anchor=MSO_ANCHOR.MIDDLE)
    return y + 3.2


def size_of(name: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(PNG / f"{name}.png") as im:
        return im.size


def fit_h(name: str, w: float) -> float:
    pw, ph = size_of(name)
    return w * ph / pw


def prepare_assets() -> None:
    import numpy as np
    import segno
    from PIL import Image, ImageFilter

    PNG.mkdir(parents=True, exist_ok=True)
    qr = segno.make(PAPER_DOI, error="m")
    qr.save(str(A0 / "paper_qr.svg"), scale=10, border=0, dark="#" + NAVY)
    qr.save(str(PNG / "paper_qr.png"), scale=40, border=0, dark="#" + NAVY)

    logo = A0 / "technion_logo.png"
    source = A0 / "technion_logo_source.png"
    if source.exists() and not logo.exists():
        im = Image.open(source).convert("RGBA")
        alpha = im.getchannel("A").resize((im.width * 3, im.height * 3), Image.LANCZOS)
        alpha = alpha.filter(ImageFilter.GaussianBlur(1.2)).point(lambda a: max(0, min(255, (a - 128) * 3 + 128)))
        px = np.asarray(im).reshape(-1, 4)
        color = tuple(int(v) for v in px[px[:, 3].argmax(), :3])
        out = Image.new("RGBA", alpha.size, color + (0,))
        out.putalpha(alpha)
        out.save(logo)


def header(slide) -> None:
    box(slide, 0, 0, W, 11.2, fill=NAVY, shape=MSO_SHAPE.RECTANGLE)
    box(slide, 0, 11.2, W, 0.35, fill=TEAL, shape=MSO_SHAPE.RECTANGLE)
    text(slide, 3, 1.1, W - 6, 9.4, [
        para("Reading Between the Posts: Suicide-Risk Signals in LLM Hidden States", 74, "FFFFFF", True,
             align=PP_ALIGN.CENTER, space_after=10),
        para("Revisiting the single- and multi-task networks of Ophir et al. (2020) with four open-weight LLMs",
             36, "CBD5E1", align=PP_ALIGN.CENTER, space_after=22),
        para("Michael Bazkor  ·  Eliyahu Sushansky  ·  Nitzan Yehezkel-Agam", 42, "FFFFFF", True,
             align=PP_ALIGN.CENTER, space_after=8),
        para("Technion – Israel Institute of Technology  ·  Course project", 30, "94A3B8", align=PP_ALIGN.CENTER),
    ], anchor=MSO_ANCHOR.MIDDLE)


def left_column(slide) -> None:
    x, w = COL_L
    iw = w - 2 * PAD
    y = BODY_TOP

    h = 10.5
    top = card(slide, COL_L, y, h, "TL;DR", accent=TEAL, fill="ECFDF5")
    text(slide, x + PAD, top, iw, h - (top - y) - 0.6, bullets([
        [("We swap the paper's ELMo embeddings for ", False),
         ("hidden states of four open LLMs", True), (" that read all of a user's posts at once.", False)],
        [("Same 1,003 users, same high-risk label: test AUC rises from ", False), ("0.629 to 0.716", True, STM),
         (" (single-task) and from ", False), ("0.697 to 0.734", True, PHQ), (" (multi-task + PHQ-9).", False)],
        [("Research only, not a screening tool: ", True), ("yes/no calls stay weak.", False)],
    ], size=25, space_after=10))
    y += h + GAP

    h = 11.0
    top = card(slide, COL_L, y, h, "Background")
    text(slide, x + PAD, top, iw, h - (top - y) - 0.6, bullets([
        "Suicide is a leading cause of death worldwide. Five decades of research on demographic, psychological, "
        "and medical risk factors predict it only slightly better than chance (AUC 0.56–0.58).",
        "Many people at risk never see a clinician, but they do post. Everyday social-media language may carry "
        "signals that clinic-based assessments miss.",
        [("LLMs hold rich meaning in their hidden layers. Asked directly, our four LLMs' own RISK = 1/0 answers "
          "scored near chance (AUC 0.52–0.58), ", False), ("so we read their hidden states instead.", True)],
    ], size=22, space_after=8))
    y += h + GAP

    h = 21.2
    top = card(slide, COL_L, y, h, "Related work: Ophir et al. (2020)")
    text(slide, x + PAD, top - 0.45, iw, 1.2,
         [para("“Deep neural networks detect suicide risk from textual Facebook posts”, Scientific Reports 10, 16685",
               18, MUTED, italic=True)])
    body = [para("Data", 24, TEAL, True, space_after=3, space_before=4)]
    body += bullets([
        "1,002 adults recruited on Amazon MTurk with authenticated Facebook accounts: 83,292 posts from up to "
        "12 months, 10 or more posts per user.",
        "Suicide risk from the Columbia Suicide Severity Rating Scale. General risk (any ideation): 361 users, 36%. "
        "High risk (a method, intention, or plan): 132 users, 13%.",
        "Questionnaires: depression (PHQ-9), anxiety (GAD-7), brooding, worry, loneliness, life satisfaction, "
        "Big Five personality.",
    ], size=20, space_after=5)
    body += [para("Model", 24, TEAL, True, space_after=3, space_before=6)]
    body += bullets([
        "Text only. ELMo word embeddings averaged per post, then per user: 1,024 numbers.",
        "STM: fully connected layers predict risk. MTM: personality → psychosocial → psychiatric → suicide, "
        "every stage fed by a shared layer.",
        "Five random 70/15/15 train/dev/test splits.",
    ], size=20, space_after=5)
    text(slide, x + PAD, top + 0.8, iw, 11.5, body)
    table_y = y + h - 0.8 - 3.45
    rows = [("Test AUC [95% CI]", "STM", "MTM"),
            ("General risk", "0.621 [0.576, 0.657]", "0.746 [0.727, 0.765]"),
            ("High risk (our target)", "0.629 [0.606, 0.660]", "0.697 [0.690, 0.707]")]
    shape = slide.shapes.add_table(3, 3, Cm(x + PAD), Cm(table_y), Cm(iw), Cm(3.45))
    shape._element.graphic.graphicData.tbl.tblPr.set("bandRow", "0")
    table = shape.table
    for j, cw_ in enumerate([10.6, 9.7, 9.7]):
        table.columns[j].width = Cm(cw_)
    for i, row in enumerate(rows):
        table.rows[i].height = Cm(1.15)
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(NAVY if i == 0 else ("FEF3C7" if i == 2 else "F8FAFC"))
            cell.margin_left = cell.margin_right = Cm(0.3)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.text = ""
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = value
            r.font.name = FONT
            r.font.size = Pt(19)
            r.font.bold = i == 0 or j == 0 or (i == 2 and j == 2)
            r.font.color.rgb = rgb("FFFFFF" if i == 0 else INK)
    y += h + GAP

    foot_h = 6.0
    h = BODY_BOTTOM - foot_h - GAP - y
    top = card(slide, COL_L, y, h, "Our cohort")
    text(slide, x + PAD, top, iw, 2.2, bullets([
        "Same users and filters as the paper: 1,003 labeled users, 132 high risk (13.2%). We add photo captions; "
        "the paper used text only.",
    ], size=21, space_after=0))
    cap_y = y + h - 1.75
    pw, ph = size_of("corr_bars")
    ch = cap_y - (top + 2.3) - 0.2
    cw_ = min(iw, ch * pw / ph)
    picture(slide, "corr_bars", x + PAD + (iw - cw_) / 2, top + 2.3, w=cw_)
    text(slide, x + PAD, cap_y, iw, 1.4, [para(
        "Pearson r with the 0–6 suicide score, matching the paper's Table 1. Life satisfaction is left out: "
        "its column in our file repeats extraversion.", 17, MUTED, italic=True)])

    fy = BODY_BOTTOM - foot_h
    box(slide, x, fy, w, foot_h, fill="FFFFFF", line="D5DDE8")
    logo = A0 / "technion_logo.png"
    if logo.exists():
        slide.shapes.add_picture(str(logo), Cm(x + PAD), Cm(fy + 1.4), height=Cm(3.2))
    q = foot_h - 1.4
    qx = x + w - PAD - q
    picture(slide, "paper_qr", qx, fy + 0.7, w=q)
    text(slide, qx - 9.6, fy + 0.9, 9.1, q, [
        para("Original paper", 22, NAVY, True, align=PP_ALIGN.RIGHT, space_after=4),
        para("Ophir et al., Sci. Rep. 2020", 19, INK, align=PP_ALIGN.RIGHT, space_after=4),
        para("doi:10.1038/s41598-020-73917-0", 16, MUTED, align=PP_ALIGN.RIGHT, space_after=4),
        para("scan →", 19, TEAL, True, align=PP_ALIGN.RIGHT),
    ], anchor=MSO_ANCHOR.MIDDLE)


def middle_column(slide) -> None:
    x, w = COL_M
    y, h = BODY_TOP, BODY_BOTTOM - BODY_TOP
    top = card(slide, COL_M, y, h, "Method: from a user's posts to a risk score")
    avail_h = y + h - top - 0.7
    iw = w - 2 * PAD
    fw = min(iw, avail_h * size_of("method")[0] / size_of("method")[1])
    picture(slide, "method", x + (w - fw) / 2, top + 0.1, w=fw)


def right_column(slide) -> None:
    x, w = COL_R
    iw = w - 2 * PAD
    y = BODY_TOP
    lim_h = 13.4
    h = BODY_BOTTOM - y - lim_h - GAP
    top = card(slide, COL_R, y, h, "Results: high suicide risk on held-out users")
    gap, cap, cap_h = 0.8, 17, 2.1

    roc_w0, roc_h0 = size_of("roc")
    ha = 16.4
    roc_w = ha * roc_w0 / roc_h0
    head_w = iw - gap - roc_w
    picture(slide, "headline", x + PAD, top, w=head_w)
    picture(slide, "roc", x + PAD + head_w + gap, top, h=ha)
    cy = top + max(ha, fit_h("headline", head_w)) + 0.15
    text(slide, x + PAD, cy, head_w, cap_h, [para(
        "Whiskers: 95% CI (paper as reported; ours across the 5 folds). White numbers: gain over the paper's "
        "model of the same type. d = Cohen's d. Dashed line: the paper's best.", cap, MUTED, italic=True)])
    text(slide, x + PAD + head_w + gap, cy, roc_w, cap_h, [para(
        "ROC on the 5 test folds: thin lines are folds, thick lines and bands the mean ± SD.",
        cap, MUTED, italic=True)])
    half = (iw - gap) / 2
    y2 = cy + cap_h + 0.3
    hb = max(fit_h("errors_by_score", half), fit_h("aux_scores", half))
    picture(slide, "errors_by_score", x + PAD, y2, w=half)
    picture(slide, "aux_scores", x + PAD + half + gap, y2, w=half)
    cy = y2 + hb + 0.15
    text(slide, x + PAD, cy, half, cap_h, [para(
        "Calls at each fold's development-F1 threshold. Accuracy is 77%, below the 87% of always answering "
        "“not high risk”.", cap, MUTED, italic=True)])
    text(slide, x + PAD + half + gap, cy, half, cap_h, [para(
        "The MTM's middle stages barely learn the questionnaires (dots = folds), yet the cascade still lifts "
        "the suicide output.", cap, MUTED, italic=True)])
    y3 = cy + cap_h + 0.3
    ex_h = y + h - y3 - 1.9
    ex_w = min(iw, ex_h * size_of("examples")[0] / size_of("examples")[1])
    picture(slide, "examples", x + PAD + (iw - ex_w) / 2, y3, w=ex_w)
    text(slide, x + PAD, y3 + fit_h("examples", ex_w) + 0.15, iw, 1.2, [para(
        "One held-out user per cell, picked for the most extreme score. Quotes are verbatim excerpts from "
        "their posts; names removed.", cap, MUTED, italic=True, align=PP_ALIGN.CENTER)])

    ly = BODY_BOTTOM - lim_h
    top = card(slide, COL_R, ly, lim_h, "Limitations & future work", accent=WRONG)
    half_t = (iw - 1.2) / 2
    text(slide, x + PAD, top, half_t, lim_h - 3.6, [para("Limitations", 24, WRONG, True, space_after=6)] + bullets([
        "Not a screening tool: 77% accuracy vs 87% for calling everyone low risk.",
        "About 19 high-risk users per development set, so model selection is noisy.",
        "Middle MTM stages barely predict the questionnaires (mean r ≤ 0.13).",
        "Life-satisfaction column repeats extraversion; MTurk sample, self-reported labels.",
    ], size=20, color=WRONG, space_after=5))
    text(slide, x + PAD + half_t + 1.2, top, half_t, lim_h - 3.6,
         [para("Future work", 24, TEAL, True, space_after=6)] + bullets([
             "Model general risk (any ideation) to compare with the paper's 0.746.",
             "Fine-tune an LLM end-to-end (e.g. LoRA) instead of reading frozen states.",
             "Attention over posts to show which posts drive a user's score.",
             "Nested cross-validation, calibrated thresholds, and a second cohort.",
         ], size=20, color=TEAL, space_after=5))


def build() -> None:
    prepare_assets()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Cm(W), Cm(H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(PAGE)
    header(slide)
    left_column(slide)
    middle_column(slide)
    right_column(slide)
    prs.save(OUT)
    print("wrote", OUT)


def export() -> None:
    pdf, png = OUT.with_suffix(".pdf"), OUT.with_name("poster_A0_preview.png")
    ps = (
        "$ppt = New-Object -ComObject PowerPoint.Application; "
        f"$p = $ppt.Presentations.Open('{OUT}', $true, $false, $false); "
        f"$p.SaveAs('{pdf}', 32); "
        f"$p.Slides(1).Export('{png}', 'PNG', 5616, 3972); "
        "$p.Close(); $ppt.Quit()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    print("wrote", pdf, "and", png)


if __name__ == "__main__":
    build()
    if "--no-export" not in sys.argv:
        export()
