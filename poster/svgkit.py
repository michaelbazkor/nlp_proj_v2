"""Small helpers for hand-laid poster SVGs."""
from __future__ import annotations

import textwrap
from pathlib import Path

INK = "#1e293b"
MUTED = "#64748b"
LINE = "#e2e8f0"
EDGE = "#cbd5e1"
PAPER_C = "#94a3b8"
STM_C = "#1d4ed8"
MTM_C = "#0f766e"
PHQ_C = "#b45309"
RIGHT_C = "#0f766e"
WRONG_C = "#e11d48"
PERS_C = "#7c3aed"
PSY_C = "#0891b2"
PSYCH_C = "#b45309"
FONT = "Arial, Helvetica, sans-serif"


def esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text_width(s: str, size: float, bold: bool = False) -> float:
    return len(s) * size * (0.56 if bold else 0.51)


def wrap(s: str, width_px: float, size: float) -> list[str]:
    return textwrap.wrap(s, max(8, int(width_px / (size * 0.51))))


class Svg:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.body: list[str] = []
        self.markers: dict[str, str] = {}

    def _marker(self, color: str) -> str:
        if color not in self.markers:
            self.markers[color] = f"arrow{len(self.markers)}"
        return self.markers[color]

    def rect(self, x, y, w, h, fill="none", stroke=None, sw=1.5, rx=10, dash=None, opacity=None, shadow=False):
        a = [f'x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"']
        if stroke:
            a.append(f'stroke="{stroke}" stroke-width="{sw}"')
        if dash:
            a.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            a.append(f'opacity="{opacity}"')
        if shadow:
            a.append('filter="url(#shadow)"')
        self.body.append(f"<rect {' '.join(a)}/>")

    def circle(self, cx, cy, r, fill="none", stroke=None, sw=1.5, dash=None, opacity=None):
        a = [f'cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"']
        if stroke:
            a.append(f'stroke="{stroke}" stroke-width="{sw}"')
        if dash:
            a.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            a.append(f'opacity="{opacity}"')
        self.body.append(f"<circle {' '.join(a)}/>")

    def line(self, x1, y1, x2, y2, stroke=INK, sw=1.5, dash=None, arrow=False, opacity=None):
        a = [f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"']
        if dash:
            a.append(f'stroke-dasharray="{dash}"')
        if arrow:
            a.append(f'marker-end="url(#{self._marker(stroke)})"')
        if opacity is not None:
            a.append(f'opacity="{opacity}"')
        self.body.append(f"<line {' '.join(a)}/>")

    def path(self, d, stroke=INK, sw=1.5, fill="none", dash=None, arrow=False):
        a = [f'd="{d}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}" stroke-linejoin="round" stroke-linecap="round"']
        if dash:
            a.append(f'stroke-dasharray="{dash}"')
        if arrow:
            a.append(f'marker-end="url(#{self._marker(stroke)})"')
        self.body.append(f"<path {' '.join(a)}/>")

    def text(self, x, y, s, size=14, fill=INK, weight=400, anchor="start", italic=False):
        a = [f'x="{x}" y="{y}" font-size="{size}" fill="{fill}"']
        if weight != 400:
            a.append(f'font-weight="{weight}"')
        if anchor != "start":
            a.append(f'text-anchor="{anchor}"')
        if italic:
            a.append('font-style="italic"')
        self.body.append(f"<text {' '.join(a)}>{esc(s)}</text>")

    def lines(self, x, y, rows, size=14, fill=INK, weight=400, anchor="start", italic=False, leading=1.4):
        for i, row in enumerate(rows):
            self.text(x, y + i * size * leading, row, size, fill, weight, anchor, italic)
        return y + len(rows) * size * leading

    def raw(self, s: str):
        self.body.append(s)

    def save(self, path: Path):
        defs = [
            '<filter id="shadow" x="-10%" y="-10%" width="120%" height="135%">'
            '<feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.10"/></filter>'
        ]
        for color, mid in self.markers.items():
            defs.append(
                f'<marker id="{mid}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>'
            )
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}" font-family="{FONT}">'
        )
        out = [head, f"<defs>{''.join(defs)}</defs>", f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>']
        out.extend(self.body)
        out.append("</svg>")
        Path(path).write_text("\n".join(out), encoding="utf-8")
