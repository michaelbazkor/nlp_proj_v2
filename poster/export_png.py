"""Render poster SVGs to PNG with headless Chrome.

python poster/export_png.py        poster/NN_*.svg  -> poster/png/ at 2x
python poster/export_png.py a0     poster/a0/*.svg  -> poster/a0/png/ at 4x
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from pathlib import Path

POSTER = Path(__file__).resolve().parent
CHROME = [
    Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
    Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe",
]


def size_px(svg: Path) -> tuple[int, int]:
    head = svg.read_text(encoding="utf-8")[:2000]
    m = re.search(r'<svg[^>]*?width="([\d.]+)(pt|px)?"[^>]*?height="([\d.]+)(pt|px)?"', head, re.S)
    scale = 4 / 3 if m.group(2) == "pt" else 1.0
    return math.ceil(float(m.group(1)) * scale), math.ceil(float(m.group(3)) * scale)


def render(svgs: list[Path], out: Path, scale: int) -> None:
    browser = next((p for p in CHROME if p.exists()), None)
    if browser is None:
        sys.exit("Chrome or Edge not found")
    out.mkdir(exist_ok=True)
    for svg in svgs:
        w, h = size_px(svg)
        page = out / f"_{svg.stem}.html"
        page.write_text(
            f'<html><body style="margin:0;background:#fff"><img src="{svg.as_uri()}" width="{w}" height="{h}"></body></html>',
            encoding="utf-8",
        )
        png = out / f"{svg.stem}.png"
        subprocess.run(
            [str(browser), "--headless", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
             f"--force-device-scale-factor={scale}", f"--window-size={w},{h}", f"--screenshot={png}", page.as_uri()],
            check=True, capture_output=True,
        )
        page.unlink()
        print(png.name, w * scale, "x", h * scale)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "a0":
        render(sorted((POSTER / "a0").glob("*.svg")), POSTER / "a0" / "png", 4)
    else:
        render(sorted(POSTER.glob("[0-9][0-9]_*.svg")), POSTER / "png", 2)


if __name__ == "__main__":
    main()
