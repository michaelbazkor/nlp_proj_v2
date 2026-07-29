"""Zip-backed image store: read images from pics.zip without extracting."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterator

from PIL import Image


class ImageStore:
    """Maps blob GUIDs -> zip member paths. Reads images on demand."""

    def __init__(self, zip_path: Path):
        self.zip_path = Path(zip_path)
        self._zf: zipfile.ZipFile | None = None
        self._guid_to_member: dict[str, str] = {}
        self._index_built = False

    @property
    def available(self) -> bool:
        if not self.zip_path.exists():
            return False
        try:
            with zipfile.ZipFile(self.zip_path) as zf:
                # Need a valid central directory
                _ = zf.namelist()
            return True
        except zipfile.BadZipFile:
            return False

    def open(self) -> "ImageStore":
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path, "r")
            self._build_index()
        return self

    def close(self) -> None:
        if self._zf is not None:
            self._zf.close()
            self._zf = None

    def __enter__(self) -> "ImageStore":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def _build_index(self) -> None:
        assert self._zf is not None
        for name in self._zf.namelist():
            if name.endswith("/"):
                continue
            stem = Path(name).stem.lower()
            # Prefer exact GUID-looking names; also index basename
            self._guid_to_member[stem] = name
            # Some zips nest as pics/<guid>.jpg
            parts = Path(name).parts
            for p in parts:
                pstem = Path(p).stem.lower()
                if len(pstem) >= 32:
                    self._guid_to_member.setdefault(pstem, name)
        self._index_built = True

    def has(self, guid: str) -> bool:
        self.open()
        return guid.lower() in self._guid_to_member

    def members(self) -> list[str]:
        self.open()
        assert self._zf is not None
        return [n for n in self._zf.namelist() if not n.endswith("/")]

    def guids(self) -> list[str]:
        self.open()
        return sorted(self._guid_to_member.keys())

    def load_pil(self, guid: str) -> Image.Image | None:
        self.open()
        assert self._zf is not None
        member = self._guid_to_member.get(guid.lower())
        if member is None:
            return None
        with self._zf.open(member) as f:
            data = f.read()
        try:
            img = Image.open(io.BytesIO(data))
            return img.convert("RGB")
        except Exception:
            return None

    def iter_guids(self, guids: list[str]) -> Iterator[tuple[str, Image.Image]]:
        for g in guids:
            img = self.load_pil(g)
            if img is not None:
                yield g, img
