"""Zip-backed image store: read images from pics.zip without extracting.

pics.zip members are named like:
  pics/<PostId>.jpg
where PostId matches facebook_data.PostId (including the ' -id' suffix).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterator

from PIL import Image


class ImageStore:
    """Maps PostId (or filename stem) -> zip member path. Reads on demand."""

    def __init__(self, zip_path: Path):
        self.zip_path = Path(zip_path)
        self._zf: zipfile.ZipFile | None = None
        self._key_to_member: dict[str, str] = {}
        self._index_built = False

    @property
    def available(self) -> bool:
        if not self.zip_path.exists():
            return False
        try:
            with zipfile.ZipFile(self.zip_path) as zf:
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

    @staticmethod
    def normalize_key(key: str) -> str:
        # Strip extension if present; lowercase for safety but keep spaces
        k = str(key).strip()
        for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".JPG", ".JPEG", ".PNG"):
            if k.endswith(ext):
                k = k[: -len(ext)]
                break
        return k

    def _build_index(self) -> None:
        assert self._zf is not None
        for name in self._zf.namelist():
            if name.endswith("/"):
                continue
            stem = self.normalize_key(Path(name).name)
            self._key_to_member[stem] = name
        self._index_built = True

    def has(self, key: str) -> bool:
        self.open()
        return self.normalize_key(key) in self._key_to_member

    def members(self) -> list[str]:
        self.open()
        assert self._zf is not None
        return [n for n in self._zf.namelist() if not n.endswith("/")]

    def keys(self) -> list[str]:
        self.open()
        return sorted(self._key_to_member.keys())

    def load_pil(self, key: str) -> Image.Image | None:
        self.open()
        assert self._zf is not None
        member = self._key_to_member.get(self.normalize_key(key))
        if member is None:
            return None
        with self._zf.open(member) as f:
            data = f.read()
        try:
            img = Image.open(io.BytesIO(data))
            return img.convert("RGB")
        except Exception:
            return None

    def iter_keys(self, keys: list[str]) -> Iterator[tuple[str, Image.Image]]:
        for k in keys:
            img = self.load_pil(k)
            if img is not None:
                yield k, img
