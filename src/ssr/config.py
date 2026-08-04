"""Typed config loader for SSR pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    root: Path
    user_csv: Path
    posts_csv: Path
    pics_zip: Path
    artifacts: Path
    reports: Path

    def resolve(self, root: Path) -> "PathsConfig":
        def _p(p: Path | str) -> Path:
            pp = Path(p)
            return pp if pp.is_absolute() else (root / pp)

        return PathsConfig(
            root=root,
            user_csv=_p(self.user_csv),
            posts_csv=_p(self.posts_csv),
            pics_zip=_p(self.pics_zip),
            artifacts=_p(self.artifacts),
            reports=_p(self.reports),
        )


@dataclass
class Config:
    raw: dict[str, Any]
    seed: int
    experiment: str
    paths: PathsConfig
    cohort: dict[str, Any]
    posts: dict[str, Any]
    images: dict[str, Any]
    corpus: dict[str, Any]
    represent: dict[str, Any]
    fusion: dict[str, Any]
    train: dict[str, Any]

    def art(self, *parts: str) -> Path:
        p = self.paths.artifacts.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def report(self, *parts: str) -> Path:
        p = self.paths.reports.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def exp_dir(self, *parts: str) -> Path:
        return self.art(self.experiment, *parts)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    root = Path(raw["paths"].get("root", ".")).resolve()
    if not root.is_absolute():
        root = path.parent.parent.resolve()

    paths = PathsConfig(
        root=root,
        user_csv=Path(raw["paths"]["user_csv"]),
        posts_csv=Path(raw["paths"]["posts_csv"]),
        pics_zip=Path(raw["paths"]["pics_zip"]),
        artifacts=Path(raw["paths"]["artifacts"]),
        reports=Path(raw["paths"]["reports"]),
    ).resolve(root)

    paths.artifacts.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)

    return Config(
        raw=raw,
        seed=int(raw.get("seed", 42)),
        experiment=str(raw.get("experiment", "standard")),
        paths=paths,
        cohort=raw["cohort"],
        posts=raw.get("posts", {}),
        images=raw.get("images", {}),
        corpus=raw.get("corpus", {}),
        represent=raw.get("represent", {}),
        fusion=raw["fusion"],
        train=raw["train"],
    )
