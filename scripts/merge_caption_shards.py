"""Fold per-GPU caption shards into the model-scoped cache the pipeline reads."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssr.caption.run import caption_paths  # noqa: E402
from ssr.config import load_config  # noqa: E402
from ssr.io_utils import atomic_write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out, meta_path = caption_paths(cfg)
    shard_dir = out.parent / "shards"
    frames = []
    if out.exists() and out.stat().st_size:
        frames.append(pd.read_parquet(out))
    for p in sorted(shard_dir.glob("shard_*.parquet")):
        df = pd.read_parquet(p)
        frames.append(df)
        print(f"  {p.name}: {len(df)} captions")
    if not frames:
        raise SystemExit(f"no shards found in {shard_dir}")

    merged = pd.concat(frames, ignore_index=True).drop_duplicates("image_key", keep="last")
    merged = merged[merged["caption"].astype(str).str.len() > 0]
    merged.to_parquet(out, index=False)
    atomic_write_json(
        meta_path,
        {
            "model": cfg.images["caption_model"],
            "n_cached": int(len(merged)),
            "n_requested": int(len(merged)),
            "n_new": int(len(merged)),
            "zip_ready": True,
            "source": "scripts/caption_shard.py (batched, 4-GPU sharded)",
        },
    )
    lens = merged["caption"].str.len()
    print(
        f"merged -> {out}\n  captions={len(merged)} "
        f"mean_chars={lens.mean():.0f} p10={lens.quantile(.1):.0f} p90={lens.quantile(.9):.0f}"
    )


if __name__ == "__main__":
    main()
