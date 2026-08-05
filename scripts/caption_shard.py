"""Batched, shardable VLM captioning for the full-cohort run.

The stock `ssr.caption.run` captions one image per forward pass, which is far too
slow for ~300k images. This driver batches images and processes one shard per GPU,
writing resumable partial parquets that `merge_caption_shards.py` folds into the
model-scoped cache.

Usage (one process per GPU):
    CUDA_VISIBLE_DEVICES=0 python scripts/caption_shard.py \
        --config configs/real_full.yaml --shard 0 --num-shards 4 --batch-size 16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssr.caption.run import _model_slug  # noqa: E402
from ssr.config import load_config  # noqa: E402
from ssr.data.cohort import build_cohort  # noqa: E402
from ssr.data.images import ImageStore  # noqa: E402


def shard_out_path(cfg, shard: int) -> Path:
    slug = _model_slug(cfg.images["caption_model"])
    root = cfg.art("captions", slug, "shards")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"shard_{shard:02d}.parquet"


def load_done(path: Path) -> dict[str, str]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    df = pd.read_parquet(path)
    return dict(zip(df["image_key"], df["caption"]))


def build_model(model_id: str, dtype, max_pixels: int):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
    # Decoder-only generation needs left padding for correct batched output.
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, device_map={"": 0}, attn_implementation="sdpa"
    )

    model.eval()
    return processor, model


@torch.inference_mode()
def caption_batch(processor, model, images, prompt: str, max_new_tokens: int) -> list[str]:
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[text] * len(images), images=images, return_tensors="pt", padding=True
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return [
        c.strip()
        for c in processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--flush-every", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap images processed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cohort = build_cohort(cfg, assert_paper=False, force=False)
    posts = pd.read_parquet(cfg.art("posts.parquet"))
    posts = posts[posts["UserId"].isin(set(cohort["UserId"]))]

    keys = sorted(posts["image_key"].dropna().unique().tolist())
    mine = [k for i, k in enumerate(keys) if i % args.num_shards == args.shard]

    out_path = shard_out_path(cfg, args.shard)
    done = load_done(out_path)
    # Skip anything already captioned by an earlier full/partial run.
    slug = _model_slug(cfg.images["caption_model"])
    main_cache = cfg.art("captions", slug, "captions.parquet")
    if main_cache.exists() and main_cache.stat().st_size:
        prev = pd.read_parquet(main_cache)
        done.update(dict(zip(prev["image_key"], prev["caption"])))
    todo = [k for k in mine if k not in done]

    print(
        f"[shard {args.shard}] total_keys={len(keys)} mine={len(mine)} "
        f"cached={len(mine) - len(todo)} todo={len(todo)}",
        flush=True,
    )
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print(f"[shard {args.shard}] nothing to do", flush=True)
        return

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.images.get("dtype", "bfloat16")]
    prompt = cfg.images["caption_prompt"].strip()
    max_new = int(cfg.images.get("max_new_tokens", 128))

    print(f"[shard {args.shard}] loading {cfg.images['caption_model']} ...", flush=True)
    processor, model = build_model(cfg.images["caption_model"], dtype, args.max_pixels)

    rows: list[dict] = [{"image_key": k, "caption": v} for k, v in done.items() if k in set(mine)]
    store = ImageStore(cfg.paths.pics_zip)
    t0 = time.time()
    n_done = 0
    n_missing = 0
    pending: list[dict] = []

    def flush() -> None:
        if not pending:
            return
        rows.extend(pending)
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        pending.clear()

    with store:
        batch_keys: list[str] = []
        batch_imgs: list = []
        for key in todo:
            img = store.load_pil(key)
            if img is None:
                n_missing += 1
                continue
            batch_keys.append(key)
            batch_imgs.append(img)
            if len(batch_keys) < args.batch_size:
                continue
            try:
                caps = caption_batch(processor, model, batch_imgs, prompt, max_new)
                pending.extend(
                    {"image_key": k, "caption": c} for k, c in zip(batch_keys, caps)
                )
            except Exception as e:  # keep the shard alive on a bad image/batch
                print(f"[shard {args.shard}] batch failed ({e}); skipping", flush=True)
            n_done += len(batch_keys)
            batch_keys, batch_imgs = [], []

            if len(pending) >= args.flush_every:
                flush()
                rate = n_done / max(1e-9, time.time() - t0)
                eta_h = (len(todo) - n_done) / max(1e-9, rate) / 3600
                print(
                    f"[shard {args.shard}] {n_done}/{len(todo)} "
                    f"({100 * n_done / len(todo):.1f}%) {rate:.2f} img/s ETA {eta_h:.2f}h "
                    f"missing={n_missing}",
                    flush=True,
                )

        if batch_keys:
            try:
                caps = caption_batch(processor, model, batch_imgs, prompt, max_new)
                pending.extend(
                    {"image_key": k, "caption": c} for k, c in zip(batch_keys, caps)
                )
                n_done += len(batch_keys)
            except Exception as e:
                print(f"[shard {args.shard}] final batch failed ({e})", flush=True)
    flush()

    dt = time.time() - t0
    print(
        f"[shard {args.shard}] DONE captioned={n_done} missing_in_zip={n_missing} "
        f"elapsed={dt / 3600:.2f}h rate={n_done / max(1e-9, dt):.2f} img/s -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
