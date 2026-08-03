"""Stage 1: image captioning with a VLM.

Cache is model-scoped under ``artifacts/captions/{model_slug}/`` so POC
SmolVLM captions cannot be reused for a different VLM on the real run.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from ssr.config import Config
from ssr.data.images import ImageStore
from ssr.io_utils import atomic_write_json, exists_nonempty


def _model_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.replace("/", "__"))


def caption_paths(cfg: Config) -> tuple[Path, Path]:
    """Return (parquet_path, meta_path) for the configured caption model."""
    model_id = cfg.images["caption_model"]
    slug = _model_slug(model_id)
    root = cfg.art("captions", slug)
    root.mkdir(parents=True, exist_ok=True)
    return root / "captions.parquet", root / "captions_meta.json"


def _load_caption_model(model_id: str, device: str, dtype_name: str):
    from transformers import AutoProcessor

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(
            f"Unsupported caption dtype {dtype_name!r}; "
            f"expected one of {sorted(dtype_map)}"
        )
    dtype = dtype_map[dtype_name]
    processor = AutoProcessor.from_pretrained(model_id)
    try:
        from transformers import AutoModelForImageTextToText as _AutoVLM
    except ImportError:
        try:
            from transformers import AutoModelForVision2Seq as _AutoVLM
        except ImportError:
            from transformers import AutoModelForCausalLM as _AutoVLM
    model = _AutoVLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return processor, model


def _caption_one(processor, model, image, prompt: str, max_new_tokens: int, device: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
    except Exception:
        text = prompt

    inputs = processor(text=text, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    in_len = inputs["input_ids"].shape[-1]
    gen = out[0][in_len:]
    caption = processor.batch_decode([gen], skip_special_tokens=True)[0].strip()
    if not caption:
        caption = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    return caption


def run_captions(
    cfg: Config,
    posts: pd.DataFrame,
    *,
    force: bool = False,
) -> dict[str, str]:
    """Caption unique image_key (PostId) values present in posts."""
    model_id = cfg.images["caption_model"]
    out, meta_path = caption_paths(cfg)

    existing: dict[str, str] = {}
    if exists_nonempty(out) and exists_nonempty(meta_path) and not force:
        meta = json.loads(meta_path.read_text())
        cached_model = meta.get("model")
        if cached_model != model_id:
            raise RuntimeError(
                f"Caption cache model mismatch: cache has {cached_model!r}, "
                f"config requests {model_id!r}. Use --force or delete {out.parent}."
            )
        df = pd.read_parquet(out)
        if len(df) and ("image_key" in df.columns or "blob_guid" in df.columns):
            key_col = "image_key" if "image_key" in df.columns else "blob_guid"
            existing = dict(zip(df[key_col], df["caption"]))

    key_col = "image_key" if "image_key" in posts.columns else "blob_guid"
    keys = sorted({k for k in posts[key_col].dropna().unique().tolist()})
    max_per_user = cfg.images.get("max_images_per_user")
    if max_per_user is not None:
        capped = []
        for _, g in posts[posts[key_col].notna()].groupby("UserId"):
            capped.extend(g[key_col].dropna().unique().tolist()[: int(max_per_user)])
        keys = sorted(set(capped))

    todo = [k for k in keys if k not in existing]
    if not todo:
        atomic_write_json(
            meta_path,
            {
                "model": model_id,
                "n_cached": len(existing),
                "n_requested": len(keys),
                "n_new": 0,
                "skipped": True,
            },
        )
        return {k: existing[k] for k in keys if k in existing}

    store = ImageStore(cfg.paths.pics_zip)
    if not store.available:
        print(f"[captions] pics.zip not ready at {cfg.paths.pics_zip}; returning empty captions")
        atomic_write_json(
            meta_path,
            {
                "model": model_id,
                "n_cached": len(existing),
                "n_requested": len(keys),
                "n_new": 0,
                "zip_ready": False,
            },
        )
        return existing

    device = cfg.images.get("device", "cpu")
    prompt = cfg.images["caption_prompt"].strip()
    max_new = int(cfg.images.get("max_new_tokens", 64))
    dtype_name = cfg.images.get("dtype", "float32")

    print(f"[captions] loading {model_id} on {device} ...")
    processor, model = _load_caption_model(model_id, device, dtype_name)

    new_rows = []
    missing = 0
    with store:
        for key in tqdm(todo, desc="captioning"):
            img = store.load_pil(key)
            if img is None:
                missing += 1
                continue
            try:
                cap = _caption_one(processor, model, img, prompt, max_new, device)
            except Exception as e:
                print(f"[captions] failed {key}: {e}")
                continue
            existing[key] = cap
            new_rows.append({"image_key": key, "caption": cap})

    df = pd.DataFrame([{"image_key": k, "caption": v} for k, v in existing.items()])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    atomic_write_json(
        meta_path,
        {
            "model": model_id,
            "n_cached": len(existing),
            "n_requested": len(keys),
            "n_new": len(new_rows),
            "n_missing_in_zip": missing,
            "zip_ready": True,
        },
    )
    return {k: existing[k] for k in keys if k in existing}


def load_captions(cfg: Config, *, require: bool = False) -> dict[str, str]:
    """Load captions for the configured VLM.

    If ``require`` is True, raise when the cache is missing or for the wrong model.
    """
    model_id = cfg.images["caption_model"]
    out, meta_path = caption_paths(cfg)

    # Legacy flat path (POC) — only accept if meta model matches
    legacy = cfg.art("captions.parquet")
    legacy_meta = cfg.art("captions_meta.json")

    if exists_nonempty(out) and exists_nonempty(meta_path):
        meta = json.loads(meta_path.read_text())
        if meta.get("model") != model_id:
            if require:
                raise RuntimeError(
                    f"Caption cache model mismatch: {meta.get('model')!r} vs {model_id!r}"
                )
            return {}
        df = pd.read_parquet(out)
        key_col = "image_key" if "image_key" in df.columns else "blob_guid"
        return dict(zip(df[key_col], df["caption"]))

    if exists_nonempty(legacy):
        cached_model = None
        if exists_nonempty(legacy_meta):
            cached_model = json.loads(legacy_meta.read_text()).get("model")
        if cached_model == model_id:
            df = pd.read_parquet(legacy)
            key_col = "image_key" if "image_key" in df.columns else "blob_guid"
            return dict(zip(df[key_col], df["caption"]))
        if require:
            raise RuntimeError(
                f"Legacy captions at {legacy} are for model {cached_model!r}, "
                f"not {model_id!r}. Re-run captions --force."
            )
        return {}

    if require:
        raise RuntimeError(
            f"No captions found for {model_id!r} at {out}. Run: "
            f"python -m ssr.cli --config <cfg> captions --force"
        )
    return {}
