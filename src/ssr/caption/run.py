"""Stage 1: image captioning with a VLM. Cache keyed by PostId (image_key)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from ssr.config import Config
from ssr.data.images import ImageStore
from ssr.io_utils import atomic_write_json, exists_nonempty


def _load_caption_model(model_id: str, device: str, dtype_name: str):
    from transformers import AutoProcessor

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(
        dtype_name, torch.float32
    )
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
    out = cfg.art("captions.parquet")
    meta_path = cfg.art("captions_meta.json")

    existing: dict[str, str] = {}
    if exists_nonempty(out) and not force:
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
            {"n_cached": len(existing), "n_requested": len(keys), "n_new": 0, "skipped": True},
        )
        return {k: existing[k] for k in keys if k in existing}

    store = ImageStore(cfg.paths.pics_zip)
    if not store.available:
        print(f"[captions] pics.zip not ready at {cfg.paths.pics_zip}; returning empty captions")
        atomic_write_json(
            meta_path,
            {"n_cached": len(existing), "n_requested": len(keys), "n_new": 0, "zip_ready": False},
        )
        return existing

    device = cfg.images.get("device", "cpu")
    model_id = cfg.images["caption_model"]
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


def load_captions(cfg: Config) -> dict[str, str]:
    out = cfg.art("captions.parquet")
    if not exists_nonempty(out):
        return {}
    df = pd.read_parquet(out)
    key_col = "image_key" if "image_key" in df.columns else "blob_guid"
    return dict(zip(df[key_col], df["caption"]))
