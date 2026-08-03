"""Hidden-state representation extraction from open-source LLMs (H100-ready).

H100 rules enforced here:
  - One LLM loaded at a time; explicit VRAM cleanup between models
  - FlashAttention-2 when installed
  - FP8 / 4-bit / BF16 per model config (70B -> quantized on 80 GB)
"""
from __future__ import annotations

import gc
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ssr.config import Config
from ssr.io_utils import atomic_write_json
from ssr.represent.prompts import build_user_prompt
from ssr.represent.store import has_user_reps, save_user_reps

POST_SPLIT_RE = re.compile(r"(?=\[post\])")


def _dtype(name: str):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if hasattr(torch, "float8_e4m3fn"):
        mapping["float8_e4m3fn"] = torch.float8_e4m3fn
        mapping["fp8"] = torch.float8_e4m3fn
    return mapping.get(str(name).lower(), torch.bfloat16)


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def _is_4bit(dtype_name: str, model_cfg: dict) -> bool:
    if model_cfg.get("load_in_4bit") or model_cfg.get("quantize") in ("4bit", "nf4", True):
        return True
    return str(dtype_name).lower() in ("4bit", "nf4", "int4")


def _is_8bit(dtype_name: str, model_cfg: dict) -> bool:
    if model_cfg.get("load_in_8bit"):
        return True
    return str(dtype_name).lower() in ("8bit", "int8", "fp8_weight_only")


def _load_causal_lm(hf_id: str, model_cfg: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_name = str(model_cfg.get("dtype", "bfloat16"))
    use_flash = bool(model_cfg.get("flash_attention", True)) and _has_flash_attn()

    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    common: dict[str, Any] = {"trust_remote_code": True}
    if use_flash:
        common["attn_implementation"] = "flash_attention_2"
        print(f"[represent] FlashAttention-2 enabled for {hf_id}")

    device_map = model_cfg.get("device_map", "auto")

    if _is_4bit(dtype_name, model_cfg):
        from transformers import BitsAndBytesConfig

        try:
            import bitsandbytes  # noqa: F401
        except ImportError as e:
            raise ImportError("4-bit loading requires: pip install bitsandbytes") from e

        compute = _dtype(str(model_cfg.get("bnb_compute_dtype", "bfloat16")))
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute,
            bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_double_quant", True)),
            bnb_4bit_quant_type=str(model_cfg.get("bnb_quant_type", "nf4")),
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            quantization_config=bnb,
            device_map=device_map,
            **common,
        )
    elif _is_8bit(dtype_name, model_cfg):
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            quantization_config=bnb,
            device_map=device_map,
            **common,
        )
    else:
        torch_dtype = _dtype(dtype_name)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **common,
        )

    model.eval()
    run_device = str(next(model.parameters()).device)
    print(f"[represent] loaded {hf_id} dtype={dtype_name} device={run_device}")
    return tok, model, run_device


def _free_model(model, tok=None) -> None:
    del model
    if tok is not None:
        del tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _layer_indices(n_layers: int, taps: int | None, stride: int | None) -> list[int]:
    if taps is not None:
        if taps >= n_layers:
            return list(range(1, n_layers + 1))
        idxs = np.linspace(1, n_layers, num=taps, dtype=int)
        return sorted(set(int(i) for i in idxs))
    if stride is not None:
        stride = max(1, int(stride))
        idxs = list(range(stride, n_layers + 1, stride))
        if n_layers not in idxs:
            idxs.append(n_layers)
        return idxs
    return [n_layers]


def _split_corpus_into_chunks(corpus: str, max_chars: int) -> list[str]:
    if len(corpus) <= max_chars:
        return [corpus]
    parts = [p for p in POST_SPLIT_RE.split(corpus) if p.strip()]
    if not parts:
        return [corpus[i : i + max_chars] for i in range(0, len(corpus), max_chars)]
    chunks: list[str] = []
    cur = ""
    for p in parts:
        if not cur:
            cur = p
        elif len(cur) + len(p) <= max_chars:
            cur = cur + p
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    final = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            for i in range(0, len(c), max_chars):
                final.append(c[i : i + max_chars])
    return final


def _apply_chat(tok, user_text: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": user_text}]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tok.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        try:
            return tok.apply_chat_template(messages, **kwargs)
        except Exception:
            return user_text + "\n\nAssistant:"


@torch.inference_mode()
def _extract_chunk(
    tok,
    model,
    prompt_text: str,
    layer_idxs: list[int],
    positions: list[str],
    max_new_tokens: int,
    device: str,
) -> dict[str, np.ndarray]:
    enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)

    prompt_len = input_ids.shape[-1]
    corpus_start = prompt_text.find("--- USER POSTS ---")
    corpus_end = prompt_text.find("--- END ---")
    if corpus_start < 0:
        corpus_start = 0
    if corpus_end < 0:
        corpus_end = len(prompt_text)

    enc_off = tok(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = enc_off.get("offset_mapping")
    if offsets is not None:
        offsets = offsets[0].tolist()
        tok_start, tok_end = 0, prompt_len - 1
        for i, (a, b) in enumerate(offsets):
            if a <= corpus_start < b or (a >= corpus_start and tok_start == 0 and i > 0):
                if a >= corpus_start and tok_start == 0:
                    tok_start = i
            if a < corpus_end <= b or b <= corpus_end:
                tok_end = i
        tok_start = max(0, min(tok_start, prompt_len - 1))
        tok_end = max(tok_start, min(tok_end, prompt_len - 1))
    else:
        tok_start, tok_end = 0, prompt_len - 1

    out_prompt = model(
        input_ids=input_ids,
        attention_mask=attn,
        output_hidden_states=True,
        use_cache=True,
    )
    hs_prompt = out_prompt.hidden_states

    gen_out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True,
        pad_token_id=tok.pad_token_id,
    )
    sequences = gen_out.sequences
    gen_len = sequences.shape[-1] - prompt_len

    gen_hs_by_layer: dict[int, list[torch.Tensor]] = {li: [] for li in layer_idxs}
    if gen_len > 0 and getattr(gen_out, "hidden_states", None) is not None:
        for step_hs in gen_out.hidden_states:
            for li in layer_idxs:
                if li >= len(step_hs):
                    continue
                t = step_hs[li]
                gen_hs_by_layer[li].append(t[:, -1, :])

    results: dict[str, np.ndarray] = {}
    for li in layer_idxs:
        h = hs_prompt[li][0]
        for pos in positions:
            if pos == "input_only":
                vec = h[tok_start : tok_end + 1].mean(dim=0)
            elif pos == "last_prompt":
                vec = h[prompt_len - 1]
            elif pos == "cot":
                if gen_hs_by_layer[li]:
                    stacked = torch.cat(gen_hs_by_layer[li], dim=0)
                    vec = stacked[:-1].mean(dim=0) if stacked.shape[0] > 1 else stacked[0]
                else:
                    vec = h[prompt_len - 1]
            elif pos == "final_pred":
                vec = gen_hs_by_layer[li][-1][0] if gen_hs_by_layer[li] else h[prompt_len - 1]
            else:
                raise ValueError(f"Unknown position: {pos}")
            results[f"{li}:{pos}"] = vec.detach().float().cpu().numpy()

    if gen_len > 0:
        gen_ids = sequences[0, prompt_len:]
        results["__gen_text__"] = np.array([tok.decode(gen_ids, skip_special_tokens=True)])
    else:
        results["__gen_text__"] = np.array([""])
    return results


def extract_for_model(
    cfg: Config,
    model_cfg: dict,
    corpora: pd.DataFrame,
    *,
    force: bool = False,
) -> Path:
    name = model_cfg["name"]
    hf_id = model_cfg["hf_id"]
    max_ctx = int(model_cfg.get("max_context", 8192))
    max_new = int(model_cfg.get("max_new_tokens", 256))
    enable_thinking = bool(model_cfg.get("enable_thinking", False))
    positions = list(cfg.represent.get("positions", ["input_only", "last_prompt", "cot", "final_pred"]))
    task_prompt = cfg.represent["task_prompt"]

    rep_root = cfg.exp_dir("reps", name)
    rep_root.mkdir(parents=True, exist_ok=True)

    todo = []
    for _, row in corpora.iterrows():
        p = rep_root / f"{str(row['UserId']).replace('/', '_').replace(' ', '')}.npz"
        if force or not has_user_reps(p):
            todo.append(row)

    if not todo:
        print(f"[represent:{name}] all {len(corpora)} users cached")
        return rep_root

    print(f"[represent:{name}] loading {hf_id} ({len(todo)} users todo) ...")
    tok, model, run_device = _load_causal_lm(hf_id, model_cfg)
    n_layers = int(getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", 12)))
    layer_idxs = _layer_indices(n_layers, model_cfg.get("layer_taps"), model_cfg.get("layer_stride"))
    print(f"[represent:{name}] n_layers={n_layers} tapping={layer_idxs}")

    max_prompt_tokens = max_ctx - max_new - 64
    max_chars = max_prompt_tokens * 3

    meta_global = {
        "model": name,
        "hf_id": hf_id,
        "n_layers": n_layers,
        "layer_idxs": layer_idxs,
        "positions": positions,
        "hidden_size": int(getattr(model.config, "hidden_size", 0)),
        "dtype": model_cfg.get("dtype"),
        "flash_attention": bool(model_cfg.get("flash_attention", True)) and _has_flash_attn(),
    }
    atomic_write_json(rep_root / "meta.json", meta_global)

    for row in tqdm(todo, desc=f"represent:{name}"):
        uid = row["UserId"]
        corpus = str(row.get("corpus") or "")
        out_path = rep_root / f"{str(uid).replace('/', '_').replace(' ', '')}.npz"
        chunks = _split_corpus_into_chunks(corpus if corpus else "(no posts)", max_chars)
        chunk_results: list[dict[str, np.ndarray]] = []
        gen_texts = []
        for ch in chunks:
            ut = build_user_prompt(task_prompt, ch)
            prompt_text = _apply_chat(tok, ut, enable_thinking)
            enc = tok(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=max_prompt_tokens,
            )
            prompt_text = tok.decode(enc["input_ids"][0], skip_special_tokens=False)
            res = _extract_chunk(tok, model, prompt_text, layer_idxs, positions, max_new, run_device)
            gen_texts.append(str(res.pop("__gen_text__", np.array([""]))[0]))
            chunk_results.append(res)

        keys = chunk_results[0].keys()
        averaged = {k: np.stack([c[k] for c in chunk_results], axis=0).mean(axis=0) for k in keys}
        save_user_reps(
            out_path,
            averaged,
            meta={**meta_global, "UserId": uid, "n_chunks": len(chunks), "gen_text": gen_texts[0][:500]},
        )

    _free_model(model, tok)
    return rep_root


def run_representations(cfg: Config, corpora: pd.DataFrame, *, force: bool = False) -> dict[str, Path]:
    """Extract one model at a time; clear VRAM between models."""
    models = cfg.represent["models"]
    roots: dict[str, Path] = {}
    for i, mcfg in enumerate(models):
        print(f"\n[represent] === model {i + 1}/{len(models)}: {mcfg['name']} ===")
        roots[mcfg["name"]] = extract_for_model(cfg, mcfg, corpora, force=force)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return roots
