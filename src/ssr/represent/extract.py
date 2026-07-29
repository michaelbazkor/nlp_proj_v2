"""Hidden-state representation extraction from open-source LLMs.

For each (model, layer, position) we mean-pool / select a vector:
  - input_only:  mean-pool over the user-corpus token span
  - last_prompt: hidden state at the final prompt token
  - cot:         mean-pool over generated reasoning tokens
  - final_pred:  hidden state at the final generated token

If the corpus exceeds the model context window, split on whole-post
boundaries, extract per chunk, and average across chunks.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ssr.config import Config
from ssr.io_utils import atomic_write_json
from ssr.represent.prompts import build_user_prompt
from ssr.represent.store import has_user_reps, load_user_reps, rep_path, save_user_reps

POST_SPLIT_RE = re.compile(r"(?=\[post\])")


def _dtype(name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def _layer_indices(n_layers: int, taps: int | None, stride: int | None) -> list[int]:
    """Return 0-indexed hidden_states indices (excluding embedding at 0).

    transformers returns hidden_states as tuple length n_layers+1 where
    index 0 is the embedding output and 1..n_layers are transformer blocks.
    We tap transformer blocks only.
    """
    if taps is not None:
        # Evenly spaced across blocks 1..n_layers
        if taps >= n_layers:
            return list(range(1, n_layers + 1))
        # Include early, mid, late
        idxs = np.linspace(1, n_layers, num=taps, dtype=int)
        return sorted(set(int(i) for i in idxs))
    if stride is not None:
        stride = max(1, int(stride))
        idxs = list(range(stride, n_layers + 1, stride))
        if n_layers not in idxs:
            idxs.append(n_layers)
        return idxs
    return [n_layers]  # last layer only


def _split_corpus_into_chunks(corpus: str, max_chars: int) -> list[str]:
    """Split on [post] boundaries so no chunk exceeds ~max_chars."""
    if len(corpus) <= max_chars:
        return [corpus]
    parts = [p for p in POST_SPLIT_RE.split(corpus) if p.strip()]
    if not parts:
        # Hard char split fallback
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
    # If a single post is still too long, hard-split it
    final = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            for i in range(0, len(c), max_chars):
                final.append(c[i : i + max_chars])
    return final


def _load_causal_lm(hf_id: str, device: str, dtype_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=_dtype(dtype_name),
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return tok, model


def _apply_chat(tok, user_text: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": user_text}]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    # Qwen3 thinking mode
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
    corpus_char_hint: str,
) -> dict[str, np.ndarray]:
    """Extract representation blocks for one prompt chunk.

    Returns dict keyed by '{layer}:{position}' -> (hidden_dim,) float32.
    """
    enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)

    prompt_len = input_ids.shape[-1]

    # Locate approximate corpus span inside the prompt for input_only pooling.
    # Heuristic: tokens covering the corpus substring.
    corpus_start = prompt_text.find("--- USER POSTS ---")
    corpus_end = prompt_text.find("--- END ---")
    if corpus_start < 0:
        corpus_start = 0
    if corpus_end < 0:
        corpus_end = len(prompt_text)
    # Map char offsets -> token indices via offset mapping if available
    enc_off = tok(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = enc_off.get("offset_mapping")
    if offsets is not None:
        offsets = offsets[0].tolist()
        tok_start = 0
        tok_end = prompt_len - 1
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

    # Forward with hidden states for the prompt (for input_only + last_prompt)
    out_prompt = model(
        input_ids=input_ids,
        attention_mask=attn,
        output_hidden_states=True,
        use_cache=True,
    )
    hs_prompt = out_prompt.hidden_states  # tuple (n_layers+1,)

    # Generate continuation for CoT / final_pred
    gen_out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True,
        pad_token_id=tok.pad_token_id,
    )
    # gen_out.sequences: (1, prompt_len + gen_len)
    sequences = gen_out.sequences
    gen_len = sequences.shape[-1] - prompt_len

    # Collect per-step hidden states for generated tokens.
    # transformers: generate with output_hidden_states gives a tuple of length gen_len,
    # each entry is a tuple of layer hidden states for that step, shape (batch, 1, dim)
    # OR for some versions (batch, seq, dim). Handle both.
    gen_hs_by_layer: dict[int, list[torch.Tensor]] = {li: [] for li in layer_idxs}
    if gen_len > 0 and getattr(gen_out, "hidden_states", None) is not None:
        for step_hs in gen_out.hidden_states:
            # step_hs: tuple of layers
            for li in layer_idxs:
                if li >= len(step_hs):
                    continue
                t = step_hs[li]  # (batch, seq_or_1, dim)
                gen_hs_by_layer[li].append(t[:, -1, :])  # last pos of this step

    results: dict[str, np.ndarray] = {}
    for li in layer_idxs:
        h = hs_prompt[li][0]  # (prompt_len, dim)
        for pos in positions:
            if pos == "input_only":
                vec = h[tok_start : tok_end + 1].mean(dim=0)
            elif pos == "last_prompt":
                vec = h[prompt_len - 1]
            elif pos == "cot":
                if gen_hs_by_layer[li]:
                    stacked = torch.cat(gen_hs_by_layer[li], dim=0)  # (gen_len, dim)
                    # Mean-pool all generated tokens as CoT proxy
                    # Prefer all but last for cot; last reserved for final_pred
                    if stacked.shape[0] > 1:
                        vec = stacked[:-1].mean(dim=0)
                    else:
                        vec = stacked[0]
                else:
                    vec = h[prompt_len - 1]  # fallback
            elif pos == "final_pred":
                if gen_hs_by_layer[li]:
                    vec = gen_hs_by_layer[li][-1][0]
                else:
                    vec = h[prompt_len - 1]
            else:
                raise ValueError(f"Unknown position: {pos}")
            results[f"{li}:{pos}"] = vec.detach().float().cpu().numpy()

    # Also store generated text for debugging (not used as feature)
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
    """Extract & cache representations for all users for one LLM. Returns rep root."""
    name = model_cfg["name"]
    hf_id = model_cfg["hf_id"]
    device = model_cfg.get("device", "cpu")
    dtype_name = model_cfg.get("dtype", "float32")
    max_ctx = int(model_cfg.get("max_context", 4096))
    max_new = int(model_cfg.get("max_new_tokens", 128))
    enable_thinking = bool(model_cfg.get("enable_thinking", False))
    positions = list(cfg.represent.get("positions", ["input_only", "last_prompt", "cot", "final_pred"]))
    task_prompt = cfg.represent["task_prompt"]

    rep_root = cfg.exp_dir("reps", name)
    rep_root.mkdir(parents=True, exist_ok=True)

    todo = []
    for _, row in corpora.iterrows():
        p = rep_path(rep_root, name, row["UserId"])
        # rep_path already includes model name — fix: store directly under rep_root
        p = rep_root / f"{str(row['UserId']).replace('/', '_').replace(' ', '')}.npz"
        if force or not has_user_reps(p):
            todo.append(row)

    if not todo:
        print(f"[represent:{name}] all {len(corpora)} users cached")
        return rep_root

    print(f"[represent:{name}] loading {hf_id} on {device} ({len(todo)} users) ...")
    tok, model = _load_causal_lm(hf_id, device, dtype_name)
    n_layers = int(getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", 12)))
    layer_idxs = _layer_indices(
        n_layers,
        model_cfg.get("layer_taps"),
        model_cfg.get("layer_stride"),
    )
    print(f"[represent:{name}] n_layers={n_layers} tapping={layer_idxs}")

    # Reserve tokens for generation + template overhead
    # Rough char budget: ~3 chars/token for English social media
    max_prompt_tokens = max_ctx - max_new - 64
    max_chars = max_prompt_tokens * 3

    meta_global = {
        "model": name,
        "hf_id": hf_id,
        "n_layers": n_layers,
        "layer_idxs": layer_idxs,
        "positions": positions,
        "hidden_size": int(getattr(model.config, "hidden_size", 0)),
    }
    atomic_write_json(rep_root / "meta.json", meta_global)

    for row in tqdm(todo, desc=f"represent:{name}"):
        uid = row["UserId"]
        corpus = str(row.get("corpus") or "")
        out_path = rep_root / f"{str(uid).replace('/', '_').replace(' ', '')}.npz"

        user_text = build_user_prompt(task_prompt, corpus if corpus else "(no posts)")
        # Chunk if needed (on corpus portion)
        chunks = _split_corpus_into_chunks(corpus if corpus else "(no posts)", max_chars)
        chunk_results: list[dict[str, np.ndarray]] = []
        gen_texts = []
        for ch in chunks:
            ut = build_user_prompt(task_prompt, ch)
            prompt_text = _apply_chat(tok, ut, enable_thinking)
            # Truncate tokens hard if still over
            enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_prompt_tokens)
            prompt_text = tok.decode(enc["input_ids"][0], skip_special_tokens=False)
            res = _extract_chunk(
                tok, model, prompt_text, layer_idxs, positions, max_new, device, ch
            )
            gen_texts.append(str(res.pop("__gen_text__", np.array([""]))[0]))
            chunk_results.append(res)

        # Average across chunks
        keys = chunk_results[0].keys()
        averaged = {}
        for k in keys:
            stacked = np.stack([c[k] for c in chunk_results], axis=0)
            averaged[k] = stacked.mean(axis=0)

        save_user_reps(
            out_path,
            averaged,
            meta={
                **meta_global,
                "UserId": uid,
                "n_chunks": len(chunks),
                "gen_text": gen_texts[0][:500] if gen_texts else "",
            },
        )

    # Free memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return rep_root


def run_representations(cfg: Config, corpora: pd.DataFrame, *, force: bool = False) -> dict[str, Path]:
    models = cfg.represent["models"]
    # Experiment 4: llama_only — keep only llama-named models
    if cfg.experiment == "llama_only":
        models = [m for m in models if "llama" in m["name"].lower()]
        if not models:
            # Fall back to first model with a note
            print("[represent] llama_only requested but no llama model in config; using all")
            models = cfg.represent["models"]

    roots = {}
    for mcfg in models:
        roots[mcfg["name"]] = extract_for_model(cfg, mcfg, corpora, force=force)
    return roots
