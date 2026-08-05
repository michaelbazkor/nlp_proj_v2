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

import os
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from ssr.config import Config
from ssr.io_utils import atomic_write_json
from ssr.represent.prompts import build_user_prompt
from ssr.represent.store import has_user_reps, load_user_reps, rep_path, save_user_reps

POST_SPLIT_RE = re.compile(r"(?=\[post\])")


def _dtype(name: str):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(
            f"Unsupported dtype {name!r}. Supported: {sorted(mapping)}. "
            f"(float8 / FP8 is not implemented — use bfloat16 with device_map: auto.)"
        )
    return mapping[name]


def _text_backbone(model):
    """Return the decoder stack that owns ``.layers`` (handles multimodal wrappers)."""
    cand = getattr(model, "model", model)
    if isinstance(getattr(cand, "layers", None), torch.nn.ModuleList):
        return cand
    for attr in ("language_model", "text_model", "decoder", "transformer"):
        sub = getattr(cand, attr, None)
        if sub is not None and isinstance(getattr(sub, "layers", None), torch.nn.ModuleList):
            return sub
    for mod in cand.modules():
        if isinstance(getattr(mod, "layers", None), torch.nn.ModuleList) and len(mod.layers) > 1:
            return mod
    raise RuntimeError("Could not locate the decoder layer stack for hidden-state taps")


class _LayerTaps:
    """Collect pooled hidden states via forward hooks instead of ``output_hidden_states``.

    Requesting ``output_hidden_states`` materializes (n_layers+1, seq, dim) for the
    whole prompt, which is tens of GB at full-cohort corpus lengths. Hooks reduce each
    tapped layer to a handful of vectors as the forward runs, and let a single
    ``generate`` call serve both the prompt and generated-token positions.

    Index convention matches ``transformers`` ``hidden_states``: entry ``li`` is the
    output of block ``li-1``, and entry ``n_layers`` is after the final norm.
    """

    def __init__(self, model, layer_idxs: list[int]):
        self.backbone = _text_backbone(model)
        blocks = self.backbone.layers
        self.n_layers = len(blocks)
        self.layer_idxs = [li for li in layer_idxs if 1 <= li <= self.n_layers]
        self.span: tuple[int, int] = (0, 0)
        self.prompt_mean: dict[int, torch.Tensor] = {}
        self.prompt_last: dict[int, torch.Tensor] = {}
        self.gen: dict[int, list[torch.Tensor]] = {li: [] for li in self.layer_idxs}
        final_norm = getattr(self.backbone, "norm", None)
        self._handles = []
        for li in self.layer_idxs:
            mod = final_norm if (li == self.n_layers and final_norm is not None) else blocks[li - 1]
            self._handles.append(mod.register_forward_hook(self._hook(li)))

    def _hook(self, li: int):
        def fn(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if not isinstance(h, torch.Tensor) or h.dim() != 3:
                return
            last = h[0, -1].detach().float()
            if h.shape[1] > 1:  # prefill pass over the prompt
                a, b = self.span
                self.prompt_mean[li] = h[0, a : b + 1].detach().float().mean(dim=0)
                self.prompt_last[li] = last
            # Mirrors transformers' generate(output_hidden_states=True), whose step 0
            # entry is the prompt pass; its last position feeds the first new token.
            self.gen[li].append(last)

        return fn

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def _model_input_device(model) -> torch.device:
    """First parameter device (works with device_map / multi-GPU)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


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


def _load_causal_lm(
    hf_id: str,
    device: str,
    dtype_name: str,
    *,
    device_map: str | dict | None = None,
    attn_implementation: str | None = "sdpa",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }
    if load_in_8bit or load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=bool(load_in_8bit),
            load_in_4bit=bool(load_in_4bit),
        )
        # bitsandbytes requires a device_map; default to the requested device index
        if device_map is None:
            dev = torch.device(device)
            idx = dev.index if dev.index is not None else 0
            device_map = {"": idx}
    else:
        kwargs["torch_dtype"] = _dtype(dtype_name)

    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if device_map is not None:
        kwargs["device_map"] = device_map

    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    except TypeError:
        # Older transformers may not accept attn_implementation
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)

    if device_map is None and not (load_in_8bit or load_in_4bit):
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
    device: str | torch.device,
    corpus_char_hint: str,
    prompt_input_ids: torch.Tensor | None = None,
    prompt_attention_mask: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Extract representation blocks for one prompt chunk.

    Returns dict keyed by '{layer}:{position}' -> (hidden_dim,) float32.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    if prompt_input_ids is not None:
        input_ids = prompt_input_ids.to(device)
        attn = prompt_attention_mask.to(device) if prompt_attention_mask is not None else None
    else:
        enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)

    prompt_len = input_ids.shape[-1]

    # Locate approximate corpus span inside the prompt for input_only pooling.
    # Prefer the actual corpus/chunk text; fall back to legacy markers.
    hint = (corpus_char_hint or "").strip()
    corpus_start = prompt_text.find(hint) if hint else -1
    if corpus_start >= 0:
        corpus_end = corpus_start + len(hint)
    else:
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

    # One forward pass total: generate()'s prefill also supplies the prompt positions.
    taps = _LayerTaps(model, layer_idxs)
    taps.span = (tok_start, tok_end)
    eos_id = tok.eos_token_id
    gen_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": True,
        "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else eos_id,
    }
    if eos_id is not None:
        gen_kwargs["eos_token_id"] = eos_id
    try:
        gen_out = model.generate(**gen_kwargs)
    finally:
        taps.remove()
    sequences = gen_out.sequences
    gen_len = sequences.shape[-1] - prompt_len

    results: dict[str, np.ndarray] = {}
    for li in layer_idxs:
        if li not in taps.prompt_last:
            raise RuntimeError(f"No hidden state captured for layer {li}")
        steps = taps.gen[li]
        for pos in positions:
            if pos == "input_only":
                vec = taps.prompt_mean[li]
            elif pos == "last_prompt":
                vec = taps.prompt_last[li]
            elif pos == "cot":
                vec = torch.stack(steps[:-1]).mean(dim=0) if len(steps) > 1 else steps[0]
            elif pos == "final_pred":
                vec = steps[-1]
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
    device_map = model_cfg.get("device_map", None)
    attn_implementation = model_cfg.get("attn_implementation", "sdpa")
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

    # Optional SSR_USER_SHARD="i/n": lets several GPUs work one model's user list.
    shard_spec = os.environ.get("SSR_USER_SHARD", "").strip()
    if shard_spec:
        i_s, n_s = (int(x) for x in shard_spec.split("/"))
        todo = [r for k, r in enumerate(todo) if k % n_s == i_s]
        print(f"[represent:{name}] user shard {i_s}/{n_s}: {len(todo)} users")

    if not todo:
        print(f"[represent:{name}] all {len(corpora)} users cached")
        return rep_root

    print(f"[represent:{name}] loading {hf_id} on {device} "
          f"dtype={dtype_name} device_map={device_map!r} "
          f"8bit={bool(model_cfg.get('load_in_8bit'))} "
          f"4bit={bool(model_cfg.get('load_in_4bit'))} ({len(todo)} users) ...")
    tok, model = _load_causal_lm(
        hf_id,
        device,
        dtype_name,
        device_map=device_map,
        attn_implementation=attn_implementation,
        load_in_8bit=bool(model_cfg.get("load_in_8bit", False)),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
    )
    input_device = _model_input_device(model)
    backbone = _text_backbone(model)
    # Count real decoder blocks: multimodal/MoE wrappers (e.g. Gemma-4) nest the text
    # config, so model.config.num_hidden_layers can be wrong.
    n_layers = len(backbone.layers)
    hidden_size = int(getattr(getattr(backbone, "config", None), "hidden_size", 0) or 0)
    layer_idxs = _layer_indices(
        n_layers,
        model_cfg.get("layer_taps"),
        model_cfg.get("layer_stride"),
    )
    print(f"[represent:{name}] n_layers={n_layers} tapping={layer_idxs} input_device={input_device}")

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
        "hidden_size": hidden_size,
        "dtype": dtype_name,
        "device_map": device_map,
        "max_context": max_ctx,
        "max_new_tokens": max_new,
    }
    atomic_write_json(rep_root / "meta.json", meta_global)

    t_start = time.time()
    for i_user, row in enumerate(todo, start=1):
        uid = row["UserId"]
        corpus = str(row.get("corpus") or "")
        out_path = rep_root / f"{str(uid).replace('/', '_').replace(' ', '')}.npz"
        # Re-check here, not just when todo was built: sibling shards on other GPUs
        # may have finished this user in the meantime.
        if not force and has_user_reps(out_path):
            continue

        user_text = build_user_prompt(task_prompt, corpus if corpus else "(no posts)")
        # Chunk if needed (on corpus portion)
        chunks = _split_corpus_into_chunks(corpus if corpus else "(no posts)", max_chars)
        chunk_results: list[dict[str, np.ndarray]] = []
        gen_texts = []
        for ch in chunks:
            ut = build_user_prompt(task_prompt, ch)
            prompt_text = _apply_chat(tok, ut, enable_thinking)
            # Truncate on token ids (keep special-token ids intact).
            # Avoid decode→re-encode, which can corrupt Llama chat markers.
            enc = tok(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=max_prompt_tokens,
            )
            prompt_text = tok.decode(enc["input_ids"][0], skip_special_tokens=False)
            # Prefer staying on token ids when decode round-trips poorly
            res = _extract_chunk(
                tok,
                model,
                prompt_text,
                layer_idxs,
                positions,
                max_new,
                input_device,
                ch,
                prompt_input_ids=enc["input_ids"],
                prompt_attention_mask=enc.get("attention_mask"),
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
                "gen_text": gen_texts[0][:8000] if gen_texts else "",
            },
        )

        if i_user % 10 == 0 or i_user == len(todo):
            el = time.time() - t_start
            rate = el / i_user
            eta_h = rate * (len(todo) - i_user) / 3600
            mem = (
                torch.cuda.max_memory_allocated() / 2**30
                if torch.cuda.is_available()
                else 0.0
            )
            print(
                f"[represent:{name}] {i_user}/{len(todo)} users "
                f"({100 * i_user / len(todo):.1f}%) {rate:.1f} s/user "
                f"ETA {eta_h:.2f}h peak_gpu={mem:.1f}GiB",
                flush=True,
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

    # One-model-per-process scheduling (one GPU each) sets SSR_ONLY_MODELS.
    only = os.environ.get("SSR_ONLY_MODELS", "").strip()
    if only:
        wanted = {n.strip() for n in only.split(",") if n.strip()}
        models = [m for m in models if m["name"] in wanted]
        missing = wanted - {m["name"] for m in models}
        if missing:
            raise ValueError(f"SSR_ONLY_MODELS names not in config: {sorted(missing)}")
        print(f"[represent] restricted to {[m['name'] for m in models]}")

    roots = {}
    for mcfg in models:
        roots[mcfg["name"]] = extract_for_model(cfg, mcfg, corpora, force=force)
    return roots
