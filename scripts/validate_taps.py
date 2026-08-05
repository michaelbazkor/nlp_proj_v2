"""Check that hook-based extraction reproduces the old output_hidden_states path.

The full-cohort run cannot materialize every layer's hidden states, so extraction was
rewritten to use forward hooks and a single generate() pass. This script runs the old
reference implementation and the new one on a small model and compares every
(layer, position) vector.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssr.represent.extract import _extract_chunk, _layer_indices  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
POSITIONS = ["input_only", "last_prompt", "cot", "final_pred"]


@torch.inference_mode()
def reference(tok, model, prompt_text, layer_idxs, positions, max_new, device, hint):
    """Verbatim pre-rewrite logic: separate prompt forward + generate(output_hidden_states)."""
    enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[-1]

    corpus_start = prompt_text.find(hint)
    corpus_end = corpus_start + len(hint)
    offsets = tok(
        prompt_text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True
    )["offset_mapping"][0].tolist()
    tok_start, tok_end = 0, prompt_len - 1
    for i, (a, b) in enumerate(offsets):
        if a <= corpus_start < b or (a >= corpus_start and tok_start == 0 and i > 0):
            if a >= corpus_start and tok_start == 0:
                tok_start = i
        if a < corpus_end <= b or b <= corpus_end:
            tok_end = i
    tok_start = max(0, min(tok_start, prompt_len - 1))
    tok_end = max(tok_start, min(tok_end, prompt_len - 1))

    hs_prompt = model(
        input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=True
    ).hidden_states
    gen_out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    gen_len = gen_out.sequences.shape[-1] - prompt_len
    per_layer = {li: [] for li in layer_idxs}
    if gen_len > 0:
        for step_hs in gen_out.hidden_states:
            for li in layer_idxs:
                per_layer[li].append(step_hs[li][:, -1, :])

    out = {}
    for li in layer_idxs:
        h = hs_prompt[li][0]
        for pos in positions:
            if pos == "input_only":
                v = h[tok_start : tok_end + 1].mean(dim=0)
            elif pos == "last_prompt":
                v = h[prompt_len - 1]
            elif pos == "cot":
                st = torch.cat(per_layer[li], dim=0)
                v = st[:-1].mean(dim=0) if st.shape[0] > 1 else st[0]
            else:
                v = per_layer[li][-1][0]
            out[f"{li}:{pos}"] = v.detach().float().cpu().numpy()
    out["__span__"] = np.array([tok_start, tok_end])
    out["__gen_text__"] = np.array(
        [tok.decode(gen_out.sequences[0, prompt_len:], skip_special_tokens=True)]
    )
    return out


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device).eval()

    corpus = " ".join(
        f"[post] day {i} felt tired and alone, nobody replied to me. [image] a dim room at night."
        for i in range(1, 25)
    )
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": f"Rate suicide risk.\n\n{corpus}\n\nAnswer:"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    n_layers = len(model.model.layers)
    layer_idxs = _layer_indices(n_layers, None, 8)
    print(f"model={MODEL} n_layers={n_layers} taps={layer_idxs}")

    ref = reference(tok, model, prompt, layer_idxs, POSITIONS, 24, device, corpus)
    new = _extract_chunk(
        tok, model, prompt, layer_idxs, POSITIONS, 24, device, corpus
    )

    print(f"ref span={ref['__span__'].tolist()}")
    print(f"gen_text match: {ref['__gen_text__'][0] == new['__gen_text__'][0]}")
    worst = 0.0
    worst_key = ""
    for k in sorted(ref):
        if k.startswith("__"):
            continue
        a, b = ref[k], new[k]
        d = float(np.max(np.abs(a - b)) / (np.abs(a).max() + 1e-9))
        if d > worst:
            worst, worst_key = d, k
        print(f"  {k:>18}  max_rel_diff={d:.3e}  ref_norm={np.linalg.norm(a):.3f}")
    print(f"\nWORST: {worst_key} rel_diff={worst:.3e}")
    print("VERDICT:", "EQUIVALENT" if worst < 1e-4 else "MISMATCH")


if __name__ == "__main__":
    main()
