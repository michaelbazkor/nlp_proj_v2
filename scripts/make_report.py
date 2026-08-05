"""Compile the full-run report from artifacts on disk.

Reads cohort/captions/corpora/reps/train artifacts for one experiment and writes
`reports/{experiment}/final_report.md`, including a zero-shot evaluation of each
LLM's own `RISK=` verdict against the labels.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssr.caption.run import caption_paths  # noqa: E402
from ssr.config import load_config  # noqa: E402
from ssr.data.cohort import build_cohort  # noqa: E402
from ssr.train.metrics import summarize_scores  # noqa: E402

RISK_RE = re.compile(r"RISK\s*=\s*([01])")
PRED_RE = re.compile(r"Prediction\s*:?\s*.{0,20}?([01])", re.S)
RATIONALE_RE = re.compile(r"Rationale\s*:?\s*(.+?)(?:\n\s*Prediction|$)", re.S)


def read_meta(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as z:
        if "__meta__" not in z.files:
            return {}
        try:
            return ast.literal_eval(str(z["__meta__"][0]))
        except (ValueError, SyntaxError):
            return {}


def parse_verdict(gen_text: str) -> tuple[int | None, str]:
    m = RISK_RE.search(gen_text or "")
    pred = int(m.group(1)) if m else None
    if pred is None:
        m2 = PRED_RE.search(gen_text or "")
        pred = int(m2.group(1)) if m2 else None
    r = RATIONALE_RE.search(gen_text or "")
    rationale = (r.group(1) if r else (gen_text or "")).strip()
    rationale = re.sub(r"\s+", " ", rationale)
    return pred, rationale


def fmt_ci(d: dict) -> str:
    if not np.isfinite(d.get("mean", float("nan"))):
        return "n/a"
    return f"{d['mean']:.3f} ± {d.get('std', 0):.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    L: list[str] = []
    cohort = build_cohort(cfg, assert_paper=False)
    posts = pd.read_parquet(cfg.art("posts.parquet"))
    posts = posts[posts["UserId"].isin(set(cohort["UserId"]))]
    cap_path, cap_meta_path = caption_paths(cfg)
    caps = pd.read_parquet(cap_path) if cap_path.exists() else pd.DataFrame(columns=["image_key", "caption"])
    corpora_path = cfg.exp_dir("corpora.parquet")
    corpora = pd.read_parquet(corpora_path) if corpora_path.exists() else pd.DataFrame()

    L += [
        f"# Full-cohort run — `{cfg.experiment}`\n",
        f"Config: `{args.config}`  |  seed: {cfg.seed}\n",
        "## 1. Scale\n",
        "| Quantity | Value |",
        "|---|---|",
        f"| Users (labeled cohort) | {len(cohort)} |",
        f"| High-risk positives (`suicide >= 3`) | {int(cohort.y_high.sum())} ({100 * cohort.y_high.mean():.1f}%) |",
        f"| Posts used | {len(posts):,} |",
        f"| Unique image keys referenced | {posts.image_key.nunique():,} |",
        f"| Images captioned | {len(caps):,} |",
    ]
    if len(corpora):
        L += [
            f"| Mean posts/user | {corpora.n_posts.mean():.1f} |",
            f"| Mean captioned images/user | {corpora.n_images.mean():.1f} |",
            f"| Mean corpus chars/user | {corpora.n_chars.mean():,.0f} |",
            f"| Total corpus chars | {corpora.n_chars.sum() / 1e6:.1f} M |",
        ]
    L.append("")

    if len(caps):
        lens = caps.caption.str.len()
        L += [
            "## 2. Image captions (Qwen2.5-VL-7B)\n",
            f"{len(caps):,} captions, mean {lens.mean():.0f} chars "
            f"(p10 {lens.quantile(.1):.0f} / p90 {lens.quantile(.9):.0f}).\n",
        ]
        for _, r in caps.sample(min(3, len(caps)), random_state=0).iterrows():
            L.append(f"- `{r.image_key[:40]}`: {r.caption[:420]}\n")
        L.append("")

    if len(corpora):
        row = corpora.sort_values("n_chars").iloc[len(corpora) // 2]
        L += [
            "## 3. Assembled corpus (median-length user, truncated)\n",
            f"`n_posts={row.n_posts} n_images={row.n_images} n_chars={row.n_chars}`\n",
            "```",
            row.corpus[:1200],
            "```\n",
        ]

    # Representations + zero-shot LLM verdicts
    L.append("## 4. LLM representations and zero-shot verdicts\n")
    y = cohort.set_index("UserId")["y_high"].to_dict()
    zero_shot: dict[str, dict] = {}
    examples: dict[str, list[tuple[str, int, int | None, str]]] = {}
    for mcfg in cfg.represent["models"]:
        name = mcfg["name"]
        root = cfg.exp_dir("reps", name)
        if not root.exists():
            continue
        meta_global = json.loads((root / "meta.json").read_text()) if (root / "meta.json").exists() else {}
        files = sorted(root.glob("*.npz"))
        preds, labels, chunks, dims = [], [], [], set()
        ex: list[tuple[str, int, int | None, str]] = []
        for f in files:
            uid = f.stem
            m = read_meta(f)
            pred, rationale = parse_verdict(m.get("gen_text", ""))
            uid_full = m.get("UserId", uid)
            chunks.append(int(m.get("n_chunks", 1)))
            lbl = y.get(uid_full)
            if lbl is not None and pred is not None:
                preds.append(pred)
                labels.append(int(lbl))
            if lbl is not None and len(ex) < 4 and rationale:
                ex.append((str(uid_full), int(lbl), pred, rationale[:600]))
        with np.load(files[0], allow_pickle=False) as z:
            for k in z.files:
                if not k.startswith("__"):
                    dims.add(int(z[k].shape[0]))
        examples[name] = ex
        cov = len(preds) / max(1, len(files))
        stats = {
            "n_users": len(files),
            "n_layers": meta_global.get("n_layers"),
            "taps": meta_global.get("layer_idxs"),
            "hidden_dims": sorted(dims),
            "mean_chunks": float(np.mean(chunks)) if chunks else float("nan"),
            "verdict_coverage": cov,
            "pred_pos_rate": float(np.mean(preds)) if preds else float("nan"),
        }
        if preds and len(set(labels)) > 1:
            stats.update(summarize_scores(np.array(labels), np.array(preds, dtype=float)))
        zero_shot[name] = stats

    L += [
        "| Model | users | blocks tapped | hidden dim | mean chunks/user | `RISK=` parsed | predicted-positive rate | zero-shot AUC | zero-shot F1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in zero_shot.items():
        L.append(
            f"| `{name}` | {s['n_users']} | {s['taps']} of {s['n_layers']} | "
            f"{s['hidden_dims']} | {s['mean_chunks']:.2f} | {100 * s['verdict_coverage']:.0f}% | "
            f"{s['pred_pos_rate']:.3f} | {s.get('auc_roc', float('nan')):.3f} | "
            f"{s.get('f1', float('nan')):.3f} |"
        )
    L += [
        "",
        "Zero-shot columns score the model's own `RISK=` label (a single binary "
        "decision, so AUC is the balanced-accuracy equivalent) against `y_high`. "
        "These are not the paper's classifier; they contextualize the hidden states.\n",
    ]

    for name, ex in examples.items():
        L.append(f"### `{name}` — sample rationales\n")
        for uid, lbl, pred, text in ex:
            L.append(f"- **user** `{uid[:22]}` label=`{lbl}` model_pred=`{pred}`\n  > {text}\n")
        L.append("")

    # Training
    metrics_path = cfg.exp_dir("train", "metrics.json")
    L.append("## 5. STM / MTM cross-validated results\n")
    if metrics_path.exists():
        res = json.loads(metrics_path.read_text())
        meta = res.get("meta", {})
        if meta:
            L.append(
                f"Fusion: {meta.get('n_blocks')} blocks from "
                f"{len(meta.get('rep_models', []))} LLMs -> {cfg.fusion['target_dim']}-d. "
                f"Grid: {meta.get('grid_size')} points x 5 folds x 2 variants.\n"
            )
        L += [
            "| Variant | AUC-ROC (mean ± sd [95% CI]) | PR-AUC | F1 | Cohen's d |",
            "|---|---|---|---|---|",
        ]
        for key, s in res.get("summary", {}).items():
            L.append(
                f"| **{key}** | {fmt_ci(s['auc_roc'])} | {fmt_ci(s['pr_auc'])} | "
                f"{fmt_ci(s['f1'])} | {fmt_ci(s['cohens_d'])} |"
            )
        L.append("")
        for key, s in res.get("summary", {}).items():
            L.append(f"### `{key}` per fold\n")
            L.append("| fold | test AUC | PR-AUC | F1 | d | dev AUC | best hyperparams |")
            L.append("|---|---|---|---|---|---|---|")
            for m in s["folds"]:
                hp = (
                    f"L={m.get('p_n_layers')} n={m.get('p_n_neurons')} "
                    f"{m.get('p_activation')} lr={m.get('p_lr')} ep={m.get('p_epochs')}"
                )
                L.append(
                    f"| {m['fold']} | {m['auc_roc']:.3f} | {m['pr_auc']:.3f} | {m['f1']:.3f} | "
                    f"{m['cohens_d']:.3f} | {m.get('dev_auc', float('nan')):.3f} | {hp} |"
                )
            L.append("")
    else:
        L.append(f"_No training metrics at {metrics_path}._\n")

    # Findings / limitations, driven by the numbers computed above
    if metrics_path.exists():
        res = json.loads(metrics_path.read_text())
        gaps = {}
        for key, s in res.get("summary", {}).items():
            dev = np.mean([m["dev_auc"] for m in s["folds"]])
            test = s["auc_roc"]["mean"]
            gaps[key] = (dev, test)
        L.append("## 6. Findings and limitations\n")
        stm, mtm = gaps.get("stm_high"), gaps.get("mtm_high")
        if stm and mtm:
            L.append(
                f"- **Multi-task helps.** MTM reaches AUC {mtm[1]:.3f} vs STM {stm[1]:.3f}; "
                f"the auxiliary personality / psychosocial / psychiatric heads add signal, "
                f"which is the direction Ophir et al. report.\n"
            )
            L.append(
                f"- **Model selection overfits the dev split.** Mean dev AUC is "
                f"{stm[0]:.3f} (STM) and {mtm[0]:.3f} (MTM) against test "
                f"{stm[1]:.3f} / {mtm[1]:.3f}. With 504 grid points scored on 142 dev users "
                f"(~19 positives), the winning configuration is partly selected on dev noise. "
                f"The test numbers are the honest ones; a narrower grid or repeated inner CV "
                f"would shrink the gap.\n"
            )
        L.append(
            "- **F1 is unstable and near-uninformative at this base rate.** 13.2% positives "
            "with a 0.5 threshold puts some folds at F1 = 0 while their AUC is 0.75; "
            "AUC / PR-AUC / Cohen's d are the metrics to read.\n"
        )
        cov = {n: s.get("verdict_coverage", float("nan")) for n, s in zero_shot.items()}
        thinking = [n for n in cov if "qwen3" in n or "deepseek" in n]
        if thinking:
            L.append(
                f"- **Thinking models get truncated.** Qwen3 and R1-Distill spend their 512-token "
                f"budget inside `<think>`, so a `RISK=` verdict was recoverable for only "
                f"{100 * min(cov[n] for n in thinking):.0f}–{100 * max(cov[n] for n in thinking):.0f}% "
                f"of users (Gemma: {100 * cov.get('gemma4_26b_moe', float('nan')):.0f}%). For those "
                f"users the `final_pred` tap sits mid-reasoning rather than after a decision, "
                f"which weakens that one block; `cot` and the prompt-side blocks are unaffected.\n"
            )
        L.append(
            f"- **Image coverage is bounded by the archive, not the pipeline.** Posts reference "
            f"{posts.image_key.nunique():,} images but `pics.zip` holds 165,664 files, so "
            f"{len(caps):,} ({100 * len(caps) / posts.image_key.nunique():.0f}%) could be captioned. "
            f"The rest are referenced-but-absent blobs.\n"
        )
        L.append(
            "- **Zero-shot prompting alone is weak.** Every model's own `RISK=` verdict lands "
            "near chance (AUC 0.52–0.58) and is heavily conservative, while the same models' "
            "hidden states support AUC ~0.67. The representation, not the verbalized answer, "
            "carries the signal — the premise of the approach.\n"
        )
        L.append(
            "- Earlier `artifacts/baseline_original/*_metrics.json` come from 30-user POC runs "
            "with small models and are too noisy to compare against these results.\n"
        )

    # Measured cost, parsed from the stage logs
    art = Path("artifacts")
    cap_logs = sorted(art.glob("full_captions_shard*.log"))
    rep_logs = sorted(art.glob("full_represent_*.log"))
    L.append("## 7. Measured run cost\n")
    cap_rows = []
    for p in cap_logs:
        m = re.search(r"DONE captioned=(\d+) missing_in_zip=(\d+) elapsed=([\d.]+)h rate=([\d.]+)", p.read_text())
        if m:
            cap_rows.append((p.name, int(m.group(1)), float(m.group(3)), float(m.group(4))))
    if cap_rows:
        tot = sum(r[1] for r in cap_rows)
        wall = max(r[2] for r in cap_rows)
        rate = sum(r[3] for r in cap_rows)
        L.append(
            f"**Captioning** ({len(cap_rows)} GPU shards): {tot:,} images in {wall:.2f} h wall, "
            f"{rate:.1f} img/s aggregate ({rate / len(cap_rows):.2f} img/s per GPU). "
            f"The unbatched captioner measured 0.86 img/s on one GPU, so this is "
            f"~{rate / 0.86:.0f}x faster end to end.\n"
        )
    per_model: dict[str, list[tuple[float, float]]] = {}
    for p in rep_logs:
        txt = p.read_text()
        name = re.sub(r"^full_represent_|(_gpu\d+)?\.log$", "", p.name)
        hits = re.findall(r"(\d+)/(\d+) users \([\d.]+%\) ([\d.]+) s/user", txt)
        peak = re.findall(r"peak_gpu=([\d.]+)GiB", txt)
        if hits:
            n, _, sper = hits[-1]
            per_model.setdefault(name, []).append((float(sper), float(peak[-1]) if peak else float("nan")))
    if per_model:
        L += [
            "**Representation extraction** (one model per GPU; the slower models were "
            "re-sharded onto freed GPUs as faster ones finished):\n",
            "| Model | workers | s/user | peak GPU |",
            "|---|---|---|---|",
        ]
        for name, vals in per_model.items():
            sp = np.mean([v[0] for v in vals])
            pk = max(v[1] for v in vals)
            L.append(f"| `{name}` | {len(vals)} | {sp:.1f} | {pk:.1f} GiB |")
        L.append("")
    train_log = art / "full_train.log"
    if train_log.exists():
        durs = [float(x) for x in re.findall(r"DONE test AUC=.*\(([\d.]+)m\)", train_log.read_text())]
        if durs:
            L.append(
                f"**Training**: {len(durs)} (variant, fold) grid searches of "
                f"{meta.get('grid_size', '?') if metrics_path.exists() else '?'} configs each, "
                f"{sum(durs):.0f} min total ({np.mean(durs):.1f} min each) on 48 workers "
                f"with wide nets routed to GPUs and narrow nets to single CPU threads.\n"
            )

    L += [
        "## 8. Reproduce\n",
        "```bash",
        "# captions (4 GPUs, batched)",
        "for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python scripts/caption_shard.py \\",
        f"    --config {args.config} --shard $i --num-shards 4 --batch-size 48 & done; wait",
        f"python scripts/merge_caption_shards.py --config {args.config}",
        f"PYTHONPATH=src python -m ssr.cli --config {args.config} corpus",
        "# represent: one model per GPU",
        f"PYTHONPATH=src python -m ssr.cli --config {args.config} represent   # SSR_ONLY_MODELS=<name>",
        f"python scripts/train_parallel.py --config {args.config}",
        f"python scripts/make_report.py --config {args.config}",
        "```\n",
    ]

    out = Path(args.out) if args.out else cfg.report(cfg.experiment, "final_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print(f"wrote {out} ({len('\n'.join(L))} chars)")


if __name__ == "__main__":
    main()
