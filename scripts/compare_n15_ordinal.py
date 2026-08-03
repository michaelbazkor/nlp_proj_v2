#!/usr/bin/env python3
"""Run ordinal-train / binary-test comparison on 15-user subset (PCA vs attention).

Reuses text_only representations from the parent n30 cohort.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssr.config import load_config
from ssr.data.cohort import build_cohort
from ssr.io_utils import atomic_write_json
from ssr.train.cv import run_training

RUNS = [
    {
        "name": "pca",
        "config": "configs/poc_n15_ordinal_pca.yaml",
        "train_out": "artifacts/text_only_n15_ordinal_pca/train/metrics.json",
    },
    {
        "name": "attention",
        "config": "configs/poc_n15_ordinal_attention.yaml",
        "train_out": "artifacts/text_only_n15_ordinal_attention/train/metrics.json",
    },
]

REP_SOURCE = ROOT / "artifacts" / "text_only" / "reps"


def _rep_roots(cfg) -> dict[str, Path]:
    roots = {}
    for mcfg in cfg.represent.get("models") or []:
        name = mcfg["name"]
        roots[name] = REP_SOURCE / name
    if not roots:
        for d in REP_SOURCE.iterdir():
            if d.is_dir():
                roots[d.name] = d
    return roots


def _summarize(results: dict) -> dict:
    out = {}
    for key, s in results.get("summary", {}).items():
        out[key] = {
            "auc_mean": s["auc_roc"]["mean"],
            "pr_mean": s["pr_auc"]["mean"],
            "f1_mean": s["f1"]["mean"],
            "d_mean": s["cohens_d"]["mean"],
        }
    return out


def _format_md(cohort_meta: dict, results: dict[str, dict]) -> str:
    lines = [
        "# Ordinal train / binary test — 15-user subset (PCA vs attention)\n",
        f"- **Cohort:** 15 users stratified from fixed `cohort_n30` "
        f"({cohort_meta.get('n_high_sampled', '?')} high-risk)\n",
        "- **Train target:** suicide score 0–6 (MSE)\n",
        "- **Dev select / test metrics:** binary high-risk (`suicide >= 3`, AUC/F1)\n",
        "- **Reps:** reused from `artifacts/text_only/reps`\n",
        "\n| Fusion | Variant | AUC | PR-AUC | F1 | Cohen's d |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for fusion_name, summary in results.items():
        for variant, s in summary.items():
            lines.append(
                f"| {fusion_name} | {variant} | {s['auc_mean']:.3f} | {s['pr_mean']:.3f} | "
                f"{s['f1_mean']:.3f} | {s['d_mean']:.3f} |"
            )
    lines.append("\n## PCA vs attention delta (AUC)\n")
    pca = results.get("pca", {})
    att = results.get("attention", {})
    for variant in sorted(set(pca) | set(att)):
        if variant in pca and variant in att:
            d = att[variant]["auc_mean"] - pca[variant]["auc_mean"]
            lines.append(f"- **{variant}:** {d:+.3f}")
    return "\n".join(lines)


def main():
    all_results: dict[str, dict] = {}
    cohort_meta = {}

    for spec in RUNS:
        cfg = load_config(ROOT / spec["config"])
        if not cfg.represent.get("models"):
            cfg.represent["models"] = [
                {"name": "qwen3_0_6b"},
                {"name": "smollm2_360m"},
            ]
        cohort = build_cohort(cfg, assert_paper=False)
        meta_path = ROOT / "artifacts" / "cohort_n30_subn15_meta.json"
        if meta_path.exists():
            cohort_meta = json.loads(meta_path.read_text())
        rep_roots = _rep_roots(cfg)
        print(f"\n=== {spec['name']}: {len(cohort)} users, fusion={cfg.fusion.get('method')} ===")
        results = run_training(cfg, cohort, rep_roots)
        summary = _summarize(results)
        all_results[spec["name"]] = summary
        out_path = ROOT / spec["train_out"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_path, results)

    payload = {
        "cohort": cohort_meta,
        "train_target": "ordinal",
        "eval_target": "high",
        "results": all_results,
    }
    json_path = ROOT / "artifacts" / "n15_ordinal_fusion_comparison.json"
    atomic_write_json(json_path, payload)

    md_path = ROOT / "reports" / "n15_ordinal_fusion_comparison.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_format_md(cohort_meta, all_results), encoding="utf-8")
    print(f"\n[compare] wrote {json_path}")
    print(f"[compare] wrote {md_path}")


if __name__ == "__main__":
    main()
