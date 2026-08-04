"""Run the 8-experiment 2×2×2 training matrix on cached representations."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ssr.config import Config
from ssr.io_utils import atomic_write_json
from ssr.train.cv import run_training

# fusion × train_target × architecture
MATRIX: list[tuple[str, str, str]] = [
    ("per_block_pca", "ordinal", "stm"),
    ("per_block_pca", "ordinal", "mtm"),
    ("per_block_pca", "high", "stm"),
    ("per_block_pca", "high", "mtm"),
    ("attention_pool", "ordinal", "stm"),
    ("attention_pool", "ordinal", "mtm"),
    ("attention_pool", "high", "stm"),
    ("attention_pool", "high", "mtm"),
]


def _run_id(fusion: str, train_target: str, model: str) -> str:
    fusion_short = "pca" if fusion in ("per_block_pca", "pca") else "attention"
    train_short = "ordinal" if train_target == "ordinal" else "binary"
    return f"{fusion_short}_{train_short}_{model}"


def run_train_matrix(
    cfg: Config,
    cohort,
    rep_roots: dict[str, Path],
    *,
    out_root: Path | None = None,
) -> dict[str, Any]:
    out_root = out_root or cfg.exp_dir("train_matrix")
    out_root.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {}
    for fusion, train_target, model in MATRIX:
        run_id = _run_id(fusion, train_target, model)
        print(f"\n[matrix] === {run_id} (fusion={fusion}, train={train_target}, model={model}) ===")

        run_cfg = copy.deepcopy(cfg)
        run_cfg.fusion = {**cfg.fusion, "method": fusion}
        run_cfg.train = {
            **cfg.train,
            "train_target": train_target,
            "eval_target": "high",
            "variants": [{"model": model}],
        }
        # Patch experiment dir for this run's checkpoints
        run_out = out_root / run_id
        run_out.mkdir(parents=True, exist_ok=True)

        # Temporarily redirect exp_dir train output via monkeypatch on cfg.experiment subpath
        orig_exp = cfg.experiment
        nested_exp = f"{orig_exp}/train_matrix/{run_id}"
        run_cfg.experiment = nested_exp

        results = run_training(run_cfg, cohort, rep_roots)
        all_results[run_id] = results

    summary_path = cfg.exp_dir("matrix_comparison.json")
    atomic_write_json(summary_path, {"matrix": all_results})
    return all_results


def format_comparison_md(results: dict[str, Any]) -> str:
    lines = [
        "# H100 full cohort — 8-experiment matrix\n",
        "**Eval metric (all runs):** binary high-risk AUC (`suicide >= 3`)\n",
        "**Train targets:** ordinal (0–6 MSE) or binary (BCE on `y_high`)\n",
        "\n| Run | Fusion | Train | Model | AUC | PR-AUC | F1 | Cohen's d |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for run_id, res in results.items():
        parts = run_id.split("_")
        fusion, train, model = parts[0], parts[1], parts[2]
        summary = res.get("summary", {})
        key = next(iter(summary), None)
        if not key:
            continue
        s = summary[key]
        lines.append(
            f"| {run_id} | {fusion} | {train} | {model} | "
            f"{s['auc_roc']['mean']:.3f} | {s['pr_auc']['mean']:.3f} | "
            f"{s['f1']['mean']:.3f} | {s['cohens_d']['mean']:.3f} |"
        )

    lines.extend(
        [
            "\n## Headline comparison\n",
            "| Hypothesis | Run | AUC |",
            "|---|---|---:|",
        ]
    )
    headline = [
        ("Primary baseline (PCA + ordinal + STM)", "pca_ordinal_stm"),
        ("Attention + ordinal + MTM (scale hypothesis)", "attention_ordinal_mtm"),
        ("Binary control (PCA + binary + STM)", "pca_binary_stm"),
    ]
    for label, rid in headline:
        s = results.get(rid, {}).get("summary", {})
        key = next(iter(s), None) if s else None
        auc = s[key]["auc_roc"]["mean"] if key else float("nan")
        lines.append(f"| {label} | {rid} | {auc:.3f} |")

    return "\n".join(lines)
