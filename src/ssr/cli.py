"""CLI entrypoint: ssr captions|corpus|represent|train|all|report --config ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without install: python -m ssr.cli
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssr.config import Config, load_config
from ssr.data.cohort import build_cohort
from ssr.data.corpus import build_corpora
from ssr.data.posts import build_posts, cap_posts_per_user
from ssr.io_utils import atomic_write_text, exists_nonempty


def _preflight(cfg: Config) -> None:
    """Warn about POC-contaminated caches that would break a full-cohort run."""
    n_users = cfg.cohort.get("n_users")
    posts_path = cfg.art("posts.parquet")
    posts_meta = cfg.art("posts_meta.json")
    if n_users is None and exists_nonempty(posts_meta):
        meta = json.loads(posts_meta.read_text())
        n = int(meta.get("n_users", 0))
        if n < 1000:
            print(
                f"[preflight] WARNING: posts cache has only {n} users "
                f"({posts_path}). Full-cohort runs need a full posts rebuild: "
                f"rm {posts_path} {posts_meta} then re-run with --force."
            )

    corp_path = cfg.exp_dir("corpora.parquet")
    if exists_nonempty(corp_path):
        try:
            import pandas as pd

            n_corp = len(pd.read_parquet(corp_path))
            if n_users is None and n_corp <= 30:
                print(
                    f"[preflight] WARNING: experiment '{cfg.experiment}' corpora "
                    f"has n={n_corp} (POC-sized). Use --force or a distinct "
                    f"experiment name (e.g. real_standard)."
                )
        except Exception:
            pass

    if cfg.corpus.get("include_images", True):
        from ssr.caption.run import caption_paths

        _, meta_path = caption_paths(cfg)
        if exists_nonempty(meta_path):
            meta = json.loads(meta_path.read_text())
            want = cfg.images["caption_model"]
            got = meta.get("model")
            if got and got != want:
                print(
                    f"[preflight] WARNING: caption cache model {got!r} != "
                    f"config {want!r}. Re-run captions --force."
                )


def cmd_cohort(cfg, *, force: bool = False):
    c = build_cohort(
        cfg,
        assert_paper=cfg.cohort.get("n_users") is None,
        force=force,
    )
    print(f"[cohort] n={len(c)} high={int(c.y_high.sum())} ({100*c.y_high.mean():.1f}%)")
    print(c[["UserId", "suicide", "y_high", "status_posts"]].head(5).to_string(index=False))
    return c


def cmd_posts(cfg, cohort, *, force: bool = False):
    posts = build_posts(cfg, set(cohort["UserId"]), force=force)
    posts = cap_posts_per_user(
        posts,
        cfg.posts.get("max_posts_per_user"),
        int(cfg.posts.get("shuffle_seed", cfg.seed)),
    )
    # overwrite capped cache for this experiment subset
    out = cfg.exp_dir("posts_capped.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    posts.to_parquet(out, index=False)
    print(f"[posts] rows={len(posts)} users={posts.UserId.nunique()} "
          f"with_text={int(posts.has_text.sum())} with_image={int(posts.has_image.sum())}")
    print(posts[["UserId", "PostType", "free_text", "blob_guid"]].head(3).to_string(index=False))
    return posts


def cmd_captions(cfg, posts, *, force: bool = False):
    from ssr.caption.run import run_captions

    caps = run_captions(cfg, posts, force=force)
    print(f"[captions] n={len(caps)}")
    for i, (g, c) in enumerate(caps.items()):
        if i >= 3:
            break
        print(f"  {str(g)[:40]}... -> {c[:120]}")
    return caps


def cmd_corpus(cfg, cohort, posts, captions, *, force: bool = False):
    corpora = build_corpora(cfg, cohort, posts, captions, force=force)
    print(f"[corpus] users={len(corpora)} mean_posts={corpora.n_posts.mean():.1f} "
          f"mean_chars={corpora.n_chars.mean():.0f}")
    sample = corpora.iloc[0]
    snippet = sample.corpus[:500].replace("\n", " ")
    print(f"  sample UserId={sample.UserId} n_posts={sample.n_posts}")
    print(f"  snippet: {snippet}...")
    return corpora


def cmd_represent(cfg, corpora, *, force: bool = False):
    from ssr.represent.extract import run_representations

    roots = run_nxt(cfg, corpora, force=force)
    for name, root in roots.items():
        n = len(list(root.glob("*.npz")))
        print(f"[represent] {name}: {n} user files in {root}")
    return roots


def cmd_train(cfg, cohort, rep_roots):
    from ssr.train.cv import run_training

    results = run_training(cfg, cohort, rep_roots)
    print("\n=== SUMMARY ===")
    for key, s in results["summary"].items():
        auc = s["auc_roc"]
        print(f"  {key}: AUC={auc['mean']:.3f} "
              f"[{auc['ci95'][0]:.3f}, {auc['ci95'][1]:.3f}]  "
              f"PR={s['pr_auc']['mean']:.3f}  F1={s['f1']['mean']:.3f}  "
              f"d={s['cohens_d']['mean']:.3f}")
    return results


def cmd_report(cfg, cohort, posts, corpora, captions, results=None):
    """Write reports/{experiment}/stage_examples.md with real data at every stage."""
    from ssr.fusion.project import collect_user_blocks, fit_fusion
    from ssr.represent.store import load_user_reps

    lines = [f"# Stage examples ({cfg.experiment})\n"]
    uid = cohort.iloc[0]["UserId"]
    lines.append("## 0. Cohort row\n")
    row = cohort[cohort.UserId == uid].iloc[0]
    lines.append("```")
    lines.append(row[["UserId", "suicide", "y_high", "status_posts",
                      "age", "gender_label", "PHQ9", "GAD", "BFI_N", "Lonely"]].to_string())
    lines.append("```\n")

    lines.append("## 1. Raw cleaned posts (sample)\n")
    up = posts[posts.UserId == uid].head(3)
    for _, p in up.iterrows():
        lines.append(f"- **PostType={p.PostType}** has_image={p.has_image}")
        lines.append(f"  text: `{str(p.free_text)[:200]}`")
        key = p.get("image_key") if "image_key" in p.index else None
        if (key is None or (isinstance(key, float) and __import__("math").isnan(key))) and "blob_guid" in p.index:
            key = p.get("blob_guid")
        if key and not (isinstance(key, float) and __import__("math").isnan(key)):
            lines.append(f"  image_key: `{key}`")
            if captions and str(key) in captions:
                lines.append(f"  caption: `[image] {captions[str(key)][:200]}`")
    lines.append("")

    lines.append("## 2. Assembled corpus (truncated)\n")
    corp = corpora[corpora.UserId == uid].iloc[0]
    lines.append(f"n_posts={corp.n_posts} n_images={corp.n_images} n_chars={corp.n_chars}\n")
    lines.append("```")
    lines.append(corp.corpus[:1500])
    lines.append("```\n")

    # Representations
    rep_roots = {}
    for mcfg in cfg.represent["models"]:
        root = cfg.exp_dir("reps", mcfg["name"])
        if root.exists():
            rep_roots[mcfg["name"]] = root

    if rep_roots:
        lines.append("## 3. Raw representation blocks\n")
        safe = str(uid).replace("/", "_").replace(" ", "")
        for name, root in rep_roots.items():
            path = root / f"{safe}.npz"
            if not path.exists():
                continue
            arrs = load_user_reps(path)
            lines.append(f"### Model `{name}`\n")
            for k, v in list(arrs.items())[:8]:
                if k.startswith("__"):
                    continue
                lines.append(f"- `{k}`: shape={v.shape} mean={v.mean():.4f} std={v.std():.4f}")
            lines.append("")

        lines.append("## 4. Fused 1024-d vector (fit on cohort for illustration)\n")
        try:
            blocks_all = [collect_user_blocks(rep_roots, u) for u in cohort.UserId.tolist()]
            fusion = fit_fusion(blocks_all, target_dim=int(cfg.fusion["target_dim"]))
            vec = fusion.transform(collect_user_blocks(rep_roots, uid))
            lines.append(f"k_per_block={fusion.k_per_block} n_blocks={len(fusion.block_keys)} "
                         f"out_dim={vec.shape[0]}\n")
            lines.append(f"vector[:16] = `{vec[:16].tolist()}`\n")
            lines.append(f"vector mean={vec.mean():.4f} std={vec.std():.4f} l2={float((vec**2).sum()**0.5):.4f}\n")
        except Exception as e:
            lines.append(f"(fusion example failed: {e})\n")

    if results is None:
        metrics_path = cfg.exp_dir("train", "metrics.json")
        if metrics_path.exists():
            results = json.loads(metrics_path.read_text())

    if results is not None:
        lines.append("## 5. CV metrics summary\n")
        lines.append("| Variant | AUC-ROC | PR-AUC | F1 | Cohen's d |")
        lines.append("|---------|---------|--------|----|-----------|")
        for key, s in results.get("summary", {}).items():
            lines.append(
                f"| {key} | {s['auc_roc']['mean']:.3f} | {s['pr_auc']['mean']:.3f} | "
                f"{s['f1']['mean']:.3f} | {s['cohens_d']['mean']:.3f} |"
            )
        lines.append("")

    path = cfg.report(cfg.experiment, "stage_examples.md")
    atomic_write_text(path, "\n".join(lines))
    print(f"[report] wrote {path}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ssr")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument(
        "command",
        choices=["cohort", "posts", "captions", "corpus", "represent", "train", "report", "all"],
    )
    ap.add_argument("--skip-captions", action="store_true", help="Skip VLM captioning (text-only)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    print(f"[ssr] experiment={cfg.experiment} config={args.config}")
    _preflight(cfg)

    if args.command == "cohort":
        cmd_cohort(cfg, force=args.force)
        return

    cohort = cmd_cohort(cfg, force=args.force)
    if args.command == "posts":
        cmd_posts(cfg, cohort, force=args.force)
        return

    posts = cmd_posts(cfg, cohort, force=args.force)

    captions = {}
    need_images = bool(cfg.corpus.get("include_images", True)) and not args.skip_captions
    if args.command in ("captions", "all", "corpus", "represent", "train", "report"):
        if args.skip_captions or not cfg.corpus.get("include_images", True):
            print("[captions] skipped (text-only / --skip-captions)")
        elif args.command == "captions" or (
            args.command == "all" and need_images
        ):
            captions = cmd_captions(cfg, posts, force=args.force)
        else:
            from ssr.caption.run import load_captions

            # Fail closed when images are required for corpus/represent/train/report
            require = args.command in ("corpus", "represent", "train", "all") and need_images
            captions = load_captions(cfg, require=require)
            if require and not captions:
                raise RuntimeError(
                    "include_images=true but no captions loaded for the configured VLM. "
                    "Run captions first (with --force if switching models)."
                )

    if args.command == "captions":
        return

    if args.command in ("corpus", "represent", "train", "report", "all"):
        corpora = cmd_corpus(cfg, cohort, posts, captions, force=args.force)

    if args.command == "corpus":
        return

    if args.command in ("represent", "train", "report", "all"):
        rep_roots = cmd_represent(cfg, corpora, force=args.force)

    if args.command == "represent":
        return

    results = None
    if args.command in ("train", "all"):
        results = cmd_train(cfg, cohort, rep_roots)

    if args.command in ("report", "all"):
        cmd_report(cfg, cohort, posts, corpora, captions, results)


if __name__ == "__main__":
    main()
