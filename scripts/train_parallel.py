"""Parallel STM/MTM grid search for the full-cohort run.

`ssr.train.cv.run_training` walks the 504-point grid serially on CPU for each of
5 folds x 2 variants (5040 fits), which does not finish in reasonable time. This
driver keeps the same splits, fusion, losses, early stopping and metric
definitions, but:
  * holds each fold's tensors resident on a GPU,
  * spreads grid points over several worker processes per GPU,
  * refits only the winning configuration to materialize weights (workers return
    scores, not state dicts), using a per-configuration seed so the refit is exact.

Output is written to the same `artifacts/{experiment}/train/metrics.json` layout
that `ssr.cli report` consumes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssr.config import load_config  # noqa: E402
from ssr.data.cohort import build_cohort  # noqa: E402
from ssr.fusion.project import collect_user_blocks, fit_fusion  # noqa: E402
from ssr.io_utils import atomic_write_json  # noqa: E402
from ssr.models import MTM, STM  # noqa: E402
from ssr.train.cv import PERSONALITY, PSYCHIATRIC, PSYCHOSOCIAL, _make_splits, _zscore_fit  # noqa: E402
from ssr.train.metrics import auc_roc, mtm_loss, stm_loss, summarize_scores  # noqa: E402

# Set in the parent before the pool forks; children inherit copy-on-write.
FOLD: dict = {}


def config_seed(base: int, fold_i: int, variant: str, cfg_idx: int) -> int:
    return (base * 1_000_003 + fold_i * 10_007 + (0 if variant == "stm" else 5) * 101 + cfg_idx) % (2**31 - 1)


def build_model(kind: str, in_dim: int, n_layers: int, n_neurons: int, activation: str):
    if kind == "stm":
        return STM(in_dim, n_layers, n_neurons, activation), False
    return MTM(in_dim, n_layers, n_neurons, activation), True


def fit_one(
    kind: str,
    params: dict,
    tensors: dict,
    seed: int,
    batch_size: int,
    momentum: float,
    patience: int,
    device: str,
):
    """Train one grid point; return (best_dev_auc, trained_model)."""
    torch.manual_seed(seed)
    model, is_mtm = build_model(
        kind,
        tensors["Xtr"].shape[1],
        int(params["n_layers"]),
        int(params["n_neurons"]),
        params["activation"],
    )
    model = model.to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=float(params["lr"]), momentum=momentum)

    Xtr, ytr = tensors["Xtr"], tensors["ytr"]
    Xdv = tensors["Xdv"]
    ydv_np = tensors["ydv_np"]
    n = Xtr.shape[0]
    bs = min(batch_size, n)

    best_auc, best_state, bad = -1.0, None, 0
    for _ in range(int(params["epochs"])):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            opt.zero_grad()
            out = model(Xtr[idx])
            if is_mtm:
                loss, _ = mtm_loss(
                    out,
                    ytr[idx],
                    tensors["pers"][idx],
                    tensors["psy"][idx],
                    tensors["psych"][idx],
                )
            else:
                loss = stm_loss(out["suicide_logit"], ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(Xdv)["suicide_logit"].float().cpu().numpy()
        auc = auc_roc(ydv_np, 1 / (1 + np.exp(-np.clip(logits, -60, 60))))
        if np.isfinite(auc) and auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return (float(best_auc) if best_auc >= 0 else float("nan")), model


# Measured on this node: wide nets are ~5x faster on GPU, narrow nets are
# kernel-launch bound and faster on a single CPU thread.
GPU_MIN_NEURONS = 512


def pick_device(params: dict) -> str:
    return "gpu" if int(params["n_neurons"]) >= GPU_MIN_NEURONS else "cpu"


def worker(job: tuple[int, dict]) -> tuple[int, float]:
    cfg_idx, params = job
    which = pick_device(params)
    device = FOLD["gpu_device"] if which == "gpu" and FOLD["gpu_device"] else "cpu"
    dev_auc, _ = fit_one(
        FOLD["kind"],
        params,
        FOLD["tensors_gpu"] if device != "cpu" else FOLD["tensors_cpu"],
        config_seed(FOLD["seed"], FOLD["fold_i"], FOLD["kind"], cfg_idx),
        FOLD["batch_size"],
        FOLD["momentum"],
        FOLD["patience"],
        device,
    )
    return cfg_idx, dev_auc


def to_tensors(fd: dict, device: str) -> dict:
    t = {
        "Xtr": torch.tensor(fd["Xtr"], dtype=torch.float32, device=device),
        "ytr": torch.tensor(fd["ytr"], dtype=torch.float32, device=device),
        "Xdv": torch.tensor(fd["Xdv"], dtype=torch.float32, device=device),
        "ydv_np": fd["ydv"],
    }
    for k in ("pers", "psy", "psych"):
        t[k] = torch.tensor(fd[k], dtype=torch.float32, device=device)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--limit-grid", type=int, default=0, help="debug: first N grid points")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg.train
    gpus = [int(g) for g in args.gpus.split(",") if g != ""] if torch.cuda.is_available() else []
    n_workers = args.workers if gpus else max(1, (os.cpu_count() or 8) // 2)

    cohort = build_cohort(cfg, assert_paper=False)
    rep_roots = {m["name"]: cfg.exp_dir("reps", m["name"]) for m in cfg.represent["models"]}
    rep_roots = {k: v for k, v in rep_roots.items() if v.exists()}
    print(f"[train] rep roots: {list(rep_roots)}", flush=True)

    user_ids = cohort["UserId"].tolist()
    y_strat = cohort["y_high"].to_numpy()
    t0 = time.time()
    all_blocks = {}
    for i, uid in enumerate(user_ids, 1):
        all_blocks[uid] = collect_user_blocks(rep_roots, uid)
        if i % 200 == 0:
            print(f"[train] loaded blocks {i}/{len(user_ids)}", flush=True)
    n_blocks = len(next(iter(all_blocks.values())))
    print(
        f"[train] {len(all_blocks)} users x {n_blocks} blocks in {time.time() - t0:.0f}s | "
        f"workers={n_workers} gpus={gpus}",
        flush=True,
    )

    grid = tcfg["grid"]
    param_grid = [
        {"n_layers": a, "n_neurons": b, "activation": c, "lr": d, "epochs": e}
        for a, b, c, d, e in itertools.product(
            grid["n_layers"], grid["n_neurons"], grid["activation"], grid["lr"], grid["epochs"]
        )
    ]
    if args.limit_grid:
        param_grid = param_grid[: args.limit_grid]
    target_dim = int(cfg.fusion.get("target_dim", 1024))
    batch_size = int(tcfg.get("batch_size", 32))
    momentum = float(tcfg.get("momentum", 0.9))
    patience = int(tcfg.get("early_stop_patience", 50))
    out_dir = cfg.exp_dir("train")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {"summary": {}, "meta": {
        "n_users": len(user_ids), "n_blocks": n_blocks, "grid_size": len(param_grid),
        "rep_models": list(rep_roots), "workers": n_workers,
    }}
    # spawn (not fork): the parent touches CUDA when refitting each fold's winner.
    ctx = mp.get_context("spawn")

    for variant in tcfg["variants"]:
        kind = variant["model"]
        key = f"{kind}_high"
        fold_metrics = []
        for fold_i, tr_idx, dv_idx, te_idx in _make_splits(
            y_strat, int(tcfg["n_folds"]), cfg.seed,
            float(tcfg["train_frac"]), float(tcfg["dev_frac"]),
        ):
            ids = {n: [user_ids[i] for i in idx] for n, idx in
                   (("tr", tr_idx), ("dv", dv_idx), ("te", te_idx))}
            tf = time.time()
            fusion = fit_fusion([all_blocks[u] for u in ids["tr"]], target_dim=target_dim)
            idx_cohort = cohort.set_index("UserId")

            def pack(split):
                X = np.stack([fusion.transform(all_blocks[u]) for u in ids[split]], axis=0)
                sub = idx_cohort.loc[ids[split]]
                return X, sub["y_high"].to_numpy().astype(np.float32), sub

            Xtr, ytr, str_ = pack("tr")
            Xdv, ydv, sdv = pack("dv")
            Xte, yte, ste = pack("te")
            aux_tr = {
                "personality": str_[PERSONALITY].to_numpy().astype(np.float32),
                "psychosocial": str_[PSYCHOSOCIAL].to_numpy().astype(np.float32),
                "psychiatric": str_[PSYCHIATRIC].to_numpy().astype(np.float32),
            }
            mu_sd = {k: _zscore_fit(v) for k, v in aux_tr.items()}
            fd = {
                "Xtr": Xtr, "ytr": ytr, "Xdv": Xdv, "ydv": ydv,
                "pers": (aux_tr["personality"] - mu_sd["personality"][0]) / mu_sd["personality"][1],
                "psy": (aux_tr["psychosocial"] - mu_sd["psychosocial"][0]) / mu_sd["psychosocial"][1],
                "psych": (aux_tr["psychiatric"] - mu_sd["psychiatric"][0]) / mu_sd["psychiatric"][1],
            }
            print(
                f"[train] {key} fold {fold_i}: fusion {time.time() - tf:.0f}s "
                f"in_dim={Xtr.shape[1]} k_block={fusion.k_per_block} "
                f"train={len(ids['tr'])} dev={len(ids['dv'])} test={len(ids['te'])} "
                f"grid={len(param_grid)}",
                flush=True,
            )

            payload = {
                "kind": kind, "fold_i": fold_i, "seed": cfg.seed, "batch_size": batch_size,
                "momentum": momentum, "patience": patience, "fd": fd,
            }
            tg = time.time()
            best_idx, best_auc = None, -1.0
            done = 0
            with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=ctx,
                initializer=_child_setup, initargs=(gpus, payload),
            ) as ex:
                # Longest-first so the slowest configs do not become the tail.
                order = sorted(
                    range(len(param_grid)),
                    key=lambda i: (
                        int(param_grid[i]["epochs"]),
                        int(param_grid[i]["n_neurons"]),
                        int(param_grid[i]["n_layers"]),
                    ),
                    reverse=True,
                )
                futs = [ex.submit(worker, (i, param_grid[i])) for i in order]
                for fut in as_completed(futs):
                    cfg_idx, dev_auc = fut.result()
                    done += 1
                    if np.isfinite(dev_auc) and dev_auc > best_auc:
                        best_auc, best_idx = dev_auc, cfg_idx
                    if done % 50 == 0 or done == len(futs):
                        el = time.time() - tg
                        print(
                            f"[train] {key} fold {fold_i}: {done}/{len(futs)} "
                            f"({el:.0f}s, {el / done:.1f}s/cfg, ETA {(len(futs) - done) * el / done / 60:.1f}m) "
                            f"best_dev_auc={best_auc:.3f}",
                            flush=True,
                        )

            # Refit the winner in-process (same seed => same weights) to score test.
            params = param_grid[best_idx]
            # Must match the worker's device so the seeded refit reproduces it.
            dev_dev = f"cuda:{gpus[0]}" if (gpus and pick_device(params) == "gpu") else "cpu"
            tensors = to_tensors(fd, dev_dev)
            dev_auc2, model = fit_one(
                kind, params, tensors,
                config_seed(cfg.seed, fold_i, kind, best_idx),
                batch_size, momentum, patience, dev_dev,
            )
            with torch.no_grad():
                logits = model(torch.tensor(Xte, dtype=torch.float32, device=dev_dev))
                scores = 1 / (1 + np.exp(-logits["suicide_logit"].float().cpu().numpy()))
            metrics = summarize_scores(yte, scores)
            metrics.update({
                "fold": fold_i, "dev_auc": best_auc, "dev_auc_refit": dev_auc2,
                **{f"p_{k}": v for k, v in params.items()},
            })
            fold_metrics.append(metrics)
            torch.save(
                {"state_dict": model.state_dict(), "params": params, "metrics": metrics},
                out_dir / f"{key}_fold{fold_i}.pt",
            )
            print(
                f"[train] {key} fold {fold_i} DONE test AUC={metrics['auc_roc']:.3f} "
                f"PR={metrics['pr_auc']:.3f} F1={metrics['f1']:.3f} d={metrics['cohens_d']:.3f} "
                f"params={params} ({(time.time() - tg) / 60:.1f}m)",
                flush=True,
            )
            del tensors
            if gpus:
                torch.cuda.empty_cache()

        def agg(name):
            vals = [m[name] for m in fold_metrics if np.isfinite(m.get(name, np.nan))]
            if not vals:
                return {"mean": float("nan"), "std": float("nan"), "ci95": [float("nan")] * 2}
            arr = np.asarray(vals, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            se = std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
            return {"mean": mean, "std": std,
                    "ci95": [mean - 1.96 * se, mean + 1.96 * se], "values": vals}

        results["summary"][key] = {
            "auc_roc": agg("auc_roc"), "pr_auc": agg("pr_auc"),
            "f1": agg("f1"), "cohens_d": agg("cohens_d"), "folds": fold_metrics,
        }
        atomic_write_json(out_dir / "metrics.json", results)

    atomic_write_json(out_dir / "metrics.json", results)
    print("\n=== SUMMARY ===")
    for k, s in results["summary"].items():
        a = s["auc_roc"]
        print(f"  {k}: AUC={a['mean']:.3f} [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}] "
              f"PR={s['pr_auc']['mean']:.3f} F1={s['f1']['mean']:.3f} d={s['cohens_d']['mean']:.3f}")
    print(json.dumps(results["meta"], indent=2))


def _child_setup(gpus: list[int], payload: dict) -> None:
    """Spawn-safe worker init: hold the fold on CPU and on this worker's GPU."""
    torch.set_num_threads(1)
    try:
        slot = int(mp.current_process().name.rsplit("-", 1)[-1]) - 1
    except ValueError:
        slot = 0
    FOLD.update(payload)
    FOLD["tensors_cpu"] = to_tensors(payload["fd"], "cpu")
    if gpus:
        FOLD["gpu_device"] = f"cuda:{gpus[slot % len(gpus)]}"
        FOLD["tensors_gpu"] = to_tensors(payload["fd"], FOLD["gpu_device"])
    else:
        FOLD["gpu_device"] = None
        FOLD["tensors_gpu"] = FOLD["tensors_cpu"]


if __name__ == "__main__":
    main()
