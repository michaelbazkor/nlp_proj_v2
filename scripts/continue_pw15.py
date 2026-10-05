"""Continue the pw_1.5 STM run (attention fusion, pos_weight_scale=1.5).

Folds 0 and 1 are already in artifacts/pw_1.5. This fits the same splits,
grid, RMSprop search, and fusion settings for the remaining folds only.

Before training, fold 0 fusion is refit and checked against the saved
fusion_dev_auc so a split or loss mismatch aborts the run.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssr.config import load_config  # noqa: E402
from ssr.data.cohort import build_cohort  # noqa: E402
from ssr.fusion.project import collect_user_blocks  # noqa: E402
from ssr.fusion.registry import fit_fusion_method  # noqa: E402
from ssr.io_utils import atomic_write_json  # noqa: E402
from ssr.models import MTM, STM  # noqa: E402
from ssr.train.cv import PERSONALITY, PSYCHIATRIC, PSYCHOSOCIAL, _make_splits, _zscore_fit  # noqa: E402
from ssr.train.metrics import auc_roc, mtm_loss, stm_loss, summarize_scores  # noqa: E402

FOLD: dict = {}
POS_WEIGHT_SCALE = 1.5
EXPECTED_FOLD0_FUSION_AUC = 0.668806161745828
EXPECTED_FOLD0 = {"n": 201, "n_pos": 26}


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
                    out, ytr[idx], tensors["pers"][idx], tensors["psy"][idx], tensors["psych"][idx]
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


def _child_setup(gpus: list[int], payload: dict) -> None:
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


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    best_t, best_f = 0.5, -1.0
    for t in np.unique(scores):
        f = float(f1_score(y_true, (scores >= t).astype(int), zero_division=0))
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def extra_scores(y_bin: np.ndarray, y_ord: np.ndarray, scores: np.ndarray, dev_y, dev_scores) -> dict:
    dev_thr, dev_f1 = best_f1_threshold(dev_y, dev_scores)
    test_f1 = float(f1_score(y_bin.astype(int), (scores >= dev_thr).astype(int), zero_division=0))
    rho = spearmanr(y_ord, scores).correlation
    return {
        "f1_threshold": 0.5,
        "f1_dev_thr": test_f1,
        "dev_thr": dev_thr,
        "dev_f1_at_thr": dev_f1,
        "spearman_ordinal": float(rho) if rho is not None and np.isfinite(rho) else float("nan"),
    }


def agg(folds: list[dict], name: str) -> dict:
    vals = [m[name] for m in folds if np.isfinite(m.get(name, np.nan))]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "ci95": [float("nan"), float("nan")], "values": []}
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    se = std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": [mean - 1.96 * se, mean + 1.96 * se], "values": vals}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="eli_matrix.yaml")
    ap.add_argument("--reps", default="more stuff/reps")
    ap.add_argument("--out", default="artifacts/pw_1.5")
    ap.add_argument("--folds", default="2,3,4")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpus", default="cpu", help="comma-separated GPU ids, or 'cpu'")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    tcfg = cfg.train
    want = {int(x) for x in args.folds.split(",") if x != ""}
    gpus = []
    if args.gpus not in ("", "cpu", "none") and torch.cuda.is_available():
        gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    n_workers = args.workers

    cohort = build_cohort(cfg, assert_paper=False)
    rep_roots = {
        m["name"]: (ROOT / args.reps / m["name"]).resolve()
        for m in cfg.represent["models"]
    }
    missing = [str(p) for p in rep_roots.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing rep dirs: {missing}")

    user_ids = cohort["UserId"].tolist()
    y_strat = cohort["y_high"].to_numpy()
    y_ord_all = cohort["suicide"].to_numpy()
    print(f"[train] loading {len(user_ids)} users from {args.reps}", flush=True)
    t0 = time.time()
    all_blocks = {}
    for i, uid in enumerate(user_ids, 1):
        all_blocks[uid] = collect_user_blocks(rep_roots, uid)
        if i % 200 == 0:
            print(f"[train] loaded blocks {i}/{len(user_ids)}", flush=True)
    n_blocks = len(next(iter(all_blocks.values())))
    print(
        f"[train] {len(all_blocks)} users x {n_blocks} blocks in {time.time() - t0:.0f}s | "
        f"fusion=attention_pool train_target=high pos_weight_scale={POS_WEIGHT_SCALE} "
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
    attention_cfg = {
        "lr": 0.01,
        "epochs": 500,
        "patience": 50,
        "pos_weight_scale": POS_WEIGHT_SCALE,
        "device": "cpu",
    }
    batch_size = int(tcfg.get("batch_size", 32))
    momentum = float(tcfg.get("momentum", 0.9))
    patience = int(tcfg.get("early_stop_patience", 200))
    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    prior = json.loads(metrics_path.read_text(encoding="utf-8"))
    fold_metrics = list(prior["summary"]["stm_high"]["folds"])
    fusion_folds = list(prior.get("fusion_folds", []))
    done_folds = {int(m["fold"]) for m in fold_metrics}

    splits = list(
        _make_splits(
            y_strat,
            int(tcfg["n_folds"]),
            cfg.seed,
            float(tcfg["train_frac"]),
            float(tcfg["dev_frac"]),
        )
    )
    idx_cohort = cohort.set_index("UserId")

    def ids_of(idx):
        return [user_ids[i] for i in idx]

    # Confirm the saved run's fold 0 split, then refit fusion and compare AUC.
    fold0 = next(s for s in splits if s[0] == 0)
    _, tr0, dv0, te0 = fold0
    te_sub = idx_cohort.loc[ids_of(te0)]
    n_te, n_pos = len(te_sub), int(te_sub["y_high"].sum())
    print(f"[verify] fold 0 test n={n_te} n_pos={n_pos}", flush=True)
    if n_te != EXPECTED_FOLD0["n"] or n_pos != EXPECTED_FOLD0["n_pos"]:
        raise SystemExit("fold 0 split does not match the saved pw_1.5 run; refusing to continue")

    tf = time.time()
    fusion0 = fit_fusion_method(
        "attention_pool",
        [all_blocks[u] for u in ids_of(tr0)],
        target_dim=1024,
        dev_blocks=[all_blocks[u] for u in ids_of(dv0)],
        train_y=idx_cohort.loc[ids_of(tr0)]["y_high"].to_numpy().astype(np.float32),
        dev_y_eval=idx_cohort.loc[ids_of(dv0)]["y_high"].to_numpy().astype(np.float32),
        attention_cfg=attention_cfg,
        seed=cfg.seed + 0,
        train_loss="bce",
    )
    got = float(fusion0.fusion_dev_auc)
    print(
        f"[verify] fold 0 fusion_dev_auc={got:.6f} expected={EXPECTED_FOLD0_FUSION_AUC:.6f} "
        f"epochs_run={fusion0.epochs_run} ({time.time() - tf:.1f}s)",
        flush=True,
    )
    if abs(got - EXPECTED_FOLD0_FUSION_AUC) > 0.015:
        raise SystemExit("fold 0 fusion AUC does not match pw_1.5; refusing to train new folds")
    if args.verify_only:
        print("[verify] ok", flush=True)
        return

    ctx = mp.get_context("spawn")
    kind = "stm"
    key = "stm_high"
    for fold_i, tr_idx, dv_idx, te_idx in splits:
        if fold_i not in want:
            continue
        if fold_i in done_folds:
            print(f"[train] {key} fold {fold_i} already present, skipping", flush=True)
            continue
        ids = {"tr": ids_of(tr_idx), "dv": ids_of(dv_idx), "te": ids_of(te_idx)}
        tf = time.time()
        fusion = fit_fusion_method(
            "attention_pool",
            [all_blocks[u] for u in ids["tr"]],
            target_dim=1024,
            dev_blocks=[all_blocks[u] for u in ids["dv"]],
            train_y=idx_cohort.loc[ids["tr"]]["y_high"].to_numpy().astype(np.float32),
            dev_y_eval=idx_cohort.loc[ids["dv"]]["y_high"].to_numpy().astype(np.float32),
            attention_cfg=attention_cfg,
            seed=cfg.seed + fold_i,
            train_loss="bce",
        )
        fusion_rec = {
            "fold": fold_i,
            "method": "attention_pool",
            "seconds": round(time.time() - tf, 1),
            "in_dim": 1024,
            "k_per_block": 0,
            "fusion_dev_auc": fusion.fusion_dev_auc,
            "epochs_run": fusion.epochs_run,
            "mean_attention_dev": fusion.mean_attention_dev,
        }
        print(
            f"[train] fold {fold_i}: fusion(attention_pool) {fusion_rec['seconds']:.0f}s "
            f"in_dim=1024 k_block=0 fusion_dev_auc={fusion.fusion_dev_auc:.3f} "
            f"train={len(ids['tr'])} dev={len(ids['dv'])} test={len(ids['te'])} grid={len(param_grid)}",
            flush=True,
        )

        def pack(split, fusion=fusion):
            X = fusion.transform_many([all_blocks[u] for u in ids[split]])
            sub = idx_cohort.loc[ids[split]]
            return X, sub["y_high"].to_numpy().astype(np.float32), sub["suicide"].to_numpy(), sub

        Xtr, ytr, _, str_ = pack("tr")
        Xdv, ydv, _, _ = pack("dv")
        Xte, yte, yord, _ = pack("te")
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
        payload = {
            "kind": kind, "fold_i": fold_i, "seed": cfg.seed, "batch_size": batch_size,
            "momentum": momentum, "patience": patience, "fd": fd,
        }
        tg = time.time()
        order = sorted(
            range(len(param_grid)),
            key=lambda i: (
                int(param_grid[i]["epochs"]),
                int(param_grid[i]["n_neurons"]),
                int(param_grid[i]["n_layers"]),
            ),
            reverse=True,
        )
        # Wide nets go to the single GPU, same cutoff as the original parallel trainer.
        # Narrow nets stay on CPU. One CUDA context avoids overflowing the 4 GB card.
        gpu_idx = [i for i in order if gpus and pick_device(param_grid[i]) == "gpu"]
        cpu_idx = [i for i in order if i not in set(gpu_idx)]
        print(
            f"[train] {key} fold {fold_i}: {len(gpu_idx)} GPU configs, {len(cpu_idx)} CPU configs",
            flush=True,
        )
        state = {"done": 0, "best_auc": -1.0, "best_idx": None}
        lock = threading.Lock()

        def note(cfg_idx: int, dev_auc: float) -> None:
            with lock:
                state["done"] += 1
                if np.isfinite(dev_auc) and dev_auc > state["best_auc"]:
                    state["best_auc"] = float(dev_auc)
                    state["best_idx"] = cfg_idx
                done = state["done"]
                best = state["best_auc"]
                total = len(param_grid)
                if done % 50 == 0 or done == total:
                    el = time.time() - tg
                    print(
                        f"[train] {key} fold {fold_i}: {done}/{total} "
                        f"({el:.0f}s, {el / done:.1f}s/cfg, ETA {(total - done) * el / done / 60:.1f}m) "
                        f"best_dev_auc={best:.3f}",
                        flush=True,
                    )

        def gpu_loop() -> None:
            device = f"cuda:{gpus[0]}"
            tensors = to_tensors(fd, device)
            for i in gpu_idx:
                dev_auc, _ = fit_one(
                    kind, param_grid[i], tensors,
                    config_seed(cfg.seed, fold_i, kind, i),
                    batch_size, momentum, patience, device,
                )
                note(i, dev_auc)
            del tensors
            torch.cuda.empty_cache()

        gpu_thread = None
        if gpu_idx:
            gpu_thread = threading.Thread(target=gpu_loop, daemon=True)
            gpu_thread.start()
        if cpu_idx:
            with ProcessPoolExecutor(
                max_workers=min(n_workers, len(cpu_idx)),
                mp_context=ctx,
                initializer=_child_setup,
                initargs=([], payload),
            ) as ex:
                futs = [ex.submit(worker, (i, param_grid[i])) for i in cpu_idx]
                for fut in as_completed(futs):
                    note(*fut.result())
        if gpu_thread is not None:
            gpu_thread.join()
        best_idx, best_auc = state["best_idx"], state["best_auc"]

        params = param_grid[best_idx]
        dev_dev = f"cuda:{gpus[0]}" if (gpus and pick_device(params) == "gpu") else "cpu"
        tensors = to_tensors(fd, dev_dev)
        dev_auc2, model = fit_one(
            kind, params, tensors,
            config_seed(cfg.seed, fold_i, kind, best_idx),
            batch_size, momentum, patience, dev_dev,
        )
        with torch.no_grad():
            te_logits = model(tensors["Xtr"].new_tensor(Xte))["suicide_logit"].float().cpu().numpy()
            dv_logits = model(tensors["Xdv"])["suicide_logit"].float().cpu().numpy()
        scores = 1 / (1 + np.exp(-np.clip(te_logits, -60, 60)))
        dev_scores = 1 / (1 + np.exp(-np.clip(dv_logits, -60, 60)))
        metrics = summarize_scores(yte, scores)
        metrics.update({
            "fold": fold_i,
            "dev_auc": best_auc,
            "dev_auc_refit": dev_auc2,
            **extra_scores(yte, yord, scores, ydv, dev_scores),
            **{f"p_{k}": v for k, v in params.items()},
        })
        fold_metrics.append(metrics)
        fusion_folds.append(fusion_rec)
        model_cpu = model.cpu()
        torch.save(
            {"state_dict": model_cpu.state_dict(), "params": params, "metrics": metrics},
            out_dir / f"{key}_fold{fold_i}.pt",
        )
        torch.save(
            {
                "state_dict": fusion.state_dict,
                "group_keys": fusion.group_keys,
                "group_dims": fusion.group_dims,
                "block_keys": fusion.block_keys,
                "means": fusion.means,
                "stds": fusion.stds,
                "target_dim": fusion.target_dim,
                "fusion_dev_auc": fusion.fusion_dev_auc,
                "epochs_run": fusion.epochs_run,
            },
            out_dir / f"fusion_fold{fold_i}.pt",
        )
        print(
            f"[train] {key} fold {fold_i} DONE test AUC={metrics['auc_roc']:.3f} "
            f"PR={metrics['pr_auc']:.3f} F1={metrics['f1']:.3f} "
            f"F1@dev={metrics['f1_dev_thr']:.3f} d={metrics['cohens_d']:.3f} "
            f"rho={metrics['spearman_ordinal']:.3f} params={params} "
            f"({(time.time() - tg) / 60:.1f}m)",
            flush=True,
        )
        del tensors
        if gpus:
            torch.cuda.empty_cache()
        _write(metrics_path, prior, fold_metrics, fusion_folds, n_blocks, n_workers)

    _write(metrics_path, prior, fold_metrics, fusion_folds, n_blocks, n_workers)
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))["summary"]["stm_high"]["auc_roc"]
    print(
        f"\n=== SUMMARY [attention_pool | high | pos_weight_scale={POS_WEIGHT_SCALE}] ===\n"
        f"  stm_high: AUC={summary['mean']:.3f} "
        f"[{summary['ci95'][0]:.3f}, {summary['ci95'][1]:.3f}] "
        f"folds={ [round(v, 3) for v in summary['values']] }",
        flush=True,
    )


def _write(path: Path, prior: dict, fold_metrics: list[dict], fusion_folds: list[dict], n_blocks: int, n_workers: int) -> None:
    fold_metrics = sorted(fold_metrics, key=lambda m: int(m["fold"]))
    fusion_folds = sorted(fusion_folds, key=lambda m: int(m["fold"]))
    names = ["auc_roc", "pr_auc", "f1", "f1_dev_thr", "cohens_d", "spearman_ordinal"]
    summary = {name: agg(fold_metrics, name) for name in names}
    summary["mae_ordinal"] = prior["summary"]["stm_high"].get("mae_ordinal", {})
    summary["qwk_ordinal"] = prior["summary"]["stm_high"].get("qwk_ordinal", {})
    summary["folds"] = fold_metrics
    out = {
        "fusion_method": "attention_pool",
        "train_target": "high",
        "eval_target": "high",
        "summary": {"stm_high": summary},
        "fusion_folds": fusion_folds,
        "meta": {
            **prior.get("meta", {}),
            "n_blocks": n_blocks,
            "workers": n_workers,
            "pos_weight_scale": POS_WEIGHT_SCALE,
            "attention_cfg": {
                "lr": 0.01,
                "epochs": 500,
                "patience": 50,
                "pos_weight_scale": POS_WEIGHT_SCALE,
            },
        },
    }
    atomic_write_json(path, out)


if __name__ == "__main__":
    main()
