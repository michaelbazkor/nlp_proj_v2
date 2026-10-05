"""Train the paper-style MTM on the pw_1.5 attention representations.

Same cohort, splits, 504-point grid, and attention fusion (pos_weight_scale 1.5)
as the STM run. Each stage still predicts the paper's targets, but the value
passed into the next stage is tanh of that prediction so a diverging medical
head cannot drown the shared suicide representation.

Progress is durable. Each finished grid point is appended and fsynced before
the next one starts. Rerun this script with the same arguments to resume
after a shutdown; completed folds and completed grid points are skipped.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import continue_pw15 as base  # noqa: E402
from ssr.config import load_config  # noqa: E402
from ssr.data.cohort import build_cohort  # noqa: E402
from ssr.fusion.attention import AttentionFusionFit  # noqa: E402
from ssr.fusion.project import collect_user_blocks  # noqa: E402
from ssr.fusion.registry import fit_fusion_method  # noqa: E402
from ssr.train.cv import PERSONALITY, PSYCHIATRIC, PSYCHOSOCIAL, _make_splits, _zscore_fit  # noqa: E402

POS_WEIGHT_SCALE = base.POS_WEIGHT_SCALE
LOG_PATH: Path | None = None


def log(msg: str) -> None:
    print(msg, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()


def durable_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=2, default=str).encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, default=str) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def resume_roots(out_dir: Path) -> list[Path]:
    """Project copy plus a local-disk copy that OneDrive does not lock."""
    local = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "nlp_proj_v2" / out_dir.name
    project = out_dir / "mtm_resume"
    for p in (local, project):
        p.mkdir(parents=True, exist_ok=True)
    return [local, project]


def write_both(paths: list[Path], writer, relative: str, obj: dict) -> None:
    errors = []
    for root in paths:
        try:
            writer(root / relative, obj)
        except OSError as exc:
            errors.append(exc)
            log(f"[resume] write failed for {root / relative}: {exc}")
    if len(errors) == len(paths):
        raise errors[0]


def durable_torch(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    torch.save(obj, tmp)
    # Windows rejects fsync on a read-only handle, so reopen for write.
    fd = os.open(tmp, os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    import ctypes

    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if handle:
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    return False


def acquire_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                pid = -1
            if pid_alive(pid):
                raise SystemExit(f"MTM trainer already running (pid {pid}). Not starting a second copy.")
            log(f"[resume] removing stale lock for dead pid {pid}")
            try:
                path.unlink()
            except OSError:
                time.sleep(0.2)
            continue
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.fsync(fd)
        os.close(fd)
        return


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            idx = int(rec["cfg_idx"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if idx in seen:
            continue
        seen.add(idx)
        rows.append(rec)
    return rows


def best_from(rows: list[dict]) -> tuple[int | None, float]:
    best_idx, best_auc = None, -1.0
    for rec in rows:
        dev = rec.get("dev_auc")
        if dev is None:
            continue
        dev = float(dev)
        if np.isfinite(dev) and dev > best_auc:
            best_auc = dev
            best_idx = int(rec["cfg_idx"])
    return best_idx, best_auc


def save_fusion(path: Path, fusion: AttentionFusionFit) -> None:
    durable_torch(
        path,
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
            "mean_attention_dev": fusion.mean_attention_dev,
        },
    )


def load_fusion(path: Path) -> AttentionFusionFit:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    keys = [tuple(k) for k in ck["group_keys"]]
    return AttentionFusionFit(
        group_keys=keys,
        group_dims=[int(d) for d in ck["group_dims"]],
        block_keys=list(ck["block_keys"]),
        means=ck["means"],
        stds=ck["stds"],
        target_dim=int(ck["target_dim"]),
        state_dict=ck["state_dict"],
        epochs_run=int(ck.get("epochs_run") or 0),
        fusion_dev_auc=float(ck.get("fusion_dev_auc", float("nan"))),
        mean_attention_dev=ck.get("mean_attention_dev"),
    )


def worker(job: tuple[int, dict]) -> tuple[int, float, str]:
    cfg_idx, params = job
    which = base.pick_device(params)
    device = base.FOLD["gpu_device"] if which == "gpu" and base.FOLD["gpu_device"] else "cpu"
    dev_auc, _ = base.fit_one(
        base.FOLD["kind"],
        params,
        base.FOLD["tensors_gpu"] if device != "cpu" else base.FOLD["tensors_cpu"],
        base.config_seed(base.FOLD["seed"], base.FOLD["fold_i"], base.FOLD["kind"], cfg_idx),
        base.FOLD["batch_size"],
        base.FOLD["momentum"],
        base.FOLD["patience"],
        device,
    )
    return cfg_idx, dev_auc, device


def aux_test_metrics(model, Xte: np.ndarray, true_psych: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> dict:
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(torch.tensor(Xte, dtype=torch.float32, device=device))
        pred_z = out["psychiatric"].float().cpu().numpy()
    pred = pred_z * sd + mu
    err = np.abs(pred - true_psych)
    names = PSYCHIATRIC
    metrics = {f"mae_{name.lower()}": float(err[:, i].mean()) for i, name in enumerate(names)}
    for i, name in enumerate(names):
        if np.std(true_psych[:, i]) > 0 and np.std(pred[:, i]) > 0:
            metrics[f"pearson_{name.lower()}"] = float(np.corrcoef(true_psych[:, i], pred[:, i])[0, 1])
        else:
            metrics[f"pearson_{name.lower()}"] = float("nan")
    return metrics


def write_metrics(path: Path, fold_rows: list[dict], fusion_rows: list[dict], n_blocks: int, n_workers: int) -> None:
    fold_rows = sorted(fold_rows, key=lambda m: int(m["fold"]))
    fusion_rows = sorted(fusion_rows, key=lambda m: int(m["fold"]))
    names = ["auc_roc", "pr_auc", "f1", "f1_dev_thr", "cohens_d", "spearman_ordinal", "mae_phq9", "mae_gad"]
    summary = {name: base.agg(fold_rows, name) for name in names}
    summary["folds"] = fold_rows
    durable_json(
        path,
        {
            "fusion_method": "attention_pool",
            "train_target": "high",
            "eval_target": "high",
            "variant": "mtm",
            "hierarchy": ["personality", "psychosocial", "psychiatric", "suicide"],
            "aux_targets": {
                "personality": PERSONALITY,
                "psychosocial": PSYCHOSOCIAL,
                "psychiatric": PSYCHIATRIC,
            },
            "summary": {"mtm_high": summary},
            "fusion_folds": fusion_rows,
            "meta": {
                "n_blocks": n_blocks,
                "workers": n_workers,
                "pos_weight_scale": POS_WEIGHT_SCALE,
                "same_representations_as": "artifacts/pw_1.5 STM attention fusion",
                "cascade": "tanh(previous prediction) concatenated with the shared trunk",
                "attention_cfg": {
                    "lr": 0.01,
                    "epochs": 500,
                    "patience": 50,
                    "pos_weight_scale": POS_WEIGHT_SCALE,
                },
            },
        },
    )


def main() -> None:
    global LOG_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="eli_matrix.yaml")
    ap.add_argument("--reps", default="more stuff/reps")
    ap.add_argument("--out", default="artifacts/pw_1.5_mtm_gated")
    ap.add_argument("--fusion-dir", default="artifacts/pw_1.5")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--gpus", default="0")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    tcfg = cfg.train
    want = [int(x) for x in args.folds.split(",") if x != ""]
    gpus = []
    if args.gpus not in ("", "cpu", "none") and torch.cuda.is_available():
        gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    elif args.gpus not in ("", "cpu", "none") and not torch.cuda.is_available():
        log("[train] CUDA not available; wide nets will run on CPU")

    out_dir = (ROOT / args.out).resolve()
    roots = resume_roots(out_dir)
    LOG_PATH = roots[0] / "train.log"
    log(f"[resume] progress copies: {roots[0]} and {roots[1]}")
    acquire_lock(roots[0] / "trainer.lock")
    try:
        _run(cfg, tcfg, want, gpus, args, out_dir, roots)
    finally:
        lock = roots[0] / "trainer.lock"
        if lock.exists():
            try:
                if int(lock.read_text(encoding="utf-8").strip()) == os.getpid():
                    lock.unlink()
            except (OSError, ValueError):
                pass


def _merged_results(roots: list[Path], fold_i: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[int] = set()
    for root in roots:
        for rec in load_results(root / f"fold{fold_i}" / "results.jsonl"):
            idx = int(rec["cfg_idx"])
            if idx in seen:
                continue
            seen.add(idx)
            rows.append(rec)
    return rows


def _load_done(roots: list[Path], fold_i: int) -> dict | None:
    for root in roots:
        path = root / f"fold{fold_i}" / "done.json"
        if path.exists() and path.stat().st_size > 0:
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _run(cfg, tcfg, want, gpus, args, out_dir: Path, roots: list[Path]) -> None:
    cohort = build_cohort(cfg, assert_paper=False)
    rep_roots = {m["name"]: (ROOT / args.reps / m["name"]).resolve() for m in cfg.represent["models"]}
    missing = [str(p) for p in rep_roots.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing rep dirs: {missing}")

    user_ids = cohort["UserId"].tolist()
    y_strat = cohort["y_high"].to_numpy()
    log(f"[train] MTM loading {len(user_ids)} users from {args.reps}")
    t0 = time.time()
    all_blocks = {}
    for i, uid in enumerate(user_ids, 1):
        all_blocks[uid] = collect_user_blocks(rep_roots, uid)
        if i % 200 == 0:
            log(f"[train] loaded blocks {i}/{len(user_ids)}")
    n_blocks = len(next(iter(all_blocks.values())))
    log(
        f"[train] {len(all_blocks)} users x {n_blocks} blocks in {time.time() - t0:.0f}s | "
        f"MTM hierarchy personality -> psychosocial -> psychiatric(PHQ9,GAD) -> suicide | "
        f"workers={args.workers} gpus={gpus}"
    )

    grid = tcfg["grid"]
    param_grid = [
        {"n_layers": a, "n_neurons": b, "activation": c, "lr": d, "epochs": e}
        for a, b, c, d, e in itertools.product(
            grid["n_layers"], grid["n_neurons"], grid["activation"], grid["lr"], grid["epochs"]
        )
    ]
    attention_cfg = {"lr": 0.01, "epochs": 500, "patience": 50, "pos_weight_scale": POS_WEIGHT_SCALE, "device": "cpu"}
    batch_size = int(tcfg.get("batch_size", 32))
    momentum = float(tcfg.get("momentum", 0.9))
    patience = int(tcfg.get("early_stop_patience", 200))
    n_workers = args.workers

    splits = list(
        _make_splits(y_strat, int(tcfg["n_folds"]), cfg.seed, float(tcfg["train_frac"]), float(tcfg["dev_frac"]))
    )
    idx_cohort = cohort.set_index("UserId")

    def ids_of(idx):
        return [user_ids[i] for i in idx]

    fold0 = next(s for s in splits if s[0] == 0)
    _, tr0, dv0, te0 = fold0
    te_sub = idx_cohort.loc[ids_of(te0)]
    n_te, n_pos = len(te_sub), int(te_sub["y_high"].sum())
    log(f"[verify] fold 0 test n={n_te} n_pos={n_pos}")
    if n_te != base.EXPECTED_FOLD0["n"] or n_pos != base.EXPECTED_FOLD0["n_pos"]:
        raise SystemExit("fold 0 split does not match the saved pw_1.5 run; refusing to train")

    ctx = mp.get_context("spawn")
    kind = "mtm"
    metrics_path = out_dir / "mtm_metrics.json"
    completed: dict[int, dict] = {}
    fusion_done: dict[int, dict] = {}
    for fold_i in want:
        rec = _load_done(roots, fold_i)
        if rec is not None:
            completed[fold_i] = rec["metrics"]
            fusion_done[fold_i] = rec["fusion"]
            log(f"[resume] fold {fold_i} already finished, test AUC={rec['metrics']['auc_roc']:.3f}")

    if completed:
        write_metrics(metrics_path, list(completed.values()), list(fusion_done.values()), n_blocks, n_workers)
        write_metrics(roots[0] / "mtm_metrics.json", list(completed.values()), list(fusion_done.values()), n_blocks, n_workers)

    for fold_i, tr_idx, dv_idx, te_idx in splits:
        if fold_i not in want or fold_i in completed:
            continue
        for root in roots:
            (root / f"fold{fold_i}").mkdir(parents=True, exist_ok=True)
        ids = {"tr": ids_of(tr_idx), "dv": ids_of(dv_idx), "te": ids_of(te_idx)}
        fusion_dir = (ROOT / args.fusion_dir).resolve()
        fusion_candidates = [
            fusion_dir / f"fusion_fold{fold_i}.pt",
            out_dir / f"fusion_fold{fold_i}.pt",
            roots[0] / f"fusion_fold{fold_i}.pt",
        ]
        fusion_path = next((p for p in fusion_candidates if p.exists() and p.stat().st_size > 0), fusion_candidates[0])
        tf = time.time()
        reused_fusion = fusion_path.exists() and fusion_path.stat().st_size > 0
        if reused_fusion:
            fusion = load_fusion(fusion_path)
            log(
                f"[train] fold {fold_i}: loaded fusion fusion_dev_auc={fusion.fusion_dev_auc:.6f} "
                f"from {fusion_path.name}"
            )
        else:
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
            log(
                f"[train] fold {fold_i}: fit fusion fusion_dev_auc={fusion.fusion_dev_auc:.6f} "
                f"epochs_run={fusion.epochs_run} ({time.time() - tf:.1f}s)"
            )
            if fold_i == 0 and abs(float(fusion.fusion_dev_auc) - base.EXPECTED_FOLD0_FUSION_AUC) > 0.015:
                raise SystemExit("fold 0 fusion AUC does not match pw_1.5; refusing to train")
            save_fusion(fusion_path, fusion)
            save_fusion(roots[0] / f"fusion_fold{fold_i}.pt", fusion)
            log(f"[train] fold {fold_i}: fusion checkpoint saved")

        if fold_i == 0 and abs(float(fusion.fusion_dev_auc) - base.EXPECTED_FOLD0_FUSION_AUC) > 0.015:
            raise SystemExit("loaded fold 0 fusion AUC does not match pw_1.5; refusing to train")

        def pack(split, fusion=fusion):
            X = fusion.transform_many([all_blocks[u] for u in ids[split]])
            sub = idx_cohort.loc[ids[split]]
            return X, sub["y_high"].to_numpy().astype(np.float32), sub["suicide"].to_numpy(), sub

        Xtr, ytr, _, str_ = pack("tr")
        Xdv, ydv, _, _ = pack("dv")
        Xte, yte, yord, ste = pack("te")
        aux_tr = {
            "personality": str_[PERSONALITY].to_numpy().astype(np.float32),
            "psychosocial": str_[PSYCHOSOCIAL].to_numpy().astype(np.float32),
            "psychiatric": str_[PSYCHIATRIC].to_numpy().astype(np.float32),
        }
        mu_sd = {k: _zscore_fit(v) for k, v in aux_tr.items()}
        fd = {
            "Xtr": Xtr,
            "ytr": ytr,
            "Xdv": Xdv,
            "ydv": ydv,
            "pers": (aux_tr["personality"] - mu_sd["personality"][0]) / mu_sd["personality"][1],
            "psy": (aux_tr["psychosocial"] - mu_sd["psychosocial"][0]) / mu_sd["psychosocial"][1],
            "psych": (aux_tr["psychiatric"] - mu_sd["psychiatric"][0]) / mu_sd["psychiatric"][1],
        }
        true_psych = ste[PSYCHIATRIC].to_numpy().astype(np.float32)
        psych_mu, psych_sd = mu_sd["psychiatric"]
        fusion_rec = {
            "fold": fold_i,
            "method": "attention_pool",
            "seconds": round(time.time() - tf, 1),
            "in_dim": 1024,
            "fusion_dev_auc": float(fusion.fusion_dev_auc),
            "epochs_run": int(fusion.epochs_run),
            "reused_stm_fusion": reused_fusion,
        }
        log(
            f"[train] mtm_high fold {fold_i}: train={len(ids['tr'])} dev={len(ids['dv'])} "
            f"test={len(ids['te'])} grid={len(param_grid)} fusion_dev_auc={fusion.fusion_dev_auc:.3f}"
        )

        done_rows = _merged_results(roots, fold_i)
        done_idx = {int(r["cfg_idx"]) for r in done_rows}
        best_idx, best_auc = best_from(done_rows)
        devices = {int(r["cfg_idx"]): r.get("device", "cpu") for r in done_rows}
        if done_idx:
            log(f"[resume] fold {fold_i}: {len(done_idx)}/{len(param_grid)} configs already saved, best_dev_auc={best_auc:.3f}")

        order = sorted(
            range(len(param_grid)),
            key=lambda i: (int(param_grid[i]["epochs"]), int(param_grid[i]["n_neurons"]), int(param_grid[i]["n_layers"])),
            reverse=True,
        )
        pending = [i for i in order if i not in done_idx]
        gpu_idx = [i for i in pending if gpus and base.pick_device(param_grid[i]) == "gpu"]
        cpu_idx = [i for i in pending if i not in set(gpu_idx)]
        log(f"[train] mtm_high fold {fold_i}: {len(gpu_idx)} GPU configs left, {len(cpu_idx)} CPU configs left")

        state = {
            "done": len(done_idx),
            "session": 0,
            "best_auc": best_auc,
            "best_idx": best_idx,
        }
        lock = threading.Lock()
        stop = threading.Event()
        gpu_err: list[BaseException] = []
        tg = time.time()

        def note(cfg_idx: int, dev_auc: float, device: str) -> None:
            total = len(param_grid)
            dev_val = float(dev_auc) if np.isfinite(dev_auc) else None
            with lock:
                if cfg_idx in done_idx:
                    return
                write_both(
                    roots,
                    append_jsonl,
                    f"fold{fold_i}/results.jsonl",
                    {"cfg_idx": cfg_idx, "dev_auc": dev_val, "device": device},
                )
                done_idx.add(cfg_idx)
                devices[cfg_idx] = device
                state["done"] += 1
                state["session"] += 1
                if dev_val is not None and dev_val > state["best_auc"]:
                    state["best_auc"] = dev_val
                    state["best_idx"] = cfg_idx
                done_n = state["done"]
                sess = state["session"]
                best = state["best_auc"]
                best_i = state["best_idx"]
                write_both(
                    roots,
                    durable_json,
                    f"fold{fold_i}/status.json",
                    {
                        "fold": fold_i,
                        "done": done_n,
                        "total": len(param_grid),
                        "best_dev_auc": best,
                        "best_idx": best_i,
                        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
            if sess == 1 or done_n % 25 == 0 or done_n == total:
                el = time.time() - tg
                eta = (total - done_n) * el / max(sess, 1) / 60
                log(
                    f"[train] mtm_high fold {fold_i}: {done_n}/{total} "
                    f"({el:.0f}s, {el / max(sess, 1):.1f}s/cfg, ETA {eta:.1f}m) best_dev_auc={best:.3f}"
                )

        def gpu_loop() -> None:
            try:
                device = f"cuda:{gpus[0]}"
                tensors = base.to_tensors(fd, device)
                for i in gpu_idx:
                    if stop.is_set():
                        break
                    try:
                        dev_auc, _ = base.fit_one(
                            kind,
                            param_grid[i],
                            tensors,
                            base.config_seed(cfg.seed, fold_i, kind, i),
                            batch_size,
                            momentum,
                            patience,
                            device,
                        )
                        used = device
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        log(f"[train] GPU OOM on cfg {i}, retrying on CPU")
                        torch.cuda.empty_cache()
                        dev_auc, _ = base.fit_one(
                            kind,
                            param_grid[i],
                            base.to_tensors(fd, "cpu"),
                            base.config_seed(cfg.seed, fold_i, kind, i),
                            batch_size,
                            momentum,
                            patience,
                            "cpu",
                        )
                        used = "cpu"
                    note(i, dev_auc, used)
                del tensors
                torch.cuda.empty_cache()
            except BaseException as exc:  # noqa: BLE001
                gpu_err.append(exc)
                stop.set()

        gpu_thread = None
        if gpu_idx:
            gpu_thread = threading.Thread(target=gpu_loop, daemon=False)
            gpu_thread.start()
        cpu_err: BaseException | None = None
        try:
            if cpu_idx:
                with ProcessPoolExecutor(
                    max_workers=min(n_workers, len(cpu_idx)),
                    mp_context=ctx,
                    initializer=base._child_setup,
                    initargs=([], {"kind": kind, "fold_i": fold_i, "seed": cfg.seed, "batch_size": batch_size, "momentum": momentum, "patience": patience, "fd": fd}),
                ) as ex:
                    futs = [ex.submit(worker, (i, param_grid[i])) for i in cpu_idx]
                    for fut in as_completed(futs):
                        note(*fut.result())
        except BaseException as exc:  # noqa: BLE001
            cpu_err = exc
            stop.set()
        if gpu_thread is not None:
            gpu_thread.join()
        if cpu_err is not None:
            raise cpu_err
        if gpu_err:
            raise gpu_err[0]

        rows = _merged_results(roots, fold_i)
        if len(rows) < len(param_grid):
            raise SystemExit(
                f"fold {fold_i} stopped with {len(rows)}/{len(param_grid)} configs saved. "
                "Rerun the same command to resume."
            )
        best_idx, best_auc = best_from(rows)
        if best_idx is None:
            raise SystemExit(f"fold {fold_i} has no finite dev AUC")
        params = param_grid[best_idx]
        dev_name = devices.get(best_idx, "cpu")
        if dev_name.startswith("cuda") and not torch.cuda.is_available():
            dev_name = "cpu"
        log(f"[train] mtm_high fold {fold_i}: refitting winner cfg {best_idx} on {dev_name} params={params}")
        write_both(
            roots,
            durable_json,
            f"fold{fold_i}/status.json",
            {"fold": fold_i, "phase": "refit", "best_idx": best_idx, "best_dev_auc": best_auc, "params": params},
        )
        tensors = base.to_tensors(fd, dev_name)
        dev_auc2, model = base.fit_one(
            kind,
            params,
            tensors,
            base.config_seed(cfg.seed, fold_i, kind, best_idx),
            batch_size,
            momentum,
            patience,
            dev_name,
        )
        with torch.no_grad():
            te_logits = model(tensors["Xtr"].new_tensor(Xte))["suicide_logit"].float().cpu().numpy()
            dv_logits = model(tensors["Xdv"])["suicide_logit"].float().cpu().numpy()
        scores = 1 / (1 + np.exp(-np.clip(te_logits, -60, 60)))
        dev_scores = 1 / (1 + np.exp(-np.clip(dv_logits, -60, 60)))
        metrics = base.summarize_scores(yte, scores)
        metrics.update(
            {
                "fold": fold_i,
                "dev_auc": best_auc,
                "dev_auc_refit": dev_auc2,
                **base.extra_scores(yte, yord, scores, ydv, dev_scores),
                **aux_test_metrics(model, Xte, true_psych, psych_mu, psych_sd),
                **{f"p_{k}": v for k, v in params.items()},
            }
        )
        model_path = out_dir / f"mtm_high_fold{fold_i}.pt"
        ckpt = {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "params": params,
            "metrics": metrics,
            "aux_mu_sd": {k: (mu.tolist(), sd.tolist()) for k, (mu, sd) in mu_sd.items()},
            "aux_names": {
                "personality": list(PERSONALITY),
                "psychosocial": list(PSYCHOSOCIAL),
                "psychiatric": list(PSYCHIATRIC),
            },
        }
        durable_torch(model_path, ckpt)
        durable_torch(roots[0] / model_path.name, ckpt)
        write_both(roots, durable_json, f"fold{fold_i}/done.json", {"metrics": metrics, "fusion": fusion_rec})
        completed[fold_i] = metrics
        fusion_done[fold_i] = fusion_rec
        write_metrics(metrics_path, list(completed.values()), list(fusion_done.values()), n_blocks, n_workers)
        write_metrics(roots[0] / "mtm_metrics.json", list(completed.values()), list(fusion_done.values()), n_blocks, n_workers)
        log(
            f"[train] mtm_high fold {fold_i} DONE test AUC={metrics['auc_roc']:.3f} "
            f"PR={metrics['pr_auc']:.3f} F1={metrics['f1']:.3f} "
            f"F1@dev={metrics['f1_dev_thr']:.3f} d={metrics['cohens_d']:.3f} "
            f"rho={metrics['spearman_ordinal']:.3f} "
            f"PHQ9_r={metrics['pearson_phq9']:.3f} GAD_r={metrics['pearson_gad']:.3f} "
            f"params={params} ({(time.time() - tg) / 60:.1f}m)"
        )
        del tensors, model
        if gpus:
            torch.cuda.empty_cache()

    write_metrics(metrics_path, list(completed.values()), list(fusion_done.values()), n_blocks, n_workers)
    write_metrics(roots[0] / "mtm_metrics.json", list(completed.values()), list(fusion_done.values()), n_blocks, n_workers)
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))["summary"]["mtm_high"]["auc_roc"]
    stm_path = out_dir / "metrics.json"
    stm_mean = None
    if stm_path.exists():
        stm_mean = json.loads(stm_path.read_text(encoding="utf-8"))["summary"]["stm_high"]["auc_roc"]["mean"]
    log(
        f"\n=== SUMMARY MTM [attention_pool | high | pos_weight_scale={POS_WEIGHT_SCALE}] ===\n"
        f"  mtm_high: AUC={summary['mean']:.3f} [{summary['ci95'][0]:.3f}, {summary['ci95'][1]:.3f}] "
        f"folds={[round(v, 3) for v in summary['values']]}"
    )
    if stm_mean is not None and summary["values"] and len(summary["values"]) == 5:
        log(f"  stm_high baseline mean AUC={stm_mean:.3f}")


if __name__ == "__main__":
    main()
