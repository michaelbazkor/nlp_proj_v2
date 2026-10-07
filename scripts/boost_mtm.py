"""Ways to raise the suicide AUC of the aux-weighted MTM (scripts/aux_weighted_mtm.py, setting g4).

Every choice (epoch, weight, variant) is made on the development folds. Test AUC is
read afterwards. The saved models in artifacts/pw_1.5* are only read.

    python scripts/boost_mtm.py prep                      cache fold features and saved-model scores
    python scripts/boost_mtm.py train VARIANT 0,1,2       train seeds 0-2 of a variant on all folds
    python scripts/boost_mtm.py analyze                   compare methods, write summary.json

Outputs: artifacts/pw_1.5_mtm_boost/ (scores per epoch, small). Best-epoch weights go to
%LOCALAPPDATA%/nlp_proj_v2/boost_states/ so OneDrive does not sync hundreds of MB.
"""
from __future__ import annotations

import copy
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from reselect_mtm import UngatedMTM
from ssr.train.metrics import auc_roc

OUT = ROOT / "artifacts" / "pw_1.5_mtm_boost"
CACHE = OUT / "cache"
RUNS = OUT / "runs"
STATES = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "nlp_proj_v2" / "boost_states"
HEADS = ["BFI_O", "BFI_C", "BFI_E", "BFI_A", "BFI_N", "Brooding", "Worry", "Lonely", "SWL", "PHQ9", "GAD"]
REAL = [i for i, c in enumerate(HEADS) if c != "SWL"]
MAX_EPOCHS, PATIENCE, TOPK, KEEP_HEADS = 600, 150, 5, 10

VARIANTS = {
    "g4": {"aux": (4, 4, 4), "opt": "rmsprop", "lr": 1e-4},
    "adam": {"aux": (4, 4, 4), "opt": "adam", "lr": 3e-4, "clip": 1.0},
    "ord": {"aux": (4, 4, 4), "opt": "rmsprop", "lr": 1e-4, "ord": 4.0},
    "psych": {"aux": (2, 4, 8), "opt": "rmsprop", "lr": 1e-4},
    "two": {"aux": (4, 4, 4), "opt": "rmsprop", "lr": 1e-4, "pretrain": 150, "trunk_mult": 0.2},
    "drop": {"aux": (4, 4, 4), "opt": "rmsprop", "lr": 1e-4, "drop": 0.3},
    "pw": {"aux": (4, 4, 4), "opt": "rmsprop", "lr": 1e-4, "pos_weight": True},
    "a1": {"aux": (1, 1, 1), "opt": "rmsprop", "lr": 1e-4},
    "g4e": {"aux": (4, 4, 4), "opt": "rmsprop", "lr": 1e-4, "ema": 0.99},
    "g4e9": {"aux": (4, 4, 4), "opt": "rmsprop", "lr": 1e-4, "ema": 0.999},
    "adame": {"aux": (4, 4, 4), "opt": "adam", "lr": 3e-4, "clip": 1.0, "ema": 0.99},
    "twoe": {"aux": (4, 4, 4), "opt": "rmsprop", "lr": 1e-4, "pretrain": 150, "trunk_mult": 0.2, "ema": 0.99},
    "slowe": {"aux": (4, 4, 4), "opt": "rmsprop", "lr": 2e-5, "ema": 0.99},
}


def zfit(x: np.ndarray):
    mu, sd = x.mean(0), x.std(0)
    return mu, np.where(sd < 1e-8, 1.0, sd)


def prep() -> None:
    from sklearn.linear_model import RidgeCV

    import train_pw15_mtm as saved
    from aux_weighted_mtm import load_world, pack_fold
    from ssr.models.stm import STM

    CACHE.mkdir(parents=True, exist_ok=True)
    world = load_world()
    _cfg, user_ids, blocks, idx, splits = world
    for fold_i, tr, dv, te in splits:
        fusion = saved.load_fusion(ROOT / "artifacts" / "pw_1.5" / f"fusion_fold{fold_i}.pt")
        f = pack_fold(fold_i, tr, dv, te, user_ids, blocks, idx, fusion)
        y = {k: f[k]["y_high"].to_numpy().astype(np.float32) for k in ("ftr", "fdv", "fte")}
        z = {s: np.concatenate([f[f"z{s}"][k] for k in ("personality", "psychosocial", "psychiatric")], 1)
             for s in ("tr", "dv", "te")}
        o_mu, o_sd = zfit(f["ftr"]["suicide"].to_numpy().astype(np.float32))
        ordinal = {s: (f[k]["suicide"].to_numpy().astype(np.float32) - o_mu) / o_sd
                   for s, k in (("tr", "ftr"), ("dv", "fdv"), ("te", "fte"))}

        xmu, xsd = zfit(f["Xtr"])
        ridge = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit((f["Xtr"] - xmu) / xsd, z["tr"][:, HEADS.index("PHQ9")])
        raw = {s: ridge.predict((f[f"X{s}"] - xmu) / xsd).astype(np.float32) for s in ("tr", "dv", "te")}
        phq = {s: (raw[s] - raw["tr"].mean()) / (raw["tr"].std() or 1.0) for s in ("dv", "te")}

        mtm = UngatedMTM(f["Xtr"].shape[1], 1, 512, "tanh")
        mtm.load_state_dict(torch.load(ROOT / f"artifacts/pw_1.5_mtm_shared/mtm_high_fold{fold_i}.pt",
                                       map_location="cpu", weights_only=False)["state_dict"])
        stm_ck = torch.load(ROOT / f"artifacts/pw_1.5/stm_high_fold{fold_i}.pt", map_location="cpu", weights_only=False)
        p = stm_ck["params"]
        stm = STM(f["Xtr"].shape[1], int(p["n_layers"]), int(p["n_neurons"]), p["activation"])
        stm.load_state_dict(stm_ck["state_dict"])
        mtm.eval()
        stm.eval()
        with torch.no_grad():
            logits = {(name, s): model(torch.tensor(f[f"X{s}"]))["suicide_logit"].numpy()
                      for name, model in (("mtm", mtm), ("stm", stm)) for s in ("dv", "te")}
        corr = np.array([np.corrcoef(z["tr"][:, j], y["ftr"])[0, 1] for j in range(len(HEADS))], dtype=np.float32)
        np.savez(
            CACHE / f"fold{fold_i}.npz",
            Xtr=f["Xtr"], Xdv=f["Xdv"], Xte=f["Xte"],
            ytr=y["ftr"], ydv=y["fdv"], yte=y["fte"],
            ztr=z["tr"], zdv=z["dv"], zte=z["te"],
            otr=ordinal["tr"], odv=ordinal["dv"], ote=ordinal["te"],
            mtm_dv=logits[("mtm", "dv")], mtm_te=logits[("mtm", "te")],
            stm_dv=logits[("stm", "dv")], stm_te=logits[("stm", "te")],
            phq_dv=phq["dv"], phq_te=phq["te"], corr=corr,
        )
        print(f"fold {fold_i} cached: train {len(y['ftr'])} dev {len(y['fdv'])} test {len(y['fte'])}", flush=True)


def corr_cols(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred = pred - pred.mean(0, keepdims=True)
    true = true - true.mean(0, keepdims=True)
    den = np.sqrt((pred ** 2).sum(0) * (true ** 2).sum(0))
    return np.where(den > 1e-8, (pred * true).sum(0) / np.maximum(den, 1e-12), np.nan)


def refit_epochs(variant: str, fold_i: int) -> int:
    """Median best-dev-AUC epoch of the dev-selected runs of this variant and fold."""
    runs = load_runs(variant)
    return int(np.median([int(np.nanargmax(runs[s][fold_i]["dev_auc"])) + 1 for s in runs]))


def run(variant: str, seed_k: int, fold_i: int, device: str, refit: int | None = None) -> dict:
    v = VARIANTS[variant]
    d = dict(np.load(CACHE / f"fold{fold_i}.npz"))
    if refit:
        for a, b in (("Xtr", "Xdv"), ("ytr", "ydv"), ("ztr", "zdv"), ("otr", "odv")):
            d[a] = np.concatenate([d[a], d[b]])
    T = {k: torch.tensor(d[k], device=device) for k in ("Xtr", "Xdv", "Xte", "ytr", "ztr", "otr")}
    ydv = d["ydv"].astype(int)
    n_tr = T["Xtr"].shape[0]
    seed = (42 * 1_000_003 + fold_i * 10_007 + 5 * 101 + 123 + 1000 * seed_k) % (2 ** 31 - 1)
    torch.manual_seed(seed)
    model = UngatedMTM(T["Xtr"].shape[1], 1, 512, "tanh").to(device)
    ord_head = nn.Linear(512, 1).to(device) if v.get("ord") else None
    linears = [model.shared[0], model.personality[0], model.psychosocial[0], model.psychiatric[0], model.suicide[0]]
    pre: dict[int, torch.Tensor] = {}
    hooks = [lin.register_forward_hook(lambda _m, _i, out, i=i: pre.__setitem__(i, out)) for i, lin in enumerate(linears)]
    shared: dict[str, torch.Tensor] = {}

    def shared_hook(_m, _i, out):
        if v.get("drop") and model.training:
            out = F.dropout(out, p=float(v["drop"]), training=True)
        shared["h"] = out
        return out

    hooks.append(model.shared[1].register_forward_hook(shared_hook))
    trunk = [p for n, p in model.named_parameters() if not n.startswith("suicide")]
    if ord_head is not None:
        trunk += list(ord_head.parameters())
    head = [p for n, p in model.named_parameters() if n.startswith("suicide")]
    groups = [{"params": trunk, "lr": float(v["lr"])}, {"params": head, "lr": float(v["lr"])}]
    opt = torch.optim.Adam(groups) if v["opt"] == "adam" else torch.optim.RMSprop(groups, momentum=0.9)

    def project() -> None:
        # RMSprop/Adam steps have a fixed size, so a saturated tanh never recovers on its own.
        # Rescaling the weights does; the optimizer state is rescaled with them.
        with torch.no_grad():
            for i, lin in enumerate(linears):
                rms = float(pre[i].detach().pow(2).mean().sqrt())
                if rms <= 1.0:
                    continue
                s = 1.0 / rms
                for p in (lin.weight, lin.bias):
                    p.mul_(s)
                    st = opt.state.get(p) or {}
                    for key, power in (("square_avg", 2), ("exp_avg_sq", 2), ("momentum_buffer", 1), ("exp_avg", 1)):
                        if key in st:
                            st[key].mul_(s ** power)

    wp, wy, wh = v["aux"]
    n_p, n_y = 5, 4
    pos = float(T["ytr"].sum())
    pos_weight = torch.tensor((n_tr - pos) / pos, device=device) if v.get("pos_weight") else None

    def loss_of(ii, with_suicide: bool) -> torch.Tensor:
        out = model(T["Xtr"][ii])
        zt = T["ztr"][ii]
        loss = (wp * F.mse_loss(out["personality"], zt[:, :n_p])
                + wy * F.mse_loss(out["psychosocial"], zt[:, n_p:n_p + n_y])
                + wh * F.mse_loss(out["psychiatric"], zt[:, n_p + n_y:]))
        if ord_head is not None:
            loss = loss + float(v["ord"]) * F.mse_loss(ord_head(shared["h"]).squeeze(-1), T["otr"][ii])
        if with_suicide:
            loss = loss + F.binary_cross_entropy_with_logits(out["suicide_logit"], T["ytr"][ii], pos_weight=pos_weight)
        return loss

    n, bs = n_tr, 32
    for _ in range(2):
        model.eval()
        with torch.no_grad():
            model(T["Xtr"][:128])
        project()
    ema_model = copy.deepcopy(model) if v.get("ema") else None
    if ema_model is not None:
        for m in ema_model.modules():
            m._forward_hooks.clear()
        ema_params, live_params = list(ema_model.parameters()), list(model.parameters())

    def ema_update() -> None:
        if ema_model is not None:
            with torch.no_grad():
                torch._foreach_mul_(ema_params, float(v["ema"]))
                torch._foreach_add_(ema_params, live_params, alpha=1.0 - float(v["ema"]))

    def epoch_pass(with_suicide: bool) -> bool:
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            ii = perm[s : s + bs]
            opt.zero_grad()
            loss = loss_of(ii, with_suicide)
            if not torch.isfinite(loss):
                return False
            loss.backward()
            if v.get("clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(v["clip"]))
            opt.step()
            project()
            ema_update()
        return True

    for _ in range(int(v.get("pretrain", 0))):
        epoch_pass(False)
    if v.get("trunk_mult"):
        opt.param_groups[0]["lr"] *= float(v["trunk_mult"])

    dev_auc, dev_r, dev_logits, test_logits, kept = [], [], [], [], []
    best, bad, best_state = -1.0, 0, None
    for epoch in range(refit or MAX_EPOCHS):
        if not epoch_pass(True):
            break
        scorer = ema_model if ema_model is not None else model
        scorer.eval()
        with torch.no_grad():
            odv, ote = scorer(T["Xdv"]), scorer(T["Xte"])
        heads_dv = torch.cat([odv[k] for k in ("personality", "psychosocial", "psychiatric")], 1).cpu().numpy()
        heads_te = torch.cat([ote[k] for k in ("personality", "psychosocial", "psychiatric")], 1).cpu().numpy()
        ldv, lte = odv["suicide_logit"].cpu().numpy(), ote["suicide_logit"].cpu().numpy()
        auc = float(auc_roc(ydv, ldv))
        r = float(np.nanmean(corr_cols(heads_dv, d["zdv"])[REAL]))
        dev_auc.append(auc)
        dev_r.append(r)
        dev_logits.append(ldv)
        test_logits.append(lte)
        kept.append((auc, epoch, heads_dv, heads_te))
        kept = sorted(kept, key=lambda t: -t[0])[:KEEP_HEADS]
        if refit:
            best_state = {k: t.detach().cpu().clone() for k, t in scorer.state_dict().items()}
        elif np.isfinite(auc) and auc > best:
            best, bad = auc, 0
            best_state = {k: t.detach().cpu().clone() for k, t in scorer.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    for h in hooks:
        h.remove()
    if refit:
        variant = f"{variant}_refit"
    out = RUNS / variant
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / f"s{seed_k}_f{fold_i}.npz",
        dev_auc=np.array(dev_auc, np.float32), dev_r=np.array(dev_r, np.float32),
        dev_logits=np.stack(dev_logits).astype(np.float32), test_logits=np.stack(test_logits).astype(np.float32),
        kept_epochs=np.array([k[1] for k in kept]), kept_dev_heads=np.stack([k[2] for k in kept]).astype(np.float32),
        kept_test_heads=np.stack([k[3] for k in kept]).astype(np.float32),
    )
    if best_state is not None:
        (STATES / variant).mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "variant": v, "fold": fold_i, "seed_k": seed_k},
                   STATES / variant / f"s{seed_k}_f{fold_i}.pt")
    i = len(dev_auc) - 1 if refit else int(np.nanargmax(dev_auc))
    yte = d["yte"].astype(int)
    return {"dev_auc": dev_auc[i], "test_auc": float(auc_roc(yte, test_logits[i])), "epoch": i, "dev_r": dev_r[i],
            "epochs": len(dev_auc)}


def train(variant: str, seeds: list[int], device: str, refit: bool = False) -> None:
    torch.set_num_threads(2)
    for seed_k in seeds:
        for fold_i in range(5):
            epochs = refit_epochs(variant, fold_i) if refit else None
            path = RUNS / (f"{variant}_refit" if refit else variant) / f"s{seed_k}_f{fold_i}.npz"
            lock = path.with_suffix(".lock")
            if path.exists() or lock.exists():
                print(f"{variant} s{seed_k} f{fold_i} cached or in progress", flush=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            lock.touch()
            t0 = time.time()
            try:
                res = run(variant, seed_k, fold_i, device, epochs)
            finally:
                lock.unlink(missing_ok=True)
            print(f"{variant} s{seed_k} f{fold_i}: best dev {res['dev_auc']:.3f} (ep {res['epoch']}, r {res['dev_r']:+.3f})"
                  f"  test {res['test_auc']:.3f}  {res['epochs']} epochs  {time.time() - t0:.0f}s", flush=True)


# ---------- analysis -------------------------------------------------------------------------------

def zdev(dv: np.ndarray, te: np.ndarray):
    mu, sd = float(dv.mean()), float(dv.std()) or 1.0
    return (dv - mu) / sd, (te - mu) / sd


def load_runs(variant: str):
    """runs[seed][fold] -> npz dict"""
    runs = {}
    for p in sorted((RUNS / variant).glob("s*_f*.npz")):
        s, f = p.stem.split("_")
        runs.setdefault(int(s[1:]), {})[int(f[1:])] = dict(np.load(p))
    return {s: r for s, r in runs.items() if len(r) == 5}


def per_fold_scores(run: dict, mode: str):
    """Dev and test logits for one run, chosen on dev only."""
    auc = run["dev_auc"]
    if mode == "best":
        i = int(np.nanargmax(auc))
        return run["dev_logits"][i], run["test_logits"][i]
    top = np.argsort(-np.nan_to_num(auc, nan=-1))[:TOPK]
    return run["dev_logits"][top].mean(0), run["test_logits"][top].mean(0)


def heads_of(run: dict):
    k = min(TOPK, len(run["kept_epochs"]))
    return run["kept_dev_heads"][:k].mean(0), run["kept_test_heads"][:k].mean(0)


def analyze() -> None:
    folds = [dict(np.load(CACHE / f"fold{i}.npz")) for i in range(5)]
    ydv = [f["ydv"].astype(int) for f in folds]
    yte = [f["yte"].astype(int) for f in folds]

    def score(dv_list, te_list):
        dv = [auc_roc(y, s) for y, s in zip(ydv, dv_list)]
        te = [auc_roc(y, s) for y, s in zip(yte, te_list)]
        return float(np.mean(dv)), float(np.mean(te)), te

    rows = []

    def add(name, dv_list, te_list, r=None, note=""):
        md, mt, te = score(dv_list, te_list)
        rows.append({"method": name, "dev_auc": md, "test_auc": mt, "test_folds": [round(x, 4) for x in te],
                     "test_r": r, "note": note})

    saved_dv = [f["mtm_dv"] for f in folds]
    saved_te = [f["mtm_te"] for f in folds]
    add("saved MTM", saved_dv, saved_te)
    add("saved MTM + 0.1 PHQ-9", [f["mtm_dv"] + 0.1 * f["phq_dv"] for f in folds],
        [f["mtm_te"] + 0.1 * f["phq_te"] for f in folds])
    add("saved STM", [f["stm_dv"] for f in folds], [f["stm_te"] for f in folds],
        note="dev AUC is optimistic: one of 504 settings was picked per fold on it")

    ensembles = {}
    for variant in VARIANTS:
        runs = load_runs(variant)
        if not runs:
            continue
        seeds = sorted(runs)
        for mode in ("best", "snap"):
            dv0, te0 = zip(*[per_fold_scores(runs[seeds[0]][i], mode) for i in range(5)])
            r0 = test_r_of([runs[seeds[0]][i] for i in range(5)], folds)
            add(f"{variant} seed0 {mode}", dv0, te0, r0)
        if len(seeds) > 1:
            parts = [[zdev(*per_fold_scores(runs[s][i], "best")) for s in seeds] for i in range(5)]
            add(f"{variant} {len(seeds)}-seed best-epoch ensemble", [np.mean([p[0] for p in ps], 0) for ps in parts],
                [np.mean([p[1] for p in ps], 0) for ps in parts], note="reloadable from the saved best-epoch weights")
            dv_e, te_e = [], []
            for i in range(5):
                parts = [zdev(*per_fold_scores(runs[s][i], "snap")) for s in seeds]
                dv_e.append(np.mean([p[0] for p in parts], 0))
                te_e.append(np.mean([p[1] for p in parts], 0))
            r_e = test_r_of([runs[s][i] for s in seeds for i in range(5)], folds, n_seeds=len(seeds))
            add(f"{variant} {len(seeds)}-seed snap ensemble", dv_e, te_e, r_e)
            ensembles[variant] = (dv_e, te_e, seeds, runs)

    if len(ensembles) > 1:
        def mix(names):
            return ([np.mean([ensembles[v][0][i] for v in names], 0) for i in range(5)],
                    [np.mean([ensembles[v][1][i] for v in names], 0) for i in range(5)])
        add(f"all {len(ensembles)} variants mixed", *mix(list(ensembles)))
        by_dev = sorted(ensembles, key=lambda v: -score(ensembles[v][0], ensembles[v][1])[0])[:3]
        add(f"top-3 variants by dev mixed ({', '.join(by_dev)})", *mix(by_dev))

    # Use the middle-stage heads: logit + w * (train-fold-weighted sum of predicted questionnaires).
    weights = [0.0, 0.25, 0.5, 1.0, 2.0]
    for variant, (dv_e, te_e, seeds, runs) in ensembles.items():
        comp_dv, comp_te = [], []
        for i, f in enumerate(folds):
            hd = np.mean([heads_of(runs[s][i])[0] for s in seeds], 0)
            ht = np.mean([heads_of(runs[s][i])[1] for s in seeds], 0)
            c = f["corr"].copy()
            c[HEADS.index("SWL")] = 0.0
            cdv, cte = zdev(hd @ c, ht @ c)
            comp_dv.append(cdv)
            comp_te.append(cte)
        best_w = max(weights, key=lambda w: score([a + w * b for a, b in zip(dv_e, comp_dv)],
                                                  [a + w * b for a, b in zip(te_e, comp_te)])[0])
        add(f"{variant} ensemble + {best_w:g} x predicted questionnaires",
            [a + best_w * b for a, b in zip(dv_e, comp_dv)], [a + best_w * b for a, b in zip(te_e, comp_te)],
            note=f"weight chosen on dev from {weights}")

        # Blend with the saved MTM + PHQ-9 and, separately, the STM. One weight set for all folds.
        saved_phq = [zdev(f["mtm_dv"] + 0.1 * f["phq_dv"], f["mtm_te"] + 0.1 * f["phq_te"]) for f in folds]
        stm = [zdev(f["stm_dv"], f["stm_te"]) for f in folds]
        grid = [w / 4 for w in range(5)]
        for name, other in (("saved MTM+PHQ", saved_phq), ("STM", stm)):
            def blend(w):
                return ([(1 - w) * a + w * b[0] for a, b in zip(dv_e, other)],
                        [(1 - w) * a + w * b[1] for a, b in zip(te_e, other)])
            w = max(grid, key=lambda g: score(*blend(g))[0])
            add(f"{variant} ensemble blended with {name} (w={w:g})", *blend(w),
                note="weight chosen on dev" + ("; STM dev AUC is optimistic" if name == "STM" else ""))
        three = []
        for a, b in itertools.product(grid, grid):
            if a + b <= 1.0:
                three.append((a, b))

        def blend3(ab):
            a, b = ab
            return ([(1 - a - b) * x + a * y[0] + b * z[0] for x, y, z in zip(dv_e, saved_phq, stm)],
                    [(1 - a - b) * x + a * y[1] + b * z[1] for x, y, z in zip(te_e, saved_phq, stm)])
        ab = max(three, key=lambda g: score(*blend3(g))[0])
        add(f"{variant} ensemble + saved MTM+PHQ + STM (w={ab[0]:g}, {ab[1]:g})", *blend3(ab),
            note="weights chosen on dev; STM dev AUC is optimistic")

    # Refit on train+dev for a fixed, dev-chosen epoch count. No honest dev score exists for these.
    for variant in VARIANTS:
        runs = load_runs(f"{variant}_refit")
        if not runs:
            continue
        te = [np.mean([zdev(runs[s][i]["dev_logits"][-5:].mean(0), runs[s][i]["test_logits"][-5:].mean(0))[1]
                       for s in runs], 0) for i in range(5)]
        aucs = [auc_roc(y, t) for y, t in zip(yte, te)]
        r = test_r_of([runs[s][i] for s in sorted(runs) for i in range(5)], folds, n_seeds=len(runs))
        rows.append({"method": f"{variant} refit on train+dev, {len(runs)} seeds", "dev_auc": float("nan"),
                     "test_auc": float(np.mean(aucs)), "test_folds": [round(x, 4) for x in aucs], "test_r": r,
                     "note": "epochs = median dev-chosen epoch; dev is in the training data"})

    rows.sort(key=lambda r: -np.nan_to_num(r["dev_auc"], nan=-1))
    print(f"{'method':62} {'dev':>6} {'test':>6}  {'test r':>6}   test by fold")
    for r in rows:
        rr = "" if r["test_r"] is None else f"{r['test_r']:+.3f}"
        print(f"{r['method']:62} {r['dev_auc']:.3f}  {r['test_auc']:.3f}  {rr:>6}   {r['test_folds']}")
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def test_r_of(run_list: list[dict], folds: list[dict], n_seeds: int = 1) -> float:
    """Mean test r of the predicted questionnaires (heads of the top-k dev-AUC epochs), real scales only."""
    rs = []
    for i in range(5):
        heads = np.mean([heads_of(run_list[s * 5 + i])[1] for s in range(n_seeds)], 0)
        rs.append(np.nanmean(corr_cols(heads, folds[i]["zte"])[REAL]))
    return float(np.mean(rs))


def bottleneck() -> None:
    """How much could the questionnaires add to suicide AUC if the network predicted them perfectly?"""
    from sklearn.linear_model import LogisticRegression

    folds = [dict(np.load(CACHE / f"fold{i}.npz")) for i in range(5)]
    runs = load_runs("g4")
    rows = {"true, all 10 scales (logistic, fit on train)": [], "true PHQ-9 alone": [],
            "predicted by g4 ensemble, same logistic": [], "predicted PHQ-9 alone (g4 ensemble)": []}
    rs = []
    for i, f in enumerate(folds):
        yte = f["yte"].astype(int)
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(f["ztr"][:, REAL], f["ytr"].astype(int))
        pred = np.mean([heads_of(runs[s][i])[1] for s in runs], 0)
        rows["true, all 10 scales (logistic, fit on train)"].append(auc_roc(yte, lr.decision_function(f["zte"][:, REAL])))
        rows["true PHQ-9 alone"].append(auc_roc(yte, f["zte"][:, HEADS.index("PHQ9")]))
        rows["predicted by g4 ensemble, same logistic"].append(auc_roc(yte, lr.decision_function(pred[:, REAL])))
        rows["predicted PHQ-9 alone (g4 ensemble)"].append(auc_roc(yte, pred[:, HEADS.index("PHQ9")]))
        rs.append(np.nanmean(corr_cols(pred, f["zte"])[REAL]))
    for name, vals in rows.items():
        print(f"{name:50} test AUC {np.mean(vals):.3f}  {[round(float(v), 3) for v in vals]}")
    print(f"g4 ensemble mean test r (real scales): {np.mean(rs):+.3f}")


def stability() -> None:
    """Dev-only view that does not reward lucky epochs: the median dev AUC over a run's epochs."""
    rows = []
    for variant in VARIANTS:
        runs = load_runs(variant)
        cells = [runs[s][f]["dev_auc"] for s in runs for f in range(5)]
        dev_r = [runs[s][f]["dev_r"][np.argsort(-np.nan_to_num(runs[s][f]["dev_auc"], nan=-1))[:TOPK]].mean()
                 for s in runs for f in range(5)]
        rows.append((float(np.mean([np.median(a[10:]) for a in cells])), variant,
                     float(np.mean([np.abs(np.diff(a)).mean() for a in cells])),
                     float(np.mean([np.nanmax(a) for a in cells])), float(np.mean(dev_r))))
    print(f"{'variant':8} {'median dev AUC':>15} {'mean epoch-to-epoch change':>27} {'max dev AUC':>12} {'dev r (top-5)':>14}")
    for med, variant, jump, top, r in sorted(rows, reverse=True):
        print(f"{variant:8} {med:15.3f} {jump:27.3f} {top:12.3f} {r:+14.3f}")


def scales(variants: list[str]) -> None:
    """Per-scale test r of the 3-seed ensemble heads, pooled over folds by averaging fold r."""
    folds = [dict(np.load(CACHE / f"fold{i}.npz")) for i in range(5)]
    print(f"{'scale':9}" + "".join(f"{v:>12}" for v in variants))
    table = {}
    for v in variants:
        runs = load_runs(v)
        per_fold = [corr_cols(np.mean([heads_of(runs[s][i])[1] for s in runs], 0), folds[i]["zte"]) for i in range(5)]
        table[v] = np.nanmean(per_fold, 0)
    for j in REAL:
        print(f"{HEADS[j]:9}" + "".join(f"{table[v][j]:+12.3f}" for v in variants))
    print(f"{'mean':9}" + "".join(f"{np.nanmean(table[v][REAL]):+12.3f}" for v in variants))


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "prep":
        prep()
    elif cmd == "train":
        variant = sys.argv[2]
        seeds = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "0").split(",")]
        device = sys.argv[4] if len(sys.argv) > 4 else "cpu"
        train(variant, seeds, device, refit=len(sys.argv) > 5 and sys.argv[5] == "refit")
    elif cmd == "bottleneck":
        bottleneck()
    elif cmd == "stability":
        stability()
    elif cmd == "scales":
        scales(sys.argv[2].split(","))
    else:
        analyze()


if __name__ == "__main__":
    main()
