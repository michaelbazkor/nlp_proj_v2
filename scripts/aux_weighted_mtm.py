"""Retrain the config-123 MTM so the questionnaire losses move the shared weights.

The saved model (artifacts/pw_1.5_mtm_shared) saturates its shared tanh on the
first epoch. RMSprop then takes fixed-size steps, so a heavier questionnaire
loss cannot move those weights: their gradient is zero. Same architecture and
the same 1x512 tanh setting, with two changes:

  * the three questionnaire MSEs are multiplied by aux_scale
  * after every step, each tanh pre-activation is rescaled to rms 1 so the
    middle-stage gradients keep reaching the shared weights

Checkpoints are chosen on the development set: the epoch with the best mean
questionnaire correlation, among epochs whose suicide AUC is within 0.05 of the
saved model's development AUC. Test scores are computed after that choice.

Writes artifacts/pw_1.5_mtm_auxstage/ only.

    python scripts/aux_weighted_mtm.py pilot     fold 0, every setting
    python scripts/aux_weighted_mtm.py s8p       all five folds of one setting
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import continue_pw15 as base
import train_pw15_mtm as saved
from reselect_mtm import UngatedMTM
from ssr.config import load_config
from ssr.data.cohort import build_cohort
from ssr.fusion.project import collect_user_blocks
from ssr.train.cv import PERSONALITY, PSYCHIATRIC, PSYCHOSOCIAL, _make_splits, _zscore_fit
from ssr.train.metrics import auc_roc

OUT_DIR = ROOT / "artifacts" / "pw_1.5_mtm_auxstage"
PARAMS = {"n_layers": 1, "n_neurons": 512, "activation": "tanh", "lr": 0.005, "epochs": 1000}
CFG_IDX = 123
# Development suicide AUC of the saved shared-config model, one entry per fold.
BASELINE_DEV = [0.7284980744544287, 0.8207103123662816, 0.8299101412066753, 0.795250320924262, 0.7351305091998289]
BASELINE_TEST = [0.7286813186813187, 0.7130693912303107, 0.7674542358450405, 0.7317639257294429, 0.6896551724137931]
AUC_SLACK = 0.05
GROUPS = [("personality", PERSONALITY), ("psychosocial", PSYCHOSOCIAL), ("psychiatric", PSYCHIATRIC)]
HEAD_KEYS = ("personality", "psychosocial", "psychiatric")

SETTINGS = [
    {"name": "p4", "aux_scale": 4.0, "sat_weight": 0.0, "trunk_lr": 0.005, "head_lr": 0.005, "project": True},
    {"name": "g4", "aux_scale": 4.0, "sat_weight": 0.0, "trunk_lr": 1e-4, "head_lr": 1e-4, "project": True},
    {"name": "g4s", "aux_scale": 4.0, "sat_weight": 0.0, "trunk_lr": 2e-5, "head_lr": 2e-5, "project": True},
]


def corr_cols(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred = pred - pred.mean(axis=0, keepdims=True)
    true = true - true.mean(axis=0, keepdims=True)
    num = (pred * true).sum(axis=0)
    den = np.sqrt((pred ** 2).sum(axis=0) * (true ** 2).sum(axis=0))
    out = np.full(pred.shape[1], np.nan)
    ok = den > 1e-8
    out[ok] = num[ok] / den[ok]
    return out


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))


class Trainer:
    def __init__(self, setting: dict, seed: int, tensors: dict, device: str):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        in_dim = tensors["Xtr"].shape[1]
        self.model = UngatedMTM(in_dim, PARAMS["n_layers"], PARAMS["n_neurons"], PARAMS["activation"]).to(device)
        self.setting = setting
        self.device = device
        self.z: dict[str, torch.Tensor] = {}
        self.hook = self.model.shared[0].register_forward_hook(lambda _m, _i, out: self.z.__setitem__("z", out))
        trunk, head = [], []
        for name, param in self.model.named_parameters():
            (head if name.startswith("suicide") else trunk).append(param)
        self.opt = torch.optim.RMSprop(
            [
                {"params": trunk, "lr": float(setting["trunk_lr"])},
                {"params": head, "lr": float(setting.get("head_lr", PARAMS["lr"]))},
            ],
            momentum=0.9,
        )
        self.linears = [
            self.model.shared[0],
            self.model.personality[0],
            self.model.psychosocial[0],
            self.model.psychiatric[0],
            self.model.suicide[0],
        ]
        self.tensors = tensors

    def project(self, batch: torch.Tensor) -> None:
        """Keep every tanh pre-activation at rms 1, and rescale RMSprop's state with it.

        RMSprop steps have a fixed size, so a heavier loss cannot pull a saturated
        tanh back. Rescaling the weights does, and rescaling the optimizer state
        stops the old second moment from freezing the next step.
        """
        if not self.setting.get("project"):
            return
        for _ in range(2):
            zs: dict[int, torch.Tensor] = {}
            hooks = [
                lin.register_forward_hook(lambda _m, _i, out, i=i: zs.__setitem__(i, out))
                for i, lin in enumerate(self.linears)
            ]
            with torch.no_grad():
                self.model(batch)
                for i, lin in enumerate(self.linears):
                    rms = float(zs[i].pow(2).mean().sqrt())
                    if rms > 1.0:
                        scale = 1.0 / rms
                        for param in (lin.weight, lin.bias):
                            param.mul_(scale)
                            state = self.opt.state.get(param)
                            if not state:
                                continue
                            if "square_avg" in state:
                                state["square_avg"].mul_(scale * scale)
                            if "momentum_buffer" in state:
                                state["momentum_buffer"].mul_(scale)
            for hook in hooks:
                hook.remove()

    def close(self) -> None:
        self.hook.remove()

    def loss_of(self, out, y, pers, psy, psych) -> torch.Tensor:
        sui = F.binary_cross_entropy_with_logits(out["suicide_logit"], y)
        aux = (
            F.mse_loss(out["personality"], pers)
            + F.mse_loss(out["psychosocial"], psy)
            + F.mse_loss(out["psychiatric"], psych)
        )
        sat = self.z["z"].pow(2).mean()
        return sui + float(self.setting["aux_scale"]) * aux + float(self.setting["sat_weight"]) * sat

    @torch.no_grad()
    def evaluate(self, X, y_high, z_targets: dict[str, np.ndarray]) -> dict:
        self.model.eval()
        out = self.model(X)
        h = torch.tanh(self.z["z"])
        rs = {}
        for key, cols in GROUPS:
            pred = out[key].float().cpu().numpy()
            for col, r in zip(cols, corr_cols(pred, z_targets[key])):
                rs[col] = None if not np.isfinite(r) else float(r)
        values = [v for v in rs.values() if v is not None and v == v]
        no_swl = [v for c, v in rs.items() if c != "SWL" and v is not None]
        logits = out["suicide_logit"].float().cpu().numpy()
        return {
            "auc": float(auc_roc(y_high, sigmoid(logits))),
            "mean_r": float(np.mean(values)) if values else float("nan"),
            "mean_r_no_swl": float(np.mean(no_swl)) if no_swl else float("nan"),
            "r": rs,
            "sat": float((h.abs() > 0.99).float().mean()),
            "logits": logits,
            "heads": {k: out[k].float().cpu().numpy() for k in HEAD_KEYS},
        }


def fit_fold(setting: dict, tensors: dict, z_dev: dict[str, np.ndarray], seed: int, device: str, dev_floor: float):
    trainer = Trainer(setting, seed, tensors, device)
    Xtr, ytr = tensors["Xtr"], tensors["ytr"]
    n, bs = Xtr.shape[0], min(32, Xtr.shape[0])
    best = {"near": None, "composite": None, "auc": None}
    bad = 0
    history = []
    with torch.no_grad():
        trainer.model(Xtr[:64])
        rms = float(trainer.z["z"].pow(2).mean().sqrt())
    print(f"  shared pre-activation rms at init {rms:.2f}", flush=True)
    trainer.project(Xtr[:128])
    for epoch in range(int(PARAMS["epochs"])):
        trainer.model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            ii = perm[s : s + bs]
            trainer.opt.zero_grad()
            out = trainer.model(Xtr[ii])
            loss = trainer.loss_of(out, ytr[ii], tensors["pers"][ii], tensors["psy"][ii], tensors["psych"][ii])
            if not torch.isfinite(loss):
                bad = 200
                break
            loss.backward()
            trainer.opt.step()
            trainer.project(Xtr[ii])
        ev = trainer.evaluate(tensors["Xdv"], tensors["ydv_np"], z_dev)
        score_c = ev["auc"] + 0.5 * ev["mean_r_no_swl"]
        state = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
        row = {"epoch": epoch, "auc": ev["auc"], "mean_r": ev["mean_r_no_swl"], "sat": ev["sat"]}
        history.append(row)

        def keep(slot, better):
            # `better` must itself be true on the first epoch that qualifies.
            # An empty slot is not a reason to keep an epoch that failed the rule.
            if better:
                best[slot] = {"score_auc": ev["auc"], "mean_r": ev["mean_r_no_swl"], "epoch": epoch, "state": state,
                              "r": ev["r"], "sat": ev["sat"]}

        keep("auc", best["auc"] is None or ev["auc"] > best["auc"]["score_auc"])
        keep("composite", best["composite"] is None or score_c > best["composite"]["score_auc"] + 0.5 * best["composite"]["mean_r"])
        near_ok = ev["auc"] >= dev_floor
        keep("near", near_ok and (best["near"] is None or ev["mean_r_no_swl"] > best["near"]["mean_r"]))
        improved = any(best[s] is not None and best[s]["epoch"] == epoch for s in best)
        bad = 0 if improved else bad + 1
        if epoch % 25 == 0 or improved:
            print(
                f"  ep {epoch:4d}  devAUC {ev['auc']:.3f}  r {ev['mean_r_no_swl']:+.3f}  sat {ev['sat']:.2f}"
                f"  {'*' if improved else ''}",
                flush=True,
            )
        if bad >= 200:
            break
    chosen = "near" if best["near"] is not None else "composite"
    trainer.model.load_state_dict(best[chosen]["state"])
    trainer.close()
    return trainer.model, chosen, best, history


def pack_fold(fold_i, tr, dv, te, user_ids, blocks, idx, fusion):
    def ids(ix):
        return [user_ids[i] for i in ix]

    def pack(ix):
        return fusion.transform_many([blocks[u] for u in ids(ix)]), idx.loc[ids(ix)]

    Xtr, ftr = pack(tr)
    Xdv, fdv = pack(dv)
    Xte, fte = pack(te)
    mu = {key: _zscore_fit(ftr[cols].to_numpy().astype(np.float32)) for key, cols in GROUPS}

    def z(frame, key):
        m, s = mu[key]
        cols = dict(GROUPS)[key]
        return (frame[cols].to_numpy().astype(np.float32) - m) / s

    return {
        "Xtr": Xtr, "Xdv": Xdv, "Xte": Xte,
        "ftr": ftr, "fdv": fdv, "fte": fte,
        "mu": mu,
        "ztr": {k: z(ftr, k) for k, _ in GROUPS},
        "zdv": {k: z(fdv, k) for k, _ in GROUPS},
        "zte": {k: z(fte, k) for k, _ in GROUPS},
    }


def tensors_of(fold, device):
    t = {
        "Xtr": torch.tensor(fold["Xtr"], dtype=torch.float32, device=device),
        "ytr": torch.tensor(fold["ftr"]["y_high"].to_numpy().astype(np.float32), device=device),
        "Xdv": torch.tensor(fold["Xdv"], dtype=torch.float32, device=device),
        "ydv_np": fold["fdv"]["y_high"].to_numpy().astype(int),
    }
    t["pers"], t["psy"], t["psych"] = (torch.tensor(fold["ztr"][k], device=device) for k in HEAD_KEYS)
    return t


def test_report(model, fold, device) -> dict:
    model.eval()
    Xte = torch.tensor(fold["Xte"], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(Xte)
    y = fold["fte"]["y_high"].to_numpy().astype(int)
    auc = float(auc_roc(y, sigmoid(out["suicide_logit"].float().cpu().numpy())))
    rs = {}
    for key in HEAD_KEYS:
        pred = out[key].float().cpu().numpy()
        for col, r in zip(dict(GROUPS)[key], corr_cols(pred, fold["zte"][key])):
            rs[col] = None if not np.isfinite(r) else round(float(r), 4)
    no_swl = [v for c, v in rs.items() if c != "SWL" and v is not None]
    return {"test_auc": round(auc, 4), "test_mean_r": round(float(np.mean(no_swl)), 4), "test_r": rs}


def load_world():
    cfg = load_config(ROOT / "eli_matrix.yaml")
    cohort = build_cohort(cfg, assert_paper=False)
    rep_roots = {m["name"]: (ROOT / "more stuff" / "reps" / m["name"]).resolve() for m in cfg.represent["models"]}
    user_ids = cohort["UserId"].tolist()
    print("loading representations", flush=True)
    blocks = {uid: collect_user_blocks(rep_roots, uid) for uid in user_ids}
    y_all = cohort["y_high"].to_numpy()
    splits = list(_make_splits(y_all, 5, cfg.seed, float(cfg.train["train_frac"]), float(cfg.train["dev_frac"])))
    return cfg, user_ids, blocks, cohort.set_index("UserId"), splits


def run_setting(setting: dict, folds_wanted: list[int], world) -> None:
    cfg, user_ids, blocks, idx, splits = world
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out = OUT_DIR / setting["name"]
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {setting['name']} aux_scale={setting['aux_scale']} project={setting.get('project', False)} "
          f"trunk_lr={setting['trunk_lr']} head_lr={setting.get('head_lr', PARAMS['lr'])} on {device} ===", flush=True)
    for fold_i, tr, dv, te in splits:
        if fold_i not in folds_wanted:
            continue
        done = out / f"fold{fold_i}.json"
        if done.exists():
            rec = json.loads(done.read_text(encoding="utf-8"))
            print(f"fold {fold_i} cached test={rec['test_auc']:.3f} r={rec['test_mean_r']:+.3f}", flush=True)
            continue
        fusion = saved.load_fusion(ROOT / "artifacts" / "pw_1.5" / f"fusion_fold{fold_i}.pt")
        fold = pack_fold(fold_i, tr, dv, te, user_ids, blocks, idx, fusion)
        seed = base.config_seed(cfg.seed, fold_i, "mtm", CFG_IDX)
        floor = BASELINE_DEV[fold_i] - AUC_SLACK
        print(f"fold {fold_i} dev floor {floor:.3f}", flush=True)
        model, chosen, best, history = fit_fold(setting, tensors_of(fold, device), fold["zdv"], seed, device, floor)
        report = test_report(model, fold, device)
        auc_state = best["auc"]["state"]
        model.load_state_dict(auc_state)
        auc_report = test_report(model, fold, device)
        model.load_state_dict(best[chosen]["state"])
        torch.save(
            {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "params": PARAMS, "setting": setting, "fold": fold_i, "chosen": chosen},
            out / f"mtm_high_fold{fold_i}.pt",
        )
        rec = {
            "fold": fold_i,
            "setting": setting,
            "chosen_rule": chosen,
            "dev_auc": best[chosen]["score_auc"],
            "dev_mean_r": best[chosen]["mean_r"],
            "dev_sat": best[chosen]["sat"],
            "dev_r": best[chosen]["r"],
            "epoch": best[chosen]["epoch"],
            "baseline_dev_auc": BASELINE_DEV[fold_i],
            "baseline_test_auc": BASELINE_TEST[fold_i],
            "best_auc_epoch": {"epoch": best["auc"]["epoch"], "dev_auc": best["auc"]["score_auc"],
                               "dev_mean_r": best["auc"]["mean_r"], **{f"alt_{k}": v for k, v in auc_report.items()}},
            "epochs_run": history[-1]["epoch"] + 1,
            **report,
        }
        done.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(
            f"  kept {chosen} ep {rec['epoch']}: devAUC {rec['dev_auc']:.3f} (saved model {BASELINE_DEV[fold_i]:.3f})"
            f"  dev r {rec['dev_mean_r']:+.3f}  testAUC {rec['test_auc']:.3f} (saved {BASELINE_TEST[fold_i]:.3f})"
            f"  test r {rec['test_mean_r']:+.3f}",
            flush=True,
        )
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def summarize(name: str) -> None:
    out = OUT_DIR / name
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("fold*.json"))]
    if len(rows) < 5:
        return
    summary = {
        "setting": rows[0]["setting"],
        "rule": "best dev mean questionnaire r among epochs with dev suicide AUC within 0.05 of the saved model;"
                " otherwise best dev AUC + 0.5 * mean r. SWL excluded from the mean (its column repeats extraversion).",
        "mean_test_auc": float(np.mean([r["test_auc"] for r in rows])),
        "mean_test_r": float(np.mean([r["test_mean_r"] for r in rows])),
        "saved_model_mean_test_auc": float(np.mean(BASELINE_TEST)),
        "folds": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{name}: mean test AUC {summary['mean_test_auc']:.3f}  mean r {summary['mean_test_r']:+.3f}", flush=True)


def report(name: str, world) -> None:
    """Score the saved model and one retrain with the same metric on every test fold."""
    from sklearn.linear_model import RidgeCV

    _cfg, user_ids, blocks, idx, splits = world
    device = "cpu"
    new_dir = OUT_DIR / name
    scales = [c for _k, cols in GROUPS for c in cols if c != "SWL"]
    saved_rows, new_rows = [], []
    for fold_i, tr, dv, te in splits:
        fusion = saved.load_fusion(ROOT / "artifacts" / "pw_1.5" / f"fusion_fold{fold_i}.pt")
        fold = pack_fold(fold_i, tr, dv, te, user_ids, blocks, idx, fusion)
        y = fold["fte"]["y_high"].to_numpy().astype(int)
        xmu, xsd = fold["Xtr"].mean(0), np.where(fold["Xtr"].std(0) < 1e-6, 1.0, fold["Xtr"].std(0))

        def norm(X):
            return (X - xmu) / xsd

        ridge = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(norm(fold["Xtr"]), fold["ztr"]["psychiatric"][:, 0])
        raw = [ridge.predict(norm(X)).astype(np.float32) for X in (fold["Xtr"], fold["Xdv"], fold["Xte"])]
        sd = float(raw[0].std()) or 1.0
        mu = float(raw[0].mean())
        phq_te = (raw[2] - mu) / sd

        def score_ckpt(path):
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model = UngatedMTM(fold["Xtr"].shape[1], 1, 512, "tanh")
            model.load_state_dict(ck["state_dict"])
            model.eval()
            with torch.no_grad():
                out = model(torch.tensor(fold["Xte"], dtype=torch.float32))
            logits = out["suicide_logit"].numpy()
            rs = {}
            groups = dict(GROUPS)
            for key in HEAD_KEYS:
                for col, r in zip(groups[key], corr_cols(out[key].numpy(), fold["zte"][key])):
                    rs[col] = float(r)
            return {
                "auc": float(auc_roc(y, sigmoid(logits))),
                "auc_phq": float(auc_roc(y, sigmoid(logits + 0.1 * phq_te))),
                "mean_r": float(np.mean([rs[c] for c in scales])),
                "r": {c: rs[c] for c in scales},
            }

        saved_rows.append(score_ckpt(ROOT / "artifacts" / "pw_1.5_mtm_shared" / f"mtm_high_fold{fold_i}.pt"))
        new_rows.append(score_ckpt(new_dir / f"mtm_high_fold{fold_i}.pt"))
        print(f"fold {fold_i}  saved auc {saved_rows[-1]['auc']:.3f} r {saved_rows[-1]['mean_r']:+.3f}"
              f"   {name} auc {new_rows[-1]['auc']:.3f} r {new_rows[-1]['mean_r']:+.3f}", flush=True)

    def mean(rows, key):
        return float(np.mean([r[key] for r in rows]))

    print(f"saved model   test AUC {mean(saved_rows, 'auc'):.3f}   +PHQ {mean(saved_rows, 'auc_phq'):.3f}"
          f"   mean r {mean(saved_rows, 'mean_r'):+.3f}", flush=True)
    print(f"{name:12} test AUC {mean(new_rows, 'auc'):.3f}   +PHQ {mean(new_rows, 'auc_phq'):.3f}"
          f"   mean r {mean(new_rows, 'mean_r'):+.3f}", flush=True)
    for col in scales:
        a = float(np.mean([r["r"][col] for r in saved_rows]))
        b = float(np.mean([r["r"][col] for r in new_rows]))
        print(f"  {col:8} {a:+.3f} -> {b:+.3f}", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arg = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    world = load_world()
    if arg == "pilot":
        for setting in SETTINGS:
            run_setting(setting, [0], world)
        return
    if arg.startswith("report"):
        name = sys.argv[2] if len(sys.argv) > 2 else "g4"
        report(name, world)
        return
    wanted = [s for s in SETTINGS if s["name"] == arg]
    if not wanted:
        sys.exit(f"unknown setting {arg}; choose pilot or one of {[s['name'] for s in SETTINGS]}")
    folds = [0, 1, 2, 3, 4]
    if len(sys.argv) > 2:
        folds = [int(x) for x in sys.argv[2].split(",")]
    run_setting(wanted[0], folds, world)
    summarize(arg)


if __name__ == "__main__":
    main()
