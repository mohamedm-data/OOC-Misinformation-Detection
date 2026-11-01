"""Step 8: train on the full NewsCLIPpings merged_balanced training split (71,072 pairs) instead of the
19,200-pair curated subset, plus a learning curve, a regularized MLP, calibration and error analysis.

Same frozen CLIP ViT-B/32 pair features as steps 5 and 7 ([img, txt, img*txt, |img-txt|, cosine], 2049-d).
Tuning uses the full validation split (7,024 pairs). The primary test set is the same curated 2,400-pair test
set as all earlier steps (a subset of the official test split, disjoint from every training pair), so numbers
compare directly with the baseline, step 5 and step 7. The full official test split (7,264 pairs) is reported too.

Embeddings: the provided pickles hold every split's embeddings; build_export() copies the rows used by
merged_balanced into data/full_export/*.npy (float16, L2-normalised) once, so this step runs in about 3 GB of RAM.

Outputs: results.json key "full_split", outputs/models/logreg_full.joblib, outputs/models/mlp_full.pt,
         outputs/figures/fig_learning_curve.png, fig_auc_full_vs_subset.png, fig_calibration.png
"""
import json
import pickle
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from common import CUR, FIG, MODELS, NC, ROOT, SEED, evaluate, pair_features, plot_auc_bars, update_results, \
    val_threshold

EXPORT = ROOT / "data" / "full_export"
FRACTIONS = [0.1, 0.27, 0.5, 1.0]          # 0.27 x 71,072 ~ 19,200, the curated subset size
C_GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01]
MLP_CONFIGS = {
    "draft_config": {"hidden": [512, 128], "dropout": 0.3, "input_dropout": 0.1, "weight_decay": 1e-2},
    "small_strong_reg": {"hidden": [256], "dropout": 0.5, "input_dropout": 0.2, "weight_decay": 5e-2},
    "wide_strong_reg": {"hidden": [512, 128], "dropout": 0.5, "input_dropout": 0.2, "weight_decay": 5e-2},
}
MLP_COMMON = {"lr": 1e-3, "batch_size": 256, "max_epochs": 60, "patience": 8}


# ---------------------------------------------------------------- data
def build_export(split):
    """One-off: copy merged_balanced rows' provided embeddings into compact float16 arrays."""
    EXPORT.mkdir(parents=True, exist_ok=True)
    d = json.load(open(NC / "data" / "merged_balanced" / f"{split}.json"))
    ann, src = d["annotations"], d["source_datasets"]
    meth = [src[a["source_dataset"]] if isinstance(src, list) else src[str(a["source_dataset"])] for a in ann]
    np.savez(EXPORT / f"{split}_meta.npz", id=np.array([a["id"] for a in ann]),
             image_id=np.array([a["image_id"] for a in ann]),
             y=np.array([int(a["falsified"]) for a in ann]), method=np.array(meth))
    for kind, key in [("image", "image_id"), ("text", "id")]:
        emb = pickle.load(open(NC / f"embeddings/clip_{kind}_embeddings/clip_{kind}_embeddings_{split}.pkl", "rb"))
        get = lambda k: np.asarray(emb[k] if k in emb else emb[str(k)], dtype=np.float32).reshape(-1)
        X = np.stack([get(a[key]) for a in ann])
        del emb
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        np.save(EXPORT / f"{split}_{kind}.npy", X.astype(np.float16))


def load_full(split):
    if not (EXPORT / f"{split}_image.npy").exists():
        build_export(split)
    m = np.load(EXPORT / f"{split}_meta.npz")
    df = pd.DataFrame({k: m[k] for k in ["id", "image_id", "y", "method"]})
    I = np.load(EXPORT / f"{split}_image.npy").astype(np.float32)
    T = np.load(EXPORT / f"{split}_text.npy").astype(np.float32)
    I /= np.linalg.norm(I, axis=1, keepdims=True)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    df["cos"] = (I * T).sum(1)
    return df, pair_features(I, T)


def curated_rows(full_df, split):
    """Row positions of the curated subset inside the full split, plus the curated frame (with captions)."""
    cur = pd.read_parquet(CUR / f"{split}.parquet")
    pos = pd.Series(np.arange(len(full_df)), index=pd.MultiIndex.from_arrays([full_df.id, full_df.image_id]))
    idx = pos.reindex(pd.MultiIndex.from_arrays([cur.id, cur.image_id])).values
    assert not np.isnan(idx.astype(float)).any(), f"curated {split} rows missing from full split"
    cur["y"] = cur["falsified"].astype(int)
    return idx.astype(int), cur


def stratified_fraction(df, frac, seed=SEED):
    if frac >= 1.0:
        return np.arange(len(df))
    parts = [g.sample(frac=frac, random_state=seed).index.values for _, g in df.groupby(["method", "y"])]
    return np.sort(np.concatenate(parts))


# ---------------------------------------------------------------- models
def fit_logreg(X, y, Xv, yv, grid=C_GRID):
    sweep = {}
    for C in grid:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=SEED))
        clf.fit(X, y)
        sweep[C] = (roc_auc_score(yv, clf.predict_proba(Xv)[:, 1]), roc_auc_score(y, clf.predict_proba(X)[:, 1]), clf)
        print(f"    C={C:<7} train AUC {sweep[C][1]:.4f}  val AUC {sweep[C][0]:.4f}", flush=True)
    best = max(sweep, key=lambda c: sweep[c][0])
    return sweep[best][2], best, {str(c): {"val_auc": float(v[0]), "train_auc": float(v[1])} for c, v in sweep.items()}


class MLP(nn.Module):
    def __init__(self, d_in, hidden, dropout, input_dropout):
        super().__init__()
        layers, d = [nn.Dropout(input_dropout)], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy()


def fit_mlp(cfg, X, y, Xv, yv):
    torch.manual_seed(SEED)
    model = MLP(X.shape[1], cfg["hidden"], cfg["dropout"], cfg["input_dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=MLP_COMMON["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    g = torch.Generator().manual_seed(SEED)
    hist, best, best_state, bad = [], -1, None, 0
    for ep in range(MLP_COMMON["max_epochs"]):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for b in range(0, len(perm), MLP_COMMON["batch_size"]):
            idx = perm[b:b + MLP_COMMON["batch_size"]]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        tr = roc_auc_score(y, predict(model, X))
        va = roc_auc_score(yv, predict(model, Xv))
        hist.append({"epoch": ep + 1, "train_auc": float(tr), "val_auc": float(va)})
        print(f"    epoch {ep + 1:3d}  train AUC {tr:.4f}  val AUC {va:.4f}", flush=True)
        if va > best:
            best, bad, best_state = va, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= MLP_COMMON["patience"]:
                break
    model.load_state_dict(best_state)
    return model, float(best), hist


# ---------------------------------------------------------------- analysis
def calibration(y, p, n_bins=10):
    """Expected calibration error with equal-width bins on P(falsified)."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0:
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(conf - acc)
        rows.append({"bin": b, "n": int(m.sum()), "mean_pred": conf, "frac_falsified": acc})
    return float(ece), rows


def error_analysis(cur, p, thr, k=5):
    out = {}
    pred = (p >= thr).astype(int)
    for meth, g in cur.assign(p=p, pred=pred).groupby("method"):
        fn = g[(g.y == 1) & (g.pred == 0)].sort_values("p")            # falsified, called genuine
        fp = g[(g.y == 0) & (g.pred == 1)].sort_values("p", ascending=False)  # genuine, called falsified
        out[meth] = {
            "n": int(len(g)), "missed_fakes": int(len(fn)), "false_alarms": int(len(fp)),
            "confident_errors_p_gt_0.8_or_lt_0.2": int(((g.p > 0.8) & (g.y == 0)).sum() + ((g.p < 0.2) & (g.y == 1)).sum()),
            "most_confident_missed_fakes": [{"p_falsified": round(float(r.p), 3), "caption": r.caption[:200],
                                             "image_path": r.image_path} for r in fn.head(k).itertuples()],
            "most_confident_false_alarms": [{"p_falsified": round(float(r.p), 3), "caption": r.caption[:200],
                                             "image_path": r.image_path} for r in fp.head(k).itertuples()],
        }
    return out


def main():
    t0 = time.time()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    full = {sp: load_full(sp) for sp in ["train", "val", "test"]}
    (trdf, Xtr), (vadf, Xva), (tedf, Xte) = full["train"], full["val"], full["test"]
    ytr, yva, yte = trdf.y.values, vadf.y.values, tedf.y.values
    tr_cur_idx, _ = curated_rows(trdf, "train")
    te_cur_idx, cur_test = curated_rows(tedf, "test")
    ctdf = tedf.iloc[te_cur_idx].reset_index(drop=True)
    print(f"full train {len(trdf)}, val {len(vadf)}, test {len(tedf)}; curated test {len(ctdf)} matched", flush=True)

    base_val, base_cur = -vadf.cos.values, -ctdf.cos.values
    base_thr = val_threshold(yva, base_val)

    # 1) reference: logistic regression on the curated 19,200 subset (step 5 setting, C=0.0003), float16 inputs
    print("logreg on curated subset (reference)", flush=True)
    lr_sub = make_pipeline(StandardScaler(), LogisticRegression(C=0.0003, max_iter=5000, random_state=SEED))
    lr_sub.fit(Xtr[tr_cur_idx], ytr[tr_cur_idx])
    p_sub_val, p_sub_cur = lr_sub.predict_proba(Xva)[:, 1], lr_sub.predict_proba(Xte[te_cur_idx])[:, 1]
    ev_sub = evaluate(ctdf, p_sub_cur, val_threshold(yva, p_sub_val), ref_score=base_cur)

    # 2) learning curve: logistic regression on 10% .. 100% of the full training split
    curve = []
    for frac in FRACTIONS:
        rows = stratified_fraction(trdf, frac)
        print(f"learning curve: {frac:.0%} = {len(rows)} pairs", flush=True)
        clf, C, sweep = fit_logreg(Xtr[rows], ytr[rows], Xva, yva)
        curve.append({"fraction": frac, "n_train": int(len(rows)), "best_C": C,
                      "train_auc": sweep[str(C)]["train_auc"], "val_auc": sweep[str(C)]["val_auc"],
                      "curated_test_auc": float(roc_auc_score(ctdf.y, clf.predict_proba(Xte[te_cur_idx])[:, 1])),
                      "c_sweep": sweep})
        if frac == 1.0:
            lr_full, lr_full_C = clf, C
    joblib.dump(lr_full, MODELS / "logreg_full.joblib")
    p_lr_val, p_lr_cur, p_lr_te = (lr_full.predict_proba(X)[:, 1] for X in (Xva, Xte[te_cur_idx], Xte))
    thr_lr = val_threshold(yva, p_lr_val)
    ev_lr = evaluate(ctdf, p_lr_cur, thr_lr, ref_score=base_cur)
    ev_lr_vs_sub = evaluate(ctdf, p_lr_cur, thr_lr, ref_score=p_sub_cur)

    # 3) MLP on the full split: draft config vs. two stronger-regularization configs, picked on validation AUC
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z = lambda X: ((X - mu) / sd).astype(np.float32)
    Ztr, Zva, Zte = Z(Xtr), Z(Xva), Z(Xte)
    mlp_runs, best_name, best_model, best_val = {}, None, None, -1
    for name, cfg in MLP_CONFIGS.items():
        print(f"MLP {name}: {cfg}", flush=True)
        model, va, hist = fit_mlp(cfg, Ztr, ytr, Zva, yva)
        be = max(hist, key=lambda h: h["val_auc"])
        mlp_runs[name] = {"config": {**cfg, **MLP_COMMON}, "best_epoch": be["epoch"], "epochs_run": len(hist),
                          "val_auc": va, "train_auc_at_best": be["train_auc"],
                          "n_params": int(sum(p.numel() for p in model.parameters())), "history": hist}
        if va > best_val:
            best_name, best_model, best_val = name, model, va
    torch.save({"state_dict": best_model.state_dict(), "config": {**MLP_CONFIGS[best_name], **MLP_COMMON},
                "feature_mean": mu, "feature_std": sd}, MODELS / "mlp_full.pt")
    p_mlp_val, p_mlp_cur, p_mlp_te = predict(best_model, Zva), predict(best_model, Zte[te_cur_idx]), predict(best_model, Zte)
    thr_mlp = val_threshold(yva, p_mlp_val)
    ev_mlp = evaluate(ctdf, p_mlp_cur, thr_mlp, ref_score=base_cur)
    ev_mlp_vs_lr = evaluate(ctdf, p_mlp_cur, thr_mlp, ref_score=p_lr_cur)
    for g in ev_mlp:
        ev_mlp[g]["auc_minus_logreg_full"] = ev_mlp_vs_lr[g]["auc_minus_baseline"]
    for g in ev_lr:
        ev_lr[g]["auc_minus_logreg_subset"] = ev_lr_vs_sub[g]["auc_minus_baseline"]

    # full official test split (7,264 pairs), secondary
    ev_full_test = {"logreg_full": evaluate(tedf, p_lr_te, thr_lr, ref_score=-tedf.cos.values),
                    "mlp_full": evaluate(tedf, p_mlp_te, thr_mlp, ref_score=-tedf.cos.values)}

    # 4) calibration + error analysis on the curated test set (captions available there)
    best_kind = "mlp_full" if roc_auc_score(yva, p_mlp_val) > roc_auc_score(yva, p_lr_val) else "logreg_full"
    p_best, thr_best = (p_mlp_cur, thr_mlp) if best_kind == "mlp_full" else (p_lr_cur, thr_lr)
    cal = {}
    for name, p in [("logreg_full", p_lr_cur), ("mlp_full", p_mlp_cur)]:
        e, rows = calibration(ctdf.y.values, p)
        cal[name] = {"ece": e, "bins": rows}
    cur_aligned = cur_test.reset_index(drop=True)
    errors = error_analysis(cur_aligned, p_best, thr_best)

    res = {
        "data": {"train_full": int(len(trdf)), "val_full": int(len(vadf)), "test_full": int(len(tedf)),
                 "test_curated": int(len(ctdf)), "embeddings": "provided CLIP ViT-B/32, stored float16, L2-normalised"},
        "logreg_subset_reference": {"C": 0.0003, "test": ev_sub},
        "learning_curve_logreg": curve,
        "logreg_full": {"best_C": lr_full_C, "val_roc_auc": float(roc_auc_score(yva, p_lr_val)),
                        "decision_threshold_prob": thr_lr, "test": ev_lr},
        "mlp_full": {"runs": mlp_runs, "selected": best_name, "decision_threshold_prob": thr_mlp, "test": ev_mlp},
        "full_test_split": ev_full_test,
        "calibration_curated_test": cal,
        "error_analysis": {"model": best_kind, **errors},
        "baseline_same_bootstrap": evaluate(ctdf, base_cur, base_thr),
        "seconds": time.time() - t0,
    }
    update_results("full_split", res)

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    n = [c["n_train"] for c in curve]
    ax.plot(n, [c["train_auc"] for c in curve], "o-", label="train AUC")
    ax.plot(n, [c["val_auc"] for c in curve], "o-", label="validation AUC")
    ax.plot(n, [c["curated_test_auc"] for c in curve], "o-", label="test AUC (curated 2,400)")
    ax.axhline(0.8, ls="--", c="#4a9d5b", lw=0.8, label="target 0.80")
    ax.set_xscale("log")
    ax.set_xlabel("training pairs (log scale)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Logistic regression learning curve", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_learning_curve.png", dpi=150)
    plt.close(fig)

    plot_auc_bars({"Cosine baseline": res["baseline_same_bootstrap"], "LogReg, 19.2k subset": ev_sub,
                   "LogReg, full 71k": ev_lr, "MLP, full 71k": ev_mlp},
                  "Test ROC-AUC per mismatch method: subset vs. full training split", FIG / "fig_auc_full_vs_subset.png")

    fig, ax = plt.subplots(figsize=(4, 4))
    for name, c in cal.items():
        ax.plot([b["mean_pred"] for b in c["bins"]], [b["frac_falsified"] for b in c["bins"]], "o-",
                label=f"{name} (ECE {c['ece']:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", c="grey")
    ax.set_xlabel("predicted P(falsified)")
    ax.set_ylabel("observed fraction falsified")
    ax.set_title("Reliability diagram, curated test", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_calibration.png", dpi=150)
    plt.close(fig)

    print(f"\nsubset LR reference: AUC {ev_sub['all']['roc_auc']:.4f}")
    for c in curve:
        print(f"curve {c['n_train']:6d}: C {c['best_C']}  train {c['train_auc']:.4f}  val {c['val_auc']:.4f}  test {c['curated_test_auc']:.4f}")
    for label, ev, key in [("LR full", ev_lr, "auc_minus_logreg_subset"), ("MLP full", ev_mlp, "auc_minus_logreg_full")]:
        for g, r in ev.items():
            d = r[key]
            print(f"{label:9s} {g:28s} AUC {r['roc_auc']:.3f} [{r['roc_auc_ci95'][0]:.3f}, {r['roc_auc_ci95'][1]:.3f}] "
                  f"acc {r['accuracy']:.3f} F1 {r['f1']:.3f} | {key} {d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
    print(f"MLP selected {best_name}; ECE " + ", ".join(f"{k} {v['ece']:.3f}" for k, v in cal.items()))
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
