"""Step 10: model stability, ROC curves and operating points (instructor feedback on Progress Report 4 and the midterm).

"Model stability is more important than just a higher ROC-AUC", "create a curve visualization for the ROC-AUC".
Runs after step 9 and reuses its cached swap-aware features (data/step9_swap_features.npz) and saved scores
(outputs/metrics/step9_scores.npz). Hyperparameters are the ones step 9 selected on validation; nothing here is
tuned on the test set.

1. Seed stability: the main models are retrained with 5 seeds (initialisation, shuffling, bagging and column
   sampling change; data and settings do not). Reported: mean, standard deviation, min and max of validation and
   curated-test ROC-AUC, overall and per mismatch method, and the train - validation AUC gap (overfitting).
2. Grouped 5-fold cross-validation on the 71k training split: folds are connected components of shared image and
   caption ids, so no image or caption is in both the training folds and the held-out fold. AUC of each held-out
   fold, overall and per method, mean and standard deviation. Early stopping uses the official validation split.
3. ROC curves of every step 9 model, overall and per method, plus a larger panel for the CLIP text-image swaps.
4. Operating points: the threshold that gives 5% and 10% false alarms on validation genuine pairs, applied to test:
   share of fakes caught (TPR), false-alarm rate (FPR) and precision, overall and per method.

Outputs: results.json key "stability_roc"; figures fig_roc_all_models.png, fig_roc_swap_subset.png,
fig_stability.png.
"""
import importlib
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import FIG, METRICS, SEED, plot_roc_by_method, update_results

s8 = importlib.import_module("08_full_split")
s9 = importlib.import_module("09_nonlinear_and_swaps")

FAST = os.environ.get("STEP9_FAST") == "1"
SEEDS = [SEED + k for k in range(3 if FAST else 5)]
N_FOLDS = 3 if FAST else 5
SWAP = s9.SWAP
FPR_TARGETS = [0.05, 0.10]


# ---------------------------------------------------------------- helpers
def group_aucs(y, p, methods):
    out = {"all": float(roc_auc_score(y, p))}
    for m in sorted(np.unique(methods)):
        k = methods == m
        if len(np.unique(y[k])) == 2:      # skip a group with one class only (cannot compute AUC)
            out[m] = float(roc_auc_score(y[k], p[k]))
    return out


def summarize(runs):
    """runs: list of {group: auc}. -> {group: {mean, sd, min, max, values}}."""
    out = {}
    for g in [g for g in runs[0] if all(g in r for r in runs)]:
        v = np.array([r[g] for r in runs])
        out[g] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "min": float(v.min()), "max": float(v.max()), "values": [float(x) for x in v]}
    return out


def operating_points(y_val, s_val, y_te, s_te, meth_te):
    """Thresholds fixed on validation genuine pairs (y=0) for a target false-alarm rate, applied to test."""
    out = {}
    neg = s_val[y_val == 0]
    for f in FPR_TARGETS:
        thr = float(np.quantile(neg, 1 - f))
        res = {"threshold": thr}
        for g in ["all"] + sorted(np.unique(meth_te)):
            k = np.ones(len(y_te), bool) if g == "all" else meth_te == g
            yy, pred = y_te[k], s_te[k] >= thr
            tp, fp = int((pred & (yy == 1)).sum()), int((pred & (yy == 0)).sum())
            res[g] = {"tpr_fakes_caught": tp / max(int((yy == 1).sum()), 1),
                      "fpr_false_alarms": fp / max(int((yy == 0).sum()), 1),
                      "precision": tp / max(tp + fp, 1)}
        out[f"val_fpr_{int(f * 100)}pct"] = res
    return out


def mlp_run(X, y, Xv, yv, Xeval, w=None, seed=SEED, n_train_auc=20000):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = lambda A: ((A - mu) / sd).astype(np.float32)
    Ztr = Z(X)
    model, va, hist = s9.fit_mlp_weighted(s9.MLP_CFG, Ztr, y, Z(Xv), yv, w=w, seed=seed)
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(y), min(n_train_auc, len(y)), replace=False)
    tr = float(roc_auc_score(y[sub], s8.predict(model, Ztr[sub])))
    best_ep = max(hist, key=lambda h: h["val_auc"])["epoch"]
    return {"val": s8.predict(model, Z(Xv)), "eval": [s8.predict(model, Z(A)) for A in Xeval],
            "train_auc": tr, "best_epoch": best_ep}


def lgbm_run(X, y, Xv, yv, Xeval, leaves, seed=SEED, n_train_auc=20000):
    m = s9.fit_lgbm(X, y, Xv, yv, leaves, seed=seed)
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(y), min(n_train_auc, len(y)), replace=False)
    return {"val": m.predict_proba(Xv)[:, 1], "eval": [m.predict_proba(A)[:, 1] for A in Xeval],
            "train_auc": float(roc_auc_score(y[sub], m.predict_proba(X[sub])[:, 1])),
            "best_iteration": int(m.best_iteration_ or 0)}


def rf_run(X, y, Xv, Xeval, leaf, seed=SEED):
    rf = RandomForestClassifier(n_estimators=50 if FAST else 400, max_features="sqrt", min_samples_leaf=leaf,
                                n_jobs=-1, random_state=seed, oob_score=True)
    rf.fit(X, y)
    out = {"val": rf.predict_proba(Xv)[:, 1], "eval": [rf.predict_proba(A)[:, 1] for A in Xeval],
           "train_auc": float(roc_auc_score(y, rf.oob_decision_function_[:, 1])),
           "train_auc_note": "out-of-bag AUC (in-sample AUC of a forest is ~1 and not informative)"}
    return out


def main():
    t0 = time.time()
    allres = json.load(open(METRICS / "results.json"))
    s9res = allres["nonlinear_and_swaps"]
    rf_leaf = s9res["classical_nonlinear"]["random_forest"]["selected_min_samples_leaf"]
    lgb_leaves = s9res["classical_nonlinear"]["lightgbm"]["selected_num_leaves"]
    lgbE_leaves = s9res["swap_aware"]["lightgbm_swapaware"]["selected_num_leaves"]
    swap_w = s9res["swap_aware"]["ensemble_swapaware"]["mlp_swap_weight"]
    lr_C = allres["full_split"]["logreg_full"]["best_C"]

    full = {sp: s8.load_full(sp) for sp in ["train", "val", "test"]}
    (trdf, Xtr), (vadf, Xva), (tedf, Xte) = full["train"], full["val"], full["test"]
    ytr, yva = trdf.y.values, vadf.y.values
    te_cur_idx, _ = s8.curated_rows(tedf, "test")
    ctdf = tedf.iloc[te_cur_idx].reset_index(drop=True)
    yte, mte, mva, mtr = ctdf.y.values, ctdf.method.values.astype(str), vadf.method.values.astype(str), \
        trdf.method.values.astype(str)
    Xcur = Xte[te_cur_idx]
    cache = np.load(s8.EXPORT.parent / "step9_swap_features.npz")
    XtrE, XvaE = np.hstack([Xtr, cache["train"]]), np.hstack([Xva, cache["val"]])
    XcurE = np.hstack([Xte, cache["test"]])[te_cur_idx]
    comps = cache["components"]
    w_swap = np.where(mtr == SWAP, swap_w, 1.0)
    res = {"settings": {"seeds": SEEDS, "folds": N_FOLDS, "rf_min_samples_leaf": rf_leaf, "lgbm_num_leaves": lgb_leaves,
                        "lgbm_swapaware_num_leaves": lgbE_leaves, "mlp_swap_weight": swap_w, "logreg_C": lr_C}}

    # ============================ 1. seed stability
    print("1) seed stability", flush=True)
    specs = {
        "mlp_step8": lambda sd: mlp_run(Xtr, ytr, Xva, yva, [Xcur], seed=sd),
        "mlp_swapaware": lambda sd: mlp_run(XtrE, ytr, XvaE, yva, [XcurE], w=w_swap, seed=sd),
        "lightgbm": lambda sd: lgbm_run(Xtr, ytr, Xva, yva, [Xcur], lgb_leaves, seed=sd),
        "lightgbm_swapaware": lambda sd: lgbm_run(XtrE, ytr, XvaE, yva, [XcurE], lgbE_leaves, seed=sd),
        "random_forest": lambda sd: rf_run(Xtr, ytr, Xva, [Xcur], rf_leaf, seed=sd),
    }
    seed_res, seed_preds = {}, {}
    for name, fn in specs.items():
        runs_val, runs_test, gaps, extra = [], [], [], []
        preds = []
        for sd in SEEDS:
            t = time.time()
            r = fn(sd)
            va = float(roc_auc_score(yva, r["val"]))
            runs_val.append({"all": va, SWAP: float(roc_auc_score(yva[mva == SWAP], r["val"][mva == SWAP]))})
            runs_test.append(group_aucs(yte, r["eval"][0], mte))
            gaps.append(r["train_auc"] - va)
            extra.append({k: v for k, v in r.items() if k in ("best_epoch", "best_iteration", "train_auc")})
            preds.append(r["eval"][0])
            print(f"  {name:20s} seed {sd}: val {va:.4f}  test {runs_test[-1]['all']:.4f}  "
                  f"swaps {runs_test[-1][SWAP]:.4f}  train-val gap {gaps[-1]:+.3f}  ({time.time() - t:.0f}s)", flush=True)
        seed_preds[name] = np.mean(preds, 0)
        seed_res[name] = {"val": summarize(runs_val), "test": summarize(runs_test),
                          "train_minus_val_auc": {"mean": float(np.mean(gaps)), "sd": float(np.std(gaps, ddof=1)),
                                                  "values": [float(g) for g in gaps]},
                          "per_seed": extra,
                          "seed_averaged_test_auc": group_aucs(yte, seed_preds[name], mte)}
    res["seed_stability"] = seed_res

    # ============================ 2. grouped k-fold CV on the training split
    print(f"2) grouped {N_FOLDS}-fold cross-validation", flush=True)
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(Xtr, ytr, groups=comps))
    cv_specs = {
        "logreg": lambda tr, ho: make_pipeline(StandardScaler(), LogisticRegression(C=lr_C, max_iter=5000, random_state=SEED))
        .fit(Xtr[tr], ytr[tr]).predict_proba(Xtr[ho])[:, 1],
        "lightgbm": lambda tr, ho: lgbm_run(Xtr[tr], ytr[tr], Xva, yva, [Xtr[ho]], lgb_leaves)["eval"][0],
        "lightgbm_swapaware": lambda tr, ho: lgbm_run(XtrE[tr], ytr[tr], XvaE, yva, [XtrE[ho]], lgbE_leaves)["eval"][0],
        "mlp_step8": lambda tr, ho: mlp_run(Xtr[tr], ytr[tr], Xva, yva, [Xtr[ho]])["eval"][0],
        "mlp_swapaware": lambda tr, ho: mlp_run(XtrE[tr], ytr[tr], XvaE, yva, [XtrE[ho]], w=w_swap[tr])["eval"][0],
    }
    cv_res = {}
    for name, fn in cv_specs.items():
        runs = []
        for k, (tr, ho) in enumerate(folds):
            t = time.time()
            p = fn(tr, ho)
            runs.append(group_aucs(ytr[ho], p, mtr[ho]))
            print(f"  {name:20s} fold {k + 1}: AUC {runs[-1]['all']:.4f}  swaps {runs[-1][SWAP]:.4f} "
                  f"({time.time() - t:.0f}s)", flush=True)
        cv_res[name] = summarize(runs)
    res["grouped_cv"] = {"note": "held-out folds of the training split; groups = connected components of shared "
                                 "image/caption ids; hyperparameters from step 9; early stopping on the validation split",
                         "fold_sizes": [int(len(ho)) for _, ho in folds],
                         "largest_component_pairs": int(np.bincount(pd.factorize(comps)[0]).max()), "models": cv_res}

    # ============================ 3. ROC curves and 4. operating points (step 9 scores)
    print("3) ROC curves and operating points", flush=True)
    sc = np.load(METRICS / "step9_scores.npz", allow_pickle=False)
    names = [str(n) for n in sc["names"]]
    assert np.array_equal(sc["y_test"], yte), "step 9 scores are not aligned with the curated test set"
    roc_models = {"LogReg (linear)": "LogReg full (step 8)", "Random Forest": "Random Forest", "LightGBM": "LightGBM",
                  "RBF SVM": "RBF SVM (15k subsample)", "MLP (step 8)": "MLP full (step 8)"}
    final_key = s9res["final_model"]["name"]
    final_disp = {"lightgbm_swapaware": "LightGBM + swap features", "ensemble_swapaware": "Ensemble (MLP + LightGBM)",
                  "logreg_swapaware": "LogReg + swap features"}.get(final_key)
    if final_disp is None:      # one of the MLP swap-weight variants
        final_disp = next(n for n in names if n.startswith("MLP + swap feats, w=") and
                          f"w{n.split('w=')[1]}" == final_key.replace("mlp_swapaware_", ""))
    roc_models["Swap-aware final"] = final_disp
    scores = {lab: sc[f"test__{key}"] for lab, key in roc_models.items() if key in names}
    plot_roc_by_method(ctdf, scores, "ROC curves on the curated test set (2,400 pairs), dashed = cosine baseline",
                       FIG / "fig_roc_all_models.png", ref_score=sc["test__cosine_baseline"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = mte == SWAP
    fig, ax = plt.subplots(figsize=(5, 5))
    for lab, s in scores.items():
        fpr, tpr, _ = roc_curve(yte[k], s[k])
        ax.plot(fpr, tpr, lw=1.8 if lab == "Swap-aware final" else 1.2,
                label=f"{lab}  AUC {roc_auc_score(yte[k], s[k]):.3f}")
    fpr, tpr, _ = roc_curve(yte[k], sc["test__cosine_baseline"][k])
    ax.plot(fpr, tpr, ls="--", c="grey", label=f"Cosine baseline  AUC {roc_auc_score(yte[k], sc['test__cosine_baseline'][k]):.3f}")
    ax.plot([0, 1], [0, 1], ls=":", c="lightgrey")
    ax.set_xlabel("false-alarm rate (genuine pairs flagged)")
    ax.set_ylabel("fakes caught (true positive rate)")
    ax.set_title("CLIP text-image swaps: ROC curves (600 test pairs)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_roc_swap_subset.png", dpi=150)
    plt.close(fig)

    ops = {}
    for lab, key in roc_models.items():
        if key in names:
            ops[lab] = operating_points(sc["y_val"], sc[f"val__{key}"], yte, sc[f"test__{key}"], mte)
    ops["Cosine baseline"] = operating_points(sc["y_val"], sc["val__cosine_baseline"], yte, sc["test__cosine_baseline"], mte)
    res["operating_points"] = ops
    res["roc_models"] = roc_models

    # stability figure: seed mean +- sd (test) and CV mean +- sd, overall and swaps
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    for ax, g, title in [(axes[0], "all", "All mismatch types"), (axes[1], SWAP, "CLIP text-image swaps")]:
        labels = list(dict.fromkeys(list(seed_res) + list(cv_res)))
        y = np.arange(len(labels))
        for off, src, lab, c in [(-0.15, seed_res, f"{len(SEEDS)} seeds, curated test", "#2a78b5"),
                                 (0.15, cv_res, f"{N_FOLDS}-fold grouped CV", "#d1495b")]:
            for i, n in enumerate(labels):
                if n not in src:
                    continue
                st = src[n]["test"][g] if src is seed_res else src[n][g]
                ax.errorbar(st["mean"], i + off, xerr=st["sd"], fmt="o", c=c, capsize=3,
                            label=lab if i == 0 or (n == labels[0]) else None)
        if g == SWAP:
            ax.axvline(0.7, ls="--", c="#4a9d5b", lw=0.8, label="0.70 target")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("ROC-AUC (mean ± 1 sd)", fontsize=8)
        ax.set_title(title, fontsize=9)
        h, l = ax.get_legend_handles_labels()
        ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Stability across seeds and cross-validation folds", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_stability.png", dpi=150)
    plt.close(fig)

    res["seconds"] = time.time() - t0
    update_results("stability_roc", res)

    print("\nsummary (curated test, mean ± sd over seeds | grouped CV mean ± sd)")
    for n in seed_res:
        a, s_ = seed_res[n]["test"]["all"], seed_res[n]["test"][SWAP]
        cv = cv_res.get(n)
        cvs = f"| CV {cv['all']['mean']:.3f} ± {cv['all']['sd']:.3f}, swaps {cv[SWAP]['mean']:.3f} ± {cv[SWAP]['sd']:.3f}" if cv else ""
        print(f"  {n:20s} test {a['mean']:.3f} ± {a['sd']:.3f}  swaps {s_['mean']:.3f} ± {s_['sd']:.3f}  "
              f"gap {seed_res[n]['train_minus_val_auc']['mean']:+.3f} {cvs}")
    for lab, o in ops.items():
        p = o["val_fpr_10pct"]["all"]
        print(f"  at 10% false alarms (val): {lab:22s} catches {p['tpr_fakes_caught']:.1%} of fakes, "
              f"test FPR {p['fpr_false_alarms']:.1%}, precision {p['precision']:.1%}")
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
