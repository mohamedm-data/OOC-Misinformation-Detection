"""Step 5: logistic regression on frozen CLIP ViT-B/32 pair features.

Features: [img, txt, img*txt, |img-txt|, cosine] (2049-d) from the provided, L2-normalised embeddings.
StandardScaler + LogisticRegression, C tuned on validation ROC-AUC, refit on train with the best C.
Test metrics overall and per mismatch method with 95% bootstrap CIs (1000 resamples, seed 5500), plus a
paired bootstrap of the AUC difference vs. the cosine-similarity baseline on the same test rows.

Outputs: outputs/models/logreg.joblib, results.json key "logistic_regression",
         outputs/figures/fig_roc_logreg_by_method.png, fig_auc_logreg_vs_baseline.png
"""
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import FIG, MODELS, SEED, evaluate, load_split, pair_features, plot_auc_bars, plot_roc_by_method, \
    update_results, val_threshold

C_GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]


def main():
    t0 = time.time()
    data = {sp: load_split(sp) for sp in ["train", "val", "test"]}
    X = {sp: pair_features(I, T) for sp, (_, I, T) in data.items()}
    y = {sp: df.y.values for sp, (df, _, _) in data.items()}

    sweep = {}
    for C in C_GRID:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=SEED))
        clf.fit(X["train"], y["train"])
        sweep[C] = {"val_auc": float(roc_auc_score(y["val"], clf.predict_proba(X["val"])[:, 1])),
                    "train_auc": float(roc_auc_score(y["train"], clf.predict_proba(X["train"])[:, 1]))}
        print(f"C={C:<7} train AUC {sweep[C]['train_auc']:.4f}  val AUC {sweep[C]['val_auc']:.4f}")
    best_C = max(sweep, key=lambda c: sweep[c]["val_auc"])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=best_C, max_iter=5000, random_state=SEED))
    clf.fit(X["train"], y["train"])
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODELS / "logreg.joblib")

    vdf, tdf = data["val"][0], data["test"][0]
    p_val, p_test = clf.predict_proba(X["val"])[:, 1], clf.predict_proba(X["test"])[:, 1]
    thr = val_threshold(y["val"], p_val)
    base_thr = val_threshold(y["val"], -vdf.cos.values)
    base_test = -tdf.cos.values
    ev = evaluate(tdf, p_test, thr, ref_score=base_test)
    ev_base = evaluate(tdf, base_test, base_thr)

    res = {
        "features": "[img, txt, img*txt, |img-txt|, cosine] (2049-d), provided L2-normalised CLIP ViT-B/32",
        "model": "StandardScaler + LogisticRegression (lbfgs, L2)",
        "C_grid_val_auc": {str(c): v for c, v in sweep.items()},
        "best_C": best_C,
        "val_roc_auc": float(roc_auc_score(y["val"], p_val)),
        "decision_threshold_prob": thr,
        "bootstrap": {"n_resamples": 1000, "seed": SEED, "ci": "percentile 95%",
                      "paired": "AUC difference vs. cosine baseline on identical resampled test rows"},
        "test": ev,
        "baseline_same_bootstrap": ev_base,
        "seconds": time.time() - t0,
    }
    update_results("logistic_regression", res)

    plot_roc_by_method(tdf, {"LogReg": p_test}, "Logistic regression vs. cosine baseline, ROC on test",
                       FIG / "fig_roc_logreg_by_method.png", ref_score=base_test)
    plot_auc_bars({"Cosine baseline": ev_base, "Logistic regression": ev},
                  "Test ROC-AUC per mismatch method: logistic regression vs. cosine baseline",
                  FIG / "fig_auc_logreg_vs_baseline.png")

    print(f"best C {best_C}; val AUC {res['val_roc_auc']:.4f}; threshold {thr:.3f}")
    for g, r in ev.items():
        d = r["auc_minus_baseline"]
        print(f"{g:28s} AUC {r['roc_auc']:.3f} [{r['roc_auc_ci95'][0]:.3f}, {r['roc_auc_ci95'][1]:.3f}]  "
              f"acc {r['accuracy']:.3f}  F1 {r['f1']:.3f}  | base {ev_base[g]['roc_auc']:.3f}  "
              f"delta {d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
