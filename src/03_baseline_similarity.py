"""Step 3 (+ week 6): CLIP cosine-similarity threshold baseline on the curated subset.

Uses the provided NewsCLIPpings CLIP ViT-B/32 embeddings (text keyed by caption id, image keyed by image_id).
Outputs figures to outputs/figures/ and numbers to outputs/metrics/results.json (key "baseline_similarity").
"""
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
NC = ROOT / "data" / "news_clippings"
CUR = ROOT / "data" / "curated"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
METRICS = OUT / "metrics"
SPLITS = ["train", "val", "test"]
SEED = 5500
COL = {"genuine": "#2a78b5", "falsified": "#d1495b"}


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def lookup(emb, key):
    v = emb.get(key)
    if v is None:
        v = emb.get(str(key))
    return np.asarray(v, dtype=np.float32).reshape(-1)


def attach_embeddings(df, split):
    img = load_pkl(NC / "embeddings/clip_image_embeddings" / f"clip_image_embeddings_{split}.pkl")
    txt = load_pkl(NC / "embeddings/clip_text_embeddings" / f"clip_text_embeddings_{split}.pkl")
    I = np.stack([lookup(img, k) for k in df["image_id"]])
    T = np.stack([lookup(txt, k) for k in df["id"]])
    I /= np.linalg.norm(I, axis=1, keepdims=True)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    return I, T


def metrics(y, pred):
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0), "n": int(len(y))}


def pick_threshold(y, score):
    """Score = -cosine (higher = more likely falsified). Return threshold maximizing val accuracy, and Youden J one."""
    cand = np.unique(score)
    if len(cand) > 4000:
        cand = np.quantile(score, np.linspace(0, 1, 4001))
    accs = [((score >= t).astype(int) == y).mean() for t in cand]
    t_acc = float(cand[int(np.argmax(accs))])
    fpr, tpr, thr = roc_curve(y, score)
    t_j = float(thr[int(np.argmax(tpr - fpr))])
    return t_acc, t_j


def bootstrap_auc(y, s, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        vals.append(roc_auc_score(y[b], s[b]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    data = {}
    for sp in SPLITS:
        df = pd.read_parquet(CUR / f"{sp}.parquet")
        I, T = attach_embeddings(df, sp)
        df["cos"] = (I * T).sum(1)
        df["y"] = df["falsified"].astype(int)
        data[sp] = (df, I, T)
    methods = sorted(data["test"][0]["method"].unique())
    res = {"n": {sp: int(len(data[sp][0])) for sp in SPLITS}, "methods": methods}

    # summary stats of cosine similarity
    stats = {}
    for sp in SPLITS:
        df = data[sp][0]
        stats[sp] = {}
        for m in ["all"] + methods:
            d = df if m == "all" else df[df.method == m]
            stats[sp][m] = {lab: {"mean": float(g.mean()), "std": float(g.std()), "median": float(g.median()), "n": int(len(g))}
                            for lab, g in [("genuine", d.loc[d.y == 0, "cos"]), ("falsified", d.loc[d.y == 1, "cos"])]}
    res["cosine_stats"] = stats

    # histograms per method (test split)
    df = data["test"][0]
    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(4.2 * (len(methods) + 1), 3.4), sharey=False)
    bins = np.linspace(df.cos.min(), df.cos.max(), 40)
    for ax, m in zip(axes, ["all"] + methods):
        d = df if m == "all" else df[df.method == m]
        for lab, yv in [("genuine", 0), ("falsified", 1)]:
            ax.hist(d.loc[d.y == yv, "cos"], bins=bins, alpha=0.6, color=COL[lab], label=lab, density=True)
        ax.set_title(m if m != "all" else "all methods", fontsize=9)
        ax.set_xlabel("CLIP cosine(image, caption)")
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Image–caption cosine similarity, test split (curated)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cosine_hist_by_method.png", dpi=150)
    plt.close(fig)

    # ROC-AUC (score = -cos), thresholds from val, test metrics
    vdf, tdf = data["val"][0], data["test"][0]
    t_acc, t_j = pick_threshold(vdf.y.values, -vdf.cos.values)
    res["threshold"] = {"val_max_accuracy_cos": -t_acc, "val_youden_j_cos": -t_j,
                        "rule": "predict falsified if cosine <= threshold"}
    auc, ci, test_m, test_m_j, val_m = {}, {}, {}, {}, {}
    for m in ["all"] + methods:
        d = tdf if m == "all" else tdf[tdf.method == m]
        dv = vdf if m == "all" else vdf[vdf.method == m]
        y, s = d.y.values, -d.cos.values
        auc[m] = float(roc_auc_score(y, s))
        ci[m] = bootstrap_auc(y, s)
        test_m[m] = metrics(y, (s >= t_acc).astype(int))
        test_m_j[m] = metrics(y, (s >= t_j).astype(int))
        val_m[m] = {"auc": float(roc_auc_score(dv.y.values, -dv.cos.values)),
                    **metrics(dv.y.values, (-dv.cos.values >= t_acc).astype(int))}
    res["test_roc_auc"] = auc
    res["test_roc_auc_bootstrap95ci"] = ci
    res["test_metrics_threshold_val_acc"] = test_m
    res["test_metrics_threshold_youden"] = test_m_j
    res["val_metrics_threshold_val_acc"] = val_m

    # per-method oracle thresholds (diagnostic only: chosen on val per method)
    per_m = {}
    for m in methods:
        dv, d = vdf[vdf.method == m], tdf[tdf.method == m]
        tm, _ = pick_threshold(dv.y.values, -dv.cos.values)
        per_m[m] = {"cos_threshold": -tm, **metrics(d.y.values, (-d.cos.values >= tm).astype(int))}
    res["test_metrics_per_method_threshold"] = per_m

    # ROC curves figure
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for m in ["all"] + methods:
        d = tdf if m == "all" else tdf[tdf.method == m]
        fpr, tpr, _ = roc_curve(d.y.values, -d.cos.values)
        ax.plot(fpr, tpr, lw=2 if m == "all" else 1.3, label=f"{m} (AUC {auc[m]:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Cosine-similarity baseline ROC (test)", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_roc_baseline.png", dpi=150)
    plt.close(fig)

    # (week 6 b) bar chart AUC + F1 per method with CI on AUC
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    labs = ["all"] + methods
    x = np.arange(len(labs))
    a = np.array([auc[m] for m in labs])
    err = np.array([[auc[m] - ci[m][0], ci[m][1] - auc[m]] for m in labs]).T
    f = np.array([test_m[m]["f1"] for m in labs])
    ax.bar(x - 0.2, a, 0.38, yerr=err, capsize=3, color="#2a78b5", label="ROC-AUC (95% bootstrap CI)")
    ax.bar(x + 0.2, f, 0.38, color="#e8a33d", label="F1 (global val threshold)")
    for xi, (ai, fi) in enumerate(zip(a, f)):
        ax.text(xi - 0.2, err[1][xi] + ai + 0.015, f"{ai:.2f}", ha="center", fontsize=7)
        ax.text(xi + 0.2, fi + 0.01, f"{fi:.2f}", ha="center", fontsize=7)
    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", "\n", 1) for l in labs], fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_title("Cosine-similarity baseline per method (test)", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig_baseline_auc_f1_by_method.png", dpi=150)
    plt.close(fig)

    # (week 6 a) 2-D embedding projection of ~2,000 test pairs: difference vector (image - text)
    df, I, T = data["test"]
    rng = np.random.default_rng(SEED)
    sel = np.sort(rng.choice(len(df), size=min(2000, len(df)), replace=False))
    sub = df.iloc[sel].reset_index(drop=True)
    proj = {}
    for name, X in [("diff", I[sel] - T[sel]), ("concat", np.hstack([I[sel], T[sel]]))]:
        p = PCA(n_components=2, random_state=SEED).fit(X)
        proj[f"pca_{name}"] = (p.transform(X), p.explained_variance_ratio_.tolist())
        Z50 = PCA(n_components=50, random_state=SEED).fit_transform(X)
        proj[f"tsne_{name}"] = (TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(Z50), None)
    res["projection"] = {"n_points": int(len(sub)), "pca_explained_variance": {k: v[1] for k, v in proj.items() if v[1]}}
    # quantitative separability in 2-D: linear probe AUC on 2-D coordinates (5-fold CV)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    sep = {}
    for k, (Z, _) in proj.items():
        pr = cross_val_predict(LogisticRegression(max_iter=1000), Z, sub.y.values, cv=5, method="predict_proba")[:, 1]
        sep[k] = float(roc_auc_score(sub.y.values, pr))
    res["projection"]["cv_logreg_auc_on_2d_coords"] = sep
    for k in ["pca_diff", "tsne_diff", "pca_concat", "tsne_concat"]:
        Z = proj[k][0]
        fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.8))
        for ax, m in zip(axes, methods):
            msk = (sub.method == m).values
            ax.scatter(Z[~msk, 0], Z[~msk, 1], s=2, c="#dddddd", lw=0)
            for lab, yv in [("genuine", 0), ("falsified", 1)]:
                mm = msk & (sub.y.values == yv)
                ax.scatter(Z[mm, 0], Z[mm, 1], s=5, c=COL[lab], alpha=0.7, lw=0, label=lab)
            ax.set_title(f"{m} (n={msk.sum()})", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        axes[0].legend(frameon=False, fontsize=7, markerscale=3)
        meth, vec = k.split("_")
        fig.suptitle(f"{meth.upper()} of {'image − text' if vec == 'diff' else '[image ; text]'} CLIP embeddings, "
                     f"{len(sub)} test pairs (grey = other methods)", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / f"fig_embed2d_{k}.png", dpi=150)
        plt.close(fig)

    path = METRICS / "results.json"
    allres = json.load(open(path)) if path.exists() else {}
    allres["baseline_similarity"] = res
    json.dump(allres, open(path, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ["n", "threshold", "test_roc_auc", "test_roc_auc_bootstrap95ci"]}, indent=1))
    for m in ["all"] + methods:
        print(m, {k: round(v, 4) for k, v in test_m[m].items()})
    print("projection", res["projection"])


if __name__ == "__main__":
    main()
