"""Shared helpers for the trained classifiers (steps 5 and 7): data loading, pair features,
bootstrap confidence intervals, per-method evaluation, results.json updates."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
NC = ROOT / "data" / "news_clippings"
CUR = ROOT / "data" / "curated"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
METRICS = OUT / "metrics"
MODELS = OUT / "models"
SPLITS = ["train", "val", "test"]
SEED = 5500
N_BOOT = 1000


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _lookup(emb, key):
    v = emb.get(key)
    if v is None:
        v = emb.get(str(key))
    return np.asarray(v, dtype=np.float32).reshape(-1)


def load_split(split):
    """Curated split with L2-normalised provided CLIP ViT-B/32 image (I) and caption (T) embeddings."""
    df = pd.read_parquet(CUR / f"{split}.parquet")
    img = _load_pkl(NC / "embeddings/clip_image_embeddings" / f"clip_image_embeddings_{split}.pkl")
    txt = _load_pkl(NC / "embeddings/clip_text_embeddings" / f"clip_text_embeddings_{split}.pkl")
    I = np.stack([_lookup(img, k) for k in df["image_id"]])
    T = np.stack([_lookup(txt, k) for k in df["id"]])
    I /= np.linalg.norm(I, axis=1, keepdims=True)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    df["y"] = df["falsified"].astype(int)
    df["cos"] = (I * T).sum(1)
    return df, I, T


def pair_features(I, T):
    """[img, txt, img*txt, |img-txt|, cosine] -> 4*512 + 1 = 2049 features."""
    return np.hstack([I, T, I * T, np.abs(I - T), (I * T).sum(1, keepdims=True)]).astype(np.float32)


def val_threshold(y, score):
    """Threshold on a score (higher = falsified) that maximises validation accuracy."""
    cand = np.unique(score)
    if len(cand) > 4000:
        cand = np.quantile(score, np.linspace(0, 1, 4001))
    accs = [((score >= t).astype(int) == y).mean() for t in cand]
    return float(cand[int(np.argmax(accs))])


def _ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def evaluate(df, score, thr, ref_score=None, n_boot=N_BOOT, seed=SEED):
    """Test metrics overall and per mismatch method with 95% bootstrap CIs.

    score: higher = more likely falsified. thr: decision threshold from validation.
    ref_score: optional reference model's score on the same rows; adds a paired bootstrap of AUC(score) - AUC(ref).
    Each group is resampled independently with a fixed seed, so the same resample indices are used for
    every model evaluated with the same seed (paired comparisons stay paired).
    """
    out = {}
    groups = [("all", np.ones(len(df), bool))] + [(m, (df.method == m).values) for m in sorted(df.method.unique())]
    for gi, (name, msk) in enumerate(groups):
        y, s = df.y.values[msk], score[msk]
        pred = (s >= thr).astype(int)
        r = {"n": int(msk.sum()), "roc_auc": float(roc_auc_score(y, s)),
             "accuracy": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred, zero_division=0))}
        rng = np.random.default_rng(seed + gi)
        auc_b, acc_b, f1_b, d_b = [], [], [], []
        rs = ref_score[msk] if ref_score is not None else None
        for _ in range(n_boot):
            b = rng.integers(0, len(y), len(y))
            yb = y[b]
            if yb.min() == yb.max():
                continue
            a = roc_auc_score(yb, s[b])
            auc_b.append(a)
            acc_b.append(accuracy_score(yb, pred[b]))
            f1_b.append(f1_score(yb, pred[b], zero_division=0))
            if rs is not None:
                d_b.append(a - roc_auc_score(yb, rs[b]))
        r.update({"roc_auc_ci95": _ci(auc_b), "accuracy_ci95": _ci(acc_b), "f1_ci95": _ci(f1_b)})
        if rs is not None:
            d = float(r["roc_auc"] - roc_auc_score(y, rs))
            d_b = np.array(d_b)
            r["auc_minus_baseline"] = {"delta": d, "ci95": _ci(d_b),
                                       "p_le_0": float((d_b <= 0).mean())}
        out[name] = r
    return out


def update_results(key, value):
    METRICS.mkdir(parents=True, exist_ok=True)
    path = METRICS / "results.json"
    allres = json.load(open(path)) if path.exists() else {}
    allres[key] = value
    json.dump(allres, open(path, "w"), indent=2)


def plot_roc_by_method(df, scores, title, path, ref_score=None):
    """One panel per group (all + methods); solid = model(s), dotted = similarity baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    groups = ["all"] + sorted(df.method.unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(3.3 * len(groups), 3.4), sharey=True)
    for ax, g in zip(axes, groups):
        msk = np.ones(len(df), bool) if g == "all" else (df.method == g).values
        y = df.y.values[msk]
        for name, s in scores.items():
            fpr, tpr, _ = roc_curve(y, s[msk])
            ax.plot(fpr, tpr, lw=1.6, label=f"{name} {roc_auc_score(y, s[msk]):.3f}")
        if ref_score is not None:
            fpr, tpr, _ = roc_curve(y, ref_score[msk])
            ax.plot(fpr, tpr, lw=1.2, ls="--", c="grey", label=f"baseline {roc_auc_score(y, ref_score[msk]):.3f}")
        ax.plot([0, 1], [0, 1], ls=":", c="lightgrey", lw=1)
        ax.set_title("all methods" if g == "all" else g, fontsize=8)
        ax.set_xlabel("FPR", fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="lower right", title="ROC-AUC", title_fontsize=7)
    axes[0].set_ylabel("TPR", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_auc_bars(evals, title, path):
    """Grouped bars of test ROC-AUC with 95% bootstrap CIs. evals: {model name: evaluate() output}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#9a9a9a", "#2a78b5", "#d1495b", "#e8a33d"]
    groups = list(next(iter(evals.values())).keys())
    x = np.arange(len(groups))
    w = 0.8 / len(evals)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for k, (name, ev) in enumerate(evals.items()):
        a = np.array([ev[g]["roc_auc"] for g in groups])
        err = np.array([[ev[g]["roc_auc"] - ev[g]["roc_auc_ci95"][0], ev[g]["roc_auc_ci95"][1] - ev[g]["roc_auc"]] for g in groups]).T
        xs = x - 0.4 + w * (k + 0.5)
        ax.bar(xs, a, w * 0.95, yerr=err, capsize=2.5, color=colors[k % len(colors)], label=name)
        for xi, ai, e in zip(xs, a, err[1]):
            ax.text(xi, ai + e + 0.01, f"{ai:.2f}", ha="center", fontsize=6.5)
    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.axhline(0.8, ls="--", c="#4a9d5b", lw=0.8, label="target AUC 0.80")
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", "\n", 1) if g != "all" else "all methods" for g in groups], fontsize=7)
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("test ROC-AUC (95% bootstrap CI)", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=len(evals) + 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
