"""Step 9: (a) nonlinear classical ML baselines on the same CLIP pair features, and (b) better detection of the
adversarial CLIP text-image swaps (semantics_clip_text_image), the weakest mismatch type so far.

Instructor feedback on Progress Report 4: the neural network is nonlinear, so the classical comparison should be
nonlinear too; and every model is weak on the CLIP text-image swaps, where AUC above 0.7 counts as reasonable.

Part A: nonlinear classical models on the 2,049-d features from steps 5 to 8, tuned on the full validation split:
  - Random Forest (full 71k training split)
  - LightGBM gradient boosting (full split, early stopping on validation)
  - RBF-kernel SVM (stratified 15k-pair subsample; kernel SVMs scale quadratically)

Part B: the CLIP text-image swaps. Their fake captions were paired with the image CLIP ranks as the best match
from a large pool, so a high caption-image cosine is the trap, not the evidence. Retrieval-chosen images tend to
be "hubs" (images that look similar to many captions), so the extra features measure how unusual this pair's
similarity is relative to a reference bank of other images and captions:
  - hubness: mean of the top-10 similarities of the image to bank captions, and of the caption to bank images
  - CSLS score (cross-domain similarity local scaling): 2*cos - image hubness - caption hubness
  - rank features: share of bank captions the image matches at least as well as this caption, and vice versa
  - nearest-neighbour similarity inside each modality (image-image, caption-caption)
  - a small contrastive adapter trained on the frozen embeddings (InfoNCE on genuine pairs + pair-level BCE that
    includes the swap negatives), and the same cosine/CSLS/rank features in the adapted space
  Leakage control: training pairs are split into two halves by connected component of shared image ids and
  caption ids (no image or caption appears in both halves). Features for half B use the bank and adapter built
  from half A and vice versa (cross-fitting). Validation and test use the average of the two. So no pair is ever
  scored against a bank that contains its own image, its own caption, or an adapter that saw it.
  Models on the 2,049 + 15 features: logistic regression, LightGBM, MLP (step 8 setting), MLP with swap pairs
  up-weighted during training, and a validation-weighted LightGBM + MLP ensemble. The swap weight and the
  ensemble weight are chosen on validation only. A swap-only LightGBM is reported as a diagnostic (it needs the
  method label at prediction time, which a real deployment does not have).

Primary test set: the same curated 2,400 pairs as every earlier step; per-method results with 95% bootstrap CIs
and paired deltas against the step 8 MLP. Outputs: results.json key "nonlinear_and_swaps",
outputs/models/{lgbm_swapaware.txt, mlp_swapaware.pt, swap_adapters.pt, swap_feature_banks.npz},
outputs/figures/{fig_auc_classical_nonlinear.png, fig_auc_swapaware.png, fig_swap_feature_ablation.png}
"""
import importlib
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import MODELS, FIG, N_BOOT, SEED, evaluate, plot_auc_bars, update_results, val_threshold

s8 = importlib.import_module("08_full_split")
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated")

SWAP = "semantics_clip_text_image"
K = 10                       # neighbours for hubness
BANK_MAX = 25000             # bank items per modality per half
SVM_N = 15000                # SVM training subsample
SVM_C = [0.3, 1.0, 3.0]
RF_LEAF = [2, 10]
LGB_LEAVES = [31, 63]
SWAP_WEIGHTS = [1.0, 2.0, 3.0]
ENS_W = [0.0, 0.25, 0.5, 0.75, 1.0]
LR_C = [0.0003, 0.001, 0.003]
MLP_CFG = {"hidden": [256], "dropout": 0.5, "input_dropout": 0.2, "weight_decay": 5e-2}   # step 8 selection
FAST = os.environ.get("STEP9_FAST") == "1"   # smoke test only: tiny grids and epochs


# ---------------------------------------------------------------- leakage-safe halves of the training split
def component_halves(df, seed=SEED):
    """Two halves of the training rows such that no image id or caption id appears in both."""
    n = len(df)
    img_codes = pd.factorize(df.image_id)[0]
    cap_codes = pd.factorize(df.id)[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for codes in (img_codes, cap_codes):
        first = {}
        for r, c in enumerate(codes):
            if c in first:
                ra, rb = find(r), find(first[c])
                if ra != rb:
                    parent[ra] = rb
            else:
                first[c] = r
    roots = np.array([find(r) for r in range(n)])
    uniq = np.unique(roots)
    rng = np.random.default_rng(seed)
    side = dict(zip(uniq, rng.integers(0, 2, len(uniq))))
    half = np.array([side[r] for r in roots])
    return half, len(uniq)


def build_bank(df, I, T, rows, seed):
    """Unique images and captions of the given rows, capped at BANK_MAX each."""
    rng = np.random.default_rng(seed)
    sub = df.iloc[rows]
    i_rows = rows[~sub.image_id.duplicated().values]
    t_rows = rows[~sub.id.duplicated().values]
    if len(i_rows) > BANK_MAX:
        i_rows = np.sort(rng.choice(i_rows, BANK_MAX, replace=False))
    if len(t_rows) > BANK_MAX:
        t_rows = np.sort(rng.choice(t_rows, BANK_MAX, replace=False))
    return {"img_rows": i_rows, "txt_rows": t_rows, "BI": I[i_rows], "BT": T[t_rows]}


# ---------------------------------------------------------------- hubness / CSLS / rank features
HUB_NAMES = ["csls", "img_hub", "txt_hub", "img_top1", "txt_top1", "img_rank", "txt_rank",
             "ii_top1", "tt_top1", "ii_hub", "tt_hub"]


def _topk_mean(S, k):
    part = np.partition(S, S.shape[1] - k, axis=1)[:, -k:]
    return part.mean(1), part.max(1)


def hub_features(QI, QT, BI, BT, k=K, chunk=2048):
    """Features of each (image, caption) query pair relative to a bank of other images (BI) and captions (BT).
    Bank entries identical to the query item (cosine > 0.9999) are ignored, as a guard against duplicates."""
    out = np.zeros((len(QI), len(HUB_NAMES)), np.float32)
    for s in range(0, len(QI), chunk):
        qi, qt = QI[s:s + chunk], QT[s:s + chunk]
        cos = (qi * qt).sum(1)
        res = {}
        for name, Q, B in [("ic", qi, BT), ("ti", qt, BI), ("ii", qi, BI), ("tt", qt, BT)]:
            S = Q @ B.T
            dup = S > 0.9999
            S[dup] = -2.0
            hub, top1 = _topk_mean(S, k)
            res[name] = (hub, top1, S, dup)
        img_hub, img_top1, S_ic, d_ic = res["ic"]
        txt_hub, txt_top1, S_ti, d_ti = res["ti"]
        img_rank = ((S_ic >= cos[:, None]).sum(1)) / np.maximum((~d_ic).sum(1), 1)
        txt_rank = ((S_ti >= cos[:, None]).sum(1)) / np.maximum((~d_ti).sum(1), 1)
        out[s:s + chunk] = np.stack([2 * cos - img_hub - txt_hub, img_hub, txt_hub, img_top1, txt_top1,
                                     img_rank, txt_rank, res["ii"][1], res["tt"][1], res["ii"][0], res["tt"][0]], 1)
    return out


# ---------------------------------------------------------------- contrastive adapter on frozen embeddings
def train_adapter(I, T, y, epochs=8, seed=SEED):
    import torch
    from torch import nn

    class Head(nn.Module):
        def __init__(self, d=512, h=512):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, d))
            nn.init.zeros_(self.f[-1].weight)
            nn.init.zeros_(self.f[-1].bias)

        def forward(self, x):
            return nn.functional.normalize(x + self.f(x), dim=-1)

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.img, self.txt = Head(), Head()
            self.log_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
            self.bias = nn.Parameter(torch.tensor(0.0))

    torch.manual_seed(seed)
    model = Adapter()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    It, Tt, yt = torch.from_numpy(np.ascontiguousarray(I)), torch.from_numpy(np.ascontiguousarray(T)), torch.from_numpy(y.astype(np.float32))
    gen = torch.Generator().manual_seed(seed)
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    bs = 1024
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(It), generator=gen)
        tot = 0.0
        for b in range(0, len(perm), bs):
            idx = perm[b:b + bs]
            zi, zt = model.img(It[idx]), model.txt(Tt[idx])
            scale = model.log_scale.exp().clamp(max=100)
            cos = (zi * zt).sum(-1)
            # pair-level loss: genuine (y=0) high similarity, falsified (y=1, includes swap negatives) low
            loss = bce(-(scale * cos + model.bias), yt[idx])
            g = yt[idx] == 0
            if g.sum() > 1:     # InfoNCE over genuine pairs in the batch, both directions
                logits = scale * zi[g] @ zt[g].T
                lab = torch.arange(int(g.sum()))
                loss = loss + 0.5 * (ce(logits, lab) + ce(logits.T, lab))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        print(f"    adapter epoch {ep + 1}: loss {tot / len(It):.4f}", flush=True)
    model.eval()
    return model


def adapt(model, X, which):
    import torch
    with torch.no_grad():
        head = model.img if which == "img" else model.txt
        X = np.ascontiguousarray(X, dtype=np.float32)
        return np.concatenate([head(torch.from_numpy(X[s:s + 8192])).numpy() for s in range(0, len(X), 8192)])


ADAPT_NAMES = ["a_cos", "a_csls", "a_img_rank", "a_txt_rank"]


def swap_features(I, T, bank, adapter):
    """15 extra features of each pair against one bank (+ the adapter trained on the same half)."""
    h = hub_features(I, T, bank["BI"], bank["BT"])
    if adapter is None:
        a = np.zeros((len(I), len(ADAPT_NAMES)), np.float32)
    else:
        aI, aT = adapt(adapter, I, "img"), adapt(adapter, T, "txt")
        aBI, aBT = adapt(adapter, bank["BI"], "img"), adapt(adapter, bank["BT"], "txt")
        ah = hub_features(aI, aT, aBI, aBT)
        a = np.stack([(aI * aT).sum(1), ah[:, 0], ah[:, 5], ah[:, 6]], 1)
    return np.hstack([h, a]).astype(np.float32)


# ---------------------------------------------------------------- models
def fit_lgbm(X, y, Xv, yv, leaves, w=None):
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(n_estimators=200 if FAST else 4000, learning_rate=0.05, num_leaves=leaves,
                             min_child_samples=50, colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
                             reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
    clf.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)])
    return clf


def fit_mlp_weighted(cfg, X, y, Xv, yv, w=None, seed=SEED):
    """Step 8 MLP training loop with optional per-sample weights."""
    import torch
    from torch import nn
    common = dict(s8.MLP_COMMON)
    if FAST:
        common.update(max_epochs=3, patience=2)
    torch.manual_seed(seed)
    model = s8.MLP(X.shape[1], cfg["hidden"], cfg["dropout"], cfg["input_dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=common["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    wt = torch.ones(len(y)) if w is None else torch.from_numpy((w / w.mean()).astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    hist, best, best_state, bad = [], -1, None, 0
    for ep in range(common["max_epochs"]):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for b in range(0, len(perm), common["batch_size"]):
            idx = perm[b:b + common["batch_size"]]
            opt.zero_grad()
            (loss_fn(model(Xt[idx]), yt[idx]) * wt[idx]).mean().backward()
            opt.step()
        va = roc_auc_score(yv, s8.predict(model, Xv))
        hist.append({"epoch": ep + 1, "val_auc": float(va)})
        if va > best:
            best, bad, best_state = va, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= common["patience"]:
                break
    model.load_state_dict(best_state)
    print(f"    MLP best val AUC {best:.4f} at epoch {max(hist, key=lambda h: h['val_auc'])['epoch']}", flush=True)
    return model, float(best), hist


def paired_delta(df, score, ref, n_boot=N_BOOT, seed=SEED):
    """AUC(score) - AUC(ref) with the same per-group resample indices as common.evaluate (stays paired with it),
    without recomputing accuracy/F1 (much faster)."""
    out = {}
    groups = [("all", np.ones(len(df), bool))] + [(m, (df.method == m).values) for m in sorted(df.method.unique())]
    for gi, (name, msk) in enumerate(groups):
        y, s, r = df.y.values[msk], score[msk], ref[msk]
        rng = np.random.default_rng(seed + gi)
        d_b = []
        for _ in range(n_boot):
            b = rng.integers(0, len(y), len(y))
            if y[b].min() == y[b].max():
                continue
            d_b.append(roc_auc_score(y[b], s[b]) - roc_auc_score(y[b], r[b]))
        d_b = np.array(d_b)
        out[name] = {"delta": float(roc_auc_score(y, s) - roc_auc_score(y, r)),
                     "ci95": [float(np.percentile(d_b, 2.5)), float(np.percentile(d_b, 97.5))],
                     "p_le_0": float((d_b <= 0).mean())}
    return out


def swap_auc(df, p):
    m = (df.method == SWAP).values
    return float(roc_auc_score(df.y.values[m], p[m]))


def main():
    t0 = time.time()
    import torch
    torch.set_num_threads(max(1, torch.get_num_threads()))
    full = {sp: s8.load_full(sp) for sp in ["train", "val", "test"]}
    (trdf, Xtr), (vadf, Xva), (tedf, Xte) = full["train"], full["val"], full["test"]
    ytr, yva = trdf.y.values, vadf.y.values
    te_cur_idx, cur_test = s8.curated_rows(tedf, "test")
    ctdf = tedf.iloc[te_cur_idx].reset_index(drop=True)
    print(f"train {len(trdf)}, val {len(vadf)}, test {len(tedf)}; curated test {len(ctdf)}", flush=True)
    base_cur = -ctdf.cos.values
    base_ev = evaluate(ctdf, base_cur, val_threshold(yva, -vadf.cos.values))

    # reference models from step 8 (saved), so every delta is against the exact reported 0.791 model
    lr_full = joblib.load(MODELS / "logreg_full.joblib")
    p_lr_val, p_lr_cur = lr_full.predict_proba(Xva)[:, 1], lr_full.predict_proba(Xte[te_cur_idx])[:, 1]
    ck = torch.load(MODELS / "mlp_full.pt", weights_only=False)
    mlp_ref = s8.MLP(Xtr.shape[1], ck["config"]["hidden"], ck["config"]["dropout"], ck["config"]["input_dropout"])
    mlp_ref.load_state_dict(ck["state_dict"])
    Zr = lambda X: ((X - ck["feature_mean"]) / ck["feature_std"]).astype(np.float32)
    p_ref_val, p_ref_cur = s8.predict(mlp_ref, Zr(Xva)), s8.predict(mlp_ref, Zr(Xte[te_cur_idx]))
    ref_val_auc = float(roc_auc_score(yva, p_ref_val))
    print(f"step 8 MLP reloaded: val AUC {ref_val_auc:.4f}, curated test AUC {roc_auc_score(ctdf.y, p_ref_cur):.4f}")

    def report(name, p_val, p_cur, extra=None):
        thr = val_threshold(yva, p_val)
        ev = evaluate(ctdf, p_cur, thr, ref_score=base_cur)
        d_ref, d_lr = paired_delta(ctdf, p_cur, p_ref_cur), paired_delta(ctdf, p_cur, p_lr_cur)
        for g in ev:
            ev[g]["auc_minus_mlp_step8"] = d_ref[g]
            ev[g]["auc_minus_logreg_full"] = d_lr[g]
        r = {"val_roc_auc": float(roc_auc_score(yva, p_val)), "val_swap_auc": swap_auc(vadf, p_val),
             "decision_threshold": thr, "test": ev, **(extra or {})}
        a, sw = ev["all"], ev[SWAP]
        print(f"  {name:28s} test AUC {a['roc_auc']:.3f} [{a['roc_auc_ci95'][0]:.3f}, {a['roc_auc_ci95'][1]:.3f}]"
              f"  swaps {sw['roc_auc']:.3f} [{sw['roc_auc_ci95'][0]:.3f}, {sw['roc_auc_ci95'][1]:.3f}]"
              f"  vs step-8 MLP {a['auc_minus_mlp_step8']['delta']:+.3f}", flush=True)
        return r

    res = {"reference": {"logreg_full": report("LogReg full (step 8)", p_lr_val, p_lr_cur),
                         "mlp_full": report("MLP full (step 8)", p_ref_val, p_ref_cur)},
           "baseline_same_bootstrap": base_ev}

    # ============================ Part A: nonlinear classical models on the same 2,049 features
    print("\nPart A: nonlinear classical models", flush=True)
    classical, cls_scores = {}, {}
    # Random Forest
    rf_sweep, best = {}, None
    for leaf in RF_LEAF:
        t = time.time()
        rf = RandomForestClassifier(n_estimators=50 if FAST else 400, max_features="sqrt", min_samples_leaf=leaf,
                                    n_jobs=-1, random_state=SEED)
        rf.fit(Xtr, ytr)
        pv = rf.predict_proba(Xva)[:, 1]
        rf_sweep[leaf] = {"val_auc": float(roc_auc_score(yva, pv)), "seconds": time.time() - t}
        print(f"    RF min_samples_leaf={leaf}: val AUC {rf_sweep[leaf]['val_auc']:.4f} ({time.time() - t:.0f}s)", flush=True)
        if best is None or rf_sweep[leaf]["val_auc"] > best[0]:
            best = (rf_sweep[leaf]["val_auc"], leaf, pv, rf.predict_proba(Xte[te_cur_idx])[:, 1])
        del rf
    classical["random_forest"] = report("Random Forest", best[2], best[3],
                                        {"n_estimators": 50 if FAST else 400, "max_features": "sqrt",
                                         "selected_min_samples_leaf": best[1], "sweep": {str(k): v for k, v in rf_sweep.items()}})
    cls_scores["random_forest"] = (best[2], best[3])

    # LightGBM
    lgb_sweep, best = {}, None
    for leaves in LGB_LEAVES:
        t = time.time()
        clf = fit_lgbm(Xtr, ytr, Xva, yva, leaves)
        pv = clf.predict_proba(Xva)[:, 1]
        lgb_sweep[leaves] = {"val_auc": float(roc_auc_score(yva, pv)), "best_iteration": int(clf.best_iteration_ or 0),
                             "seconds": time.time() - t}
        print(f"    LightGBM num_leaves={leaves}: val AUC {lgb_sweep[leaves]['val_auc']:.4f} "
              f"iters {lgb_sweep[leaves]['best_iteration']} ({time.time() - t:.0f}s)", flush=True)
        if best is None or lgb_sweep[leaves]["val_auc"] > best[0]:
            best = (lgb_sweep[leaves]["val_auc"], leaves, pv, clf.predict_proba(Xte[te_cur_idx])[:, 1])
    classical["lightgbm"] = report("LightGBM", best[2], best[3],
                                   {"learning_rate": 0.05, "colsample_bytree": 0.3, "subsample": 0.8,
                                    "selected_num_leaves": best[1], "sweep": {str(k): v for k, v in lgb_sweep.items()}})
    cls_scores["lightgbm"] = (best[2], best[3])

    # RBF SVM on a stratified subsample
    svm_rows = s8.stratified_fraction(trdf, min(1.0, SVM_N / len(trdf)))
    sc = StandardScaler().fit(Xtr[svm_rows])
    Ss, Sv, St = sc.transform(Xtr[svm_rows]), sc.transform(Xva), sc.transform(Xte[te_cur_idx])
    svm_sweep, best = {}, None
    for C in SVM_C:
        t = time.time()
        svm = SVC(kernel="rbf", C=C, gamma="scale", cache_size=2000, random_state=SEED)
        svm.fit(Ss, ytr[svm_rows])
        pv = svm.decision_function(Sv)
        svm_sweep[C] = {"val_auc": float(roc_auc_score(yva, pv)), "n_support": int(svm.n_support_.sum()),
                        "seconds": time.time() - t}
        print(f"    SVM C={C}: val AUC {svm_sweep[C]['val_auc']:.4f} ({time.time() - t:.0f}s)", flush=True)
        if best is None or svm_sweep[C]["val_auc"] > best[0]:
            best = (svm_sweep[C]["val_auc"], C, pv, svm.decision_function(St))
    classical["svm_rbf"] = report("RBF SVM (15k subsample)", best[2], best[3],
                                  {"n_train": int(len(svm_rows)), "gamma": "scale", "selected_C": best[1],
                                   "score": "decision function (not a probability)",
                                   "sweep": {str(k): v for k, v in svm_sweep.items()}})
    cls_scores["svm_rbf"] = (best[2], best[3])
    res["classical_nonlinear"] = classical

    # ============================ Part B: swap-aware features
    print("\nPart B: swap-aware features (cross-fitted banks + adapters)", flush=True)
    Itr, Ttr = Xtr[:, :512], Xtr[:, 512:1024]
    half, n_comp = component_halves(trdf)
    banks, adapters = [], []
    for h in (0, 1):
        rows = np.where(half == h)[0]
        banks.append(build_bank(trdf, Itr, Ttr, rows, SEED + h))
        print(f"  half {h}: {len(rows)} pairs, bank {len(banks[h]['img_rows'])} images / "
              f"{len(banks[h]['txt_rows'])} captions; training adapter", flush=True)
        adapters.append(train_adapter(Itr[rows], Ttr[rows], ytr[rows], epochs=2 if FAST else 8, seed=SEED + h))
    Ftr = np.zeros((len(trdf), len(HUB_NAMES) + len(ADAPT_NAMES)), np.float32)
    for h in (0, 1):
        rows = np.where(half == h)[0]
        Ftr[rows] = swap_features(Itr[rows], Ttr[rows], banks[1 - h], adapters[1 - h])

    def feats_eval(X):
        I, T = X[:, :512], X[:, 512:1024]
        return 0.5 * (swap_features(I, T, banks[0], adapters[0]) + swap_features(I, T, banks[1], adapters[1]))

    Fva, Fte = feats_eval(Xva), feats_eval(Xte)
    fnames = HUB_NAMES + ADAPT_NAMES
    # single-feature AUCs (direction-free: max(AUC, 1-AUC)) per method on validation, for the write-up
    single = {}
    for j, nm in enumerate(fnames + ["cos"]):
        f = Fva[:, j] if nm != "cos" else vadf.cos.values
        single[nm] = {g: float(roc_auc_score(vadf.y.values[m], f[m]))
                      for g, m in [("all", np.ones(len(vadf), bool))] + [(mm, (vadf.method == mm).values) for mm in sorted(vadf.method.unique())]}
    torch.save({"adapters": [a.state_dict() for a in adapters]}, MODELS / "swap_adapters.pt")
    np.savez_compressed(MODELS / "swap_feature_banks.npz", half=half,
                        img_rows_0=banks[0]["img_rows"], txt_rows_0=banks[0]["txt_rows"],
                        img_rows_1=banks[1]["img_rows"], txt_rows_1=banks[1]["txt_rows"])

    XtrE, XvaE, XteE = np.hstack([Xtr, Ftr]), np.hstack([Xva, Fva]), np.hstack([Xte, Fte])
    XcurE = XteE[te_cur_idx]
    swap_tr = (trdf.method == SWAP).values
    enh, enh_scores = {}, {}

    # logistic regression on enhanced features (does a linear model gain from them too?)
    clf, C, sweep = s8.fit_logreg(XtrE, ytr, XvaE, yva, grid=LR_C)
    enh["logreg_swapaware"] = report("LogReg + swap features", clf.predict_proba(XvaE)[:, 1],
                                     clf.predict_proba(XcurE)[:, 1], {"best_C": C, "c_sweep": sweep})

    # LightGBM on enhanced features
    lgb_sweep, best = {}, None
    for leaves in LGB_LEAVES:
        m = fit_lgbm(XtrE, ytr, XvaE, yva, leaves)
        pv = m.predict_proba(XvaE)[:, 1]
        lgb_sweep[leaves] = {"val_auc": float(roc_auc_score(yva, pv)), "val_swap_auc": swap_auc(vadf, pv),
                             "best_iteration": int(m.best_iteration_ or 0)}
        if best is None or lgb_sweep[leaves]["val_auc"] > best[0]:
            best = (lgb_sweep[leaves]["val_auc"], leaves, pv, m.predict_proba(XcurE)[:, 1], m)
    best[4].booster_.save_model(str(MODELS / "lgbm_swapaware.txt"))
    imp = best[4].booster_.feature_importance("gain")
    imp_new = {nm: float(imp[2049 + j] / imp.sum()) for j, nm in enumerate(fnames)}
    enh["lightgbm_swapaware"] = report("LightGBM + swap features", best[2], best[3],
                                       {"selected_num_leaves": best[1], "sweep": {str(k): v for k, v in lgb_sweep.items()},
                                        "gain_share_new_features": imp_new,
                                        "gain_share_new_features_total": float(sum(imp_new.values()))})
    enh_scores["lightgbm"] = (best[2], best[3])

    # MLP on enhanced features, swap pairs up-weighted; weight chosen on validation (overall AUC, ties -> swap AUC)
    mu, sd = XtrE.mean(0), XtrE.std(0) + 1e-6
    Z = lambda X: ((X - mu) / sd).astype(np.float32)
    ZtrE, ZvaE, ZcurE = Z(XtrE), Z(XvaE), Z(XcurE)
    mlp_runs, best = {}, None
    for wgt in SWAP_WEIGHTS:
        print(f"  MLP + swap features, swap weight {wgt}", flush=True)
        w = np.where(swap_tr, wgt, 1.0)
        model, va, hist = fit_mlp_weighted(MLP_CFG, ZtrE, ytr, ZvaE, yva, w=w)
        pv = s8.predict(model, ZvaE)
        mlp_runs[str(wgt)] = {"val_auc": va, "val_swap_auc": swap_auc(vadf, pv),
                              "best_epoch": max(hist, key=lambda h: h["val_auc"])["epoch"], "epochs_run": len(hist)}
        enh[f"mlp_swapaware_w{wgt:g}"] = report(f"MLP + swap feats, w={wgt:g}", pv, s8.predict(model, ZcurE),
                                                {"swap_weight": wgt, **mlp_runs[str(wgt)]})
        key = (round(va, 3), mlp_runs[str(wgt)]["val_swap_auc"])
        if best is None or key > best[0]:
            best = (key, wgt, pv, s8.predict(model, ZcurE), model)
    sel_w = best[1]
    torch.save({"state_dict": best[4].state_dict(), "config": {**MLP_CFG, **s8.MLP_COMMON}, "swap_weight": sel_w,
                "feature_mean": mu, "feature_std": sd, "extra_features": fnames}, MODELS / "mlp_swapaware.pt")
    enh_scores["mlp"] = (best[2], best[3])

    # ensemble: w * MLP + (1 - w) * LightGBM, w chosen on validation AUC
    ens = {}
    for w in ENS_W:
        pv = w * enh_scores["mlp"][0] + (1 - w) * enh_scores["lightgbm"][0]
        ens[w] = float(roc_auc_score(yva, pv))
    w_best = max(ens, key=ens.get)
    p_ens_val = w_best * enh_scores["mlp"][0] + (1 - w_best) * enh_scores["lightgbm"][0]
    p_ens_cur = w_best * enh_scores["mlp"][1] + (1 - w_best) * enh_scores["lightgbm"][1]
    enh["ensemble_swapaware"] = report("Ensemble (MLP + LightGBM)", p_ens_val, p_ens_cur,
                                       {"mlp_weight": w_best, "val_auc_by_weight": {str(k): v for k, v in ens.items()},
                                        "mlp_swap_weight": sel_w})

    # diagnostic: swap-only LightGBM (needs the method label at test time; not deployable)
    sv, st = (vadf.method == SWAP).values, (ctdf.method == SWAP).values
    m = fit_lgbm(XtrE[swap_tr], ytr[swap_tr], XvaE[sv], yva[sv], 31)
    spec_auc = float(roc_auc_score(ctdf.y.values[st], m.predict_proba(XcurE[st])[:, 1]))
    m0 = fit_lgbm(Xtr[swap_tr], ytr[swap_tr], Xva[sv], yva[sv], 31)
    spec0_auc = float(roc_auc_score(ctdf.y.values[st], m0.predict_proba(Xte[te_cur_idx][st])[:, 1]))
    res["swap_specialist_diagnostic"] = {
        "note": "LightGBM trained and tested only on CLIP text-image swap pairs; needs the method label, so not deployable",
        "test_swap_auc_with_swap_features": spec_auc, "test_swap_auc_without": spec0_auc}
    print(f"  diagnostic swap-only LightGBM: swaps AUC {spec_auc:.3f} (without new features {spec0_auc:.3f})")

    # choose the final model on validation AUC among the swap-aware models
    cands = {k: v for k, v in enh.items() if k != "logreg_swapaware"}
    final = max(cands, key=lambda k: cands[k]["val_roc_auc"])
    res["swap_aware"] = enh
    res["swap_features"] = {"names": fnames, "k": K, "bank_max_per_modality": BANK_MAX,
                            "train_components": int(n_comp), "half_sizes": [int((half == 0).sum()), int((half == 1).sum())],
                            "val_single_feature_auc": single}
    res["final_model"] = {"name": final, "selected_on": "validation ROC-AUC"}
    res["seconds"] = time.time() - t0
    update_results("nonlinear_and_swaps", res)

    # ---------------------------------------------------------------- figures
    ref = res["reference"]
    plot_auc_bars({"LogReg (linear)": ref["logreg_full"]["test"], "Random Forest": classical["random_forest"]["test"],
                   "LightGBM": classical["lightgbm"]["test"], "RBF SVM": classical["svm_rbf"]["test"],
                   "MLP (step 8)": ref["mlp_full"]["test"]},
                  "Test ROC-AUC: linear vs. nonlinear classical models vs. MLP (same CLIP features)",
                  FIG / "fig_auc_classical_nonlinear.png")
    plot_auc_bars({"Cosine baseline": base_ev, "MLP (step 8)": ref["mlp_full"]["test"],
                   "LightGBM + swap feats": enh["lightgbm_swapaware"]["test"],
                   f"Final: {final.replace('_swapaware', '')}": enh[final]["test"]},
                  "Test ROC-AUC per mismatch method with swap-aware features", FIG / "fig_auc_swapaware.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["cos"] + fnames
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    meths = sorted(vadf.method.unique())
    x = np.arange(len(names))
    for i, mm in enumerate(meths):
        ax.bar(x - 0.4 + 0.2 * (i + 0.5), [max(single[n][mm], 1 - single[n][mm]) for n in names], 0.19, label=mm)
    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.axhline(0.7, ls="--", c="#4a9d5b", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("single-feature val AUC (either direction)", fontsize=8)
    ax.set_title("How well each swap-aware feature separates genuine from falsified, by mismatch method", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig_swap_feature_ablation.png", dpi=150)
    plt.close(fig)
    print(f"\nfinal model (validation choice): {final}; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
