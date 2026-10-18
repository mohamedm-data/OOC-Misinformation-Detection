"""Step 7 (FIRST DRAFT): small PyTorch MLP on the same frozen CLIP pair features as the logistic regression.

2049 -> 512 -> 128 -> 1, ReLU, dropout, AdamW, BCE loss, early stopping on validation ROC-AUC, seed 5500, CPU.
Same test evaluation as step 5 (overall + per method, 95% bootstrap CIs, paired AUC delta vs. cosine baseline
and vs. logistic regression).

Outputs: outputs/models/mlp.pt, results.json key "mlp_draft",
         outputs/figures/fig_roc_models_by_method.png, fig_auc_models_by_method.png, fig_mlp_training.png
"""
import time

import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from common import FIG, MODELS, SEED, evaluate, load_split, pair_features, plot_auc_bars, plot_roc_by_method, \
    update_results, val_threshold

CFG = {"hidden": [512, 128], "dropout": 0.3, "input_dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-2,
       "batch_size": 256, "max_epochs": 100, "patience": 10}


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


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    data = {sp: load_split(sp) for sp in ["train", "val", "test"]}
    X = {sp: pair_features(I, T) for sp, (_, I, T) in data.items()}
    y = {sp: df.y.values for sp, (df, _, _) in data.items()}
    mu, sd = X["train"].mean(0), X["train"].std(0) + 1e-6
    X = {sp: ((x - mu) / sd).astype(np.float32) for sp, x in X.items()}

    model = MLP(X["train"].shape[1], CFG["hidden"], CFG["dropout"], CFG["input_dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(X["train"]), torch.from_numpy(y["train"].astype(np.float32))
    g = torch.Generator().manual_seed(SEED)
    hist, best, best_state, bad = [], -1, None, 0
    for ep in range(CFG["max_epochs"]):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for b in range(0, len(perm), CFG["batch_size"]):
            idx = perm[b:b + CFG["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        tr_auc = roc_auc_score(y["train"], predict(model, X["train"]))
        va_auc = roc_auc_score(y["val"], predict(model, X["val"]))
        hist.append({"epoch": ep + 1, "train_loss": tot / len(perm), "train_auc": float(tr_auc), "val_auc": float(va_auc)})
        print(f"epoch {ep + 1:3d}  loss {tot / len(perm):.4f}  train AUC {tr_auc:.4f}  val AUC {va_auc:.4f}")
        if va_auc > best:
            best, bad = va_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= CFG["patience"]:
                break
    model.load_state_dict(best_state)
    best_epoch = max(hist, key=lambda h: h["val_auc"])["epoch"]
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": CFG, "feature_mean": mu, "feature_std": sd}, MODELS / "mlp.pt")

    vdf, tdf = data["val"][0], data["test"][0]
    p_val, p_test = predict(model, X["val"]), predict(model, X["test"])
    thr = val_threshold(y["val"], p_val)
    base_test = -tdf.cos.values
    base_thr = val_threshold(y["val"], -vdf.cos.values)
    lr = joblib.load(MODELS / "logreg.joblib")
    lr_val = lr.predict_proba(pair_features(*data["val"][1:]))[:, 1]
    lr_test = lr.predict_proba(pair_features(*data["test"][1:]))[:, 1]
    ev = evaluate(tdf, p_test, thr, ref_score=base_test)
    ev_vs_lr = evaluate(tdf, p_test, thr, ref_score=lr_test)
    for gname in ev:
        ev[gname]["auc_minus_logreg"] = ev_vs_lr[gname]["auc_minus_baseline"]
    ev_base = evaluate(tdf, base_test, base_thr)
    ev_lr = evaluate(tdf, lr_test, val_threshold(y["val"], lr_val))

    res = {
        "status": "FIRST DRAFT: architecture and hyperparameters not tuned yet",
        "features": "[img, txt, img*txt, |img-txt|, cosine] (2049-d), standardised with train mean/std",
        "architecture": "2049 -> 512 -> 128 -> 1, ReLU, dropout; BCEWithLogits; AdamW",
        "config": CFG, "seed": SEED, "device": "cpu",
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "epochs_run": len(hist), "best_epoch": best_epoch, "val_roc_auc": float(best),
        "decision_threshold_prob": thr, "history": hist,
        "bootstrap": {"n_resamples": 1000, "seed": SEED, "ci": "percentile 95%"},
        "test": ev, "seconds": time.time() - t0,
    }
    update_results("mlp_draft", res)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.3))
    ep = [h["epoch"] for h in hist]
    ax.plot(ep, [h["train_auc"] for h in hist], label="train AUC")
    ax.plot(ep, [h["val_auc"] for h in hist], label="val AUC")
    ax.axvline(best_epoch, ls=":", c="grey", lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("MLP draft training (early stopping on val AUC)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_mlp_training.png", dpi=150)
    plt.close(fig)
    plot_roc_by_method(tdf, {"LogReg": lr_test, "MLP draft": p_test}, "ROC on test by mismatch method",
                       FIG / "fig_roc_models_by_method.png", ref_score=base_test)
    plot_auc_bars({"Cosine baseline": ev_base, "Logistic regression": ev_lr, "MLP (draft)": ev},
                  "Test ROC-AUC per mismatch method", FIG / "fig_auc_models_by_method.png")

    print(f"best epoch {best_epoch}; val AUC {best:.4f}; threshold {thr:.3f}")
    for gname, r in ev.items():
        d, dl = r["auc_minus_baseline"], r["auc_minus_logreg"]
        print(f"{gname:28s} AUC {r['roc_auc']:.3f} [{r['roc_auc_ci95'][0]:.3f}, {r['roc_auc_ci95'][1]:.3f}]  "
              f"acc {r['accuracy']:.3f}  F1 {r['f1']:.3f}  | vs base {d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
              f"  vs LR {dl['delta']:+.3f} [{dl['ci95'][0]:+.3f}, {dl['ci95'][1]:+.3f}]")
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
