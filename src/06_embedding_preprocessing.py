"""Step 6: why do our CLIP image embeddings differ slightly from the NewsCLIPpings-provided ones?

Re-embeds the VisualNews images from the step-4 experiment (those with a provided image embedding) under
preprocessing / model / device / precision variants and reports cosine agreement with the provided embeddings.

Usage: python src/06_embedding_preprocessing.py   (needs data/visualnews_partial/extracted from step 4)
Outputs: results.json key "embedding_preprocessing", outputs/figures/fig_embedding_variants.png
"""
import json
import pickle
import time
from collections import Counter

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms as tv
from torchvision.transforms import InterpolationMode as IM
from transformers import CLIPModel, CLIPProcessor

from common import FIG, NC, ROOT, update_results

EXTRACT = ROOT / "data" / "visualnews_partial" / "extracted"
MODEL = "openai/clip-vit-base-patch32"
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
BATCH = 64


def mapped_images():
    path2id = {}
    for sp in ["train", "val", "test"]:
        for k, r in json.load(open(NC / "metadata" / f"{sp}.json")).items():
            path2id[r["image_path"].replace("visual_news/", "", 1)] = int(k)
    prov = {}
    for sp in ["train", "val", "test"]:
        prov.update(pickle.load(open(NC / "embeddings/clip_image_embeddings" / f"clip_image_embeddings_{sp}.pkl", "rb")))
    paths = sorted(str(p.relative_to(EXTRACT)) for p in EXTRACT.rglob("*.jpg"))
    pairs = [(p, path2id[p]) for p in paths if p in path2id and path2id[p] in prov]
    P = np.stack([np.asarray(prov[i], np.float32).reshape(-1) for _, i in pairs])
    return [p for p, _ in pairs], P / np.linalg.norm(P, axis=1, keepdims=True)


def openai_order(resample):
    # OpenAI clip: Resize(224) on PIL, CenterCrop, convert RGB *after* resizing, ToTensor, Normalize
    return tv.Compose([tv.Resize(224, interpolation=resample), tv.CenterCrop(224),
                       lambda im: im.convert("RGB"), tv.ToTensor(), tv.Normalize(MEAN, STD)])


def tensor_resize(antialias):
    # resize a uint8 tensor (as a torch/cv2-style pipeline would) instead of a PIL image
    return tv.Compose([lambda im: im.convert("RGB"), tv.PILToTensor(), tv.Resize(224, interpolation=IM.BICUBIC, antialias=antialias),
                       tv.CenterCrop(224), tv.ConvertImageDtype(torch.float32), tv.Normalize(MEAN, STD)])


def main():
    t0 = time.time()
    paths, P = mapped_images()
    print(f"{len(paths)} images with provided embeddings")

    # image properties
    props = []
    for p in paths:
        im = Image.open(EXTRACT / p)
        props.append({"mode": im.mode, "format": im.format, "w": im.size[0], "h": im.size[1],
                      "exif_orientation": int(im.getexif().get(0x0112, 1))})

    hf_proc = CLIPProcessor.from_pretrained(MODEL)
    hf = {"cpu": CLIPModel.from_pretrained(MODEL).eval()}
    if torch.backends.mps.is_available():
        hf["mps"] = CLIPModel.from_pretrained(MODEL).to("mps").eval()
        hf["mps16"] = CLIPModel.from_pretrained(MODEL, dtype=torch.float16).to("mps").eval()
    oa = None
    try:
        import clip
        oa_model, oa_pre = clip.load("ViT-B/32", device="cpu", jit=False)
        oa = {"cpu": oa_model.float().eval(), "pre": oa_pre}
        if torch.backends.mps.is_available():
            oa["mps16"] = clip.load("ViT-B/32", device="cpu", jit=False)[0].half().to("mps").eval()
    except Exception as e:
        print("openai clip package unavailable:", e)

    def hf_pre(ims):
        return hf_proc(images=[im.convert("RGB") for im in ims], return_tensors="pt")["pixel_values"]

    def per_image(tf):
        return lambda ims: torch.stack([tf(im) for im in ims])

    def hf_embed(key):
        m = hf[key]
        dev, dt = ("cpu", torch.float32) if key == "cpu" else ("mps", torch.float16 if key == "mps16" else torch.float32)

        def f(x):
            out = m.get_image_features(pixel_values=x.to(dev, dt))
            return out if torch.is_tensor(out) else out.pooler_output
        return f

    def oa_embed(key):
        m = oa[key]
        dev, dt = ("cpu", torch.float32) if key == "cpu" else ("mps", torch.float16)
        return lambda x: m.encode_image(x.to(dev, dt))

    variants = {
        "hf_processor_mps_fp32 (step-4 setup)": (hf_pre, "hf", "mps"),
        "hf_processor_cpu_fp32": (hf_pre, "hf", "cpu"),
        "hf_processor_mps_fp16": (hf_pre, "hf", "mps16"),
        "pil_bilinear_resize_cpu_fp32": (per_image(openai_order(IM.BILINEAR)), "hf", "cpu"),
        "tensor_bicubic_antialias_cpu_fp32": (per_image(tensor_resize(True)), "hf", "cpu"),
        "tensor_bicubic_no_antialias_cpu_fp32": (per_image(tensor_resize(False)), "hf", "cpu"),
        "exif_transpose_then_hf_processor_cpu_fp32": (lambda ims: hf_pre([ImageOps.exif_transpose(im) for im in ims]), "hf", "cpu"),
    }
    if oa:
        variants["openai_clip_preprocess+hf_model_cpu_fp32"] = (per_image(oa["pre"]), "hf", "cpu")
        variants["openai_clip_package_cpu_fp32"] = (per_image(oa["pre"]), "oa", "cpu")
        variants["openai_clip_package_mps_fp16"] = (per_image(oa["pre"]), "oa", "mps16")

    if "mps" not in hf:
        variants = {k: v for k, v in variants.items() if "mps" not in k}
    emb, res = {}, {}
    for name, (pre, fam, key) in variants.items():
        t1 = time.time()
        enc = hf_embed(key) if fam == "hf" else oa_embed(key)
        out = []
        for b in range(0, len(paths), BATCH):
            ims = [Image.open(EXTRACT / p) for p in paths[b:b + BATCH]]
            for im in ims:
                im.load()
            with torch.no_grad():
                f = enc(pre(ims)).float().cpu()
            out.append(torch.nn.functional.normalize(f, dim=-1).numpy())
        E = np.concatenate(out)
        emb[name] = E
        c = (E * P).sum(1)
        res[name] = {"mean": float(c.mean()), "median": float(np.median(c)), "min": float(c.min()),
                     "p01": float(np.percentile(c, 1)), "frac_gt_0.999": float((c > 0.999).mean()),
                     "frac_gt_0.99": float((c > 0.99).mean()), "seconds": time.time() - t1}
        print(f"{name:45s} mean {c.mean():.5f}  median {np.median(c):.5f}  min {c.min():.4f}  "
              f">0.999 {res[name]['frac_gt_0.999']:.3f}  ({time.time() - t1:.0f}s)")

    # how far apart are our own variants from each other? (if << distance to provided, variants don't explain it)
    ref = "hf_processor_cpu_fp32"
    between = {k: float((emb[k] * emb[ref]).sum(1).mean()) for k in emb if k != ref}
    best = max(res, key=lambda k: res[k]["mean"])
    cb = (emb[best] * P).sum(1)
    worst = np.argsort(cb)[:15]
    sizes = np.array([pr["w"] * pr["h"] for pr in props])
    short = np.array([min(pr["w"], pr["h"]) for pr in props])
    diag = {
        "image_modes": dict(Counter(pr["mode"] for pr in props)),
        "image_formats": dict(Counter(str(pr["format"]) for pr in props)),
        "exif_orientation": dict(Counter(pr["exif_orientation"] for pr in props)),
        "short_side_px": {"min": int(short.min()), "median": float(np.median(short)), "max": int(short.max())},
        "spearman_cos_vs_pixels": float(np.corrcoef(np.argsort(np.argsort(cb)), np.argsort(np.argsort(sizes)))[0, 1]),
        "mean_cos_by_short_side": {f"<{q}": float(cb[short < q].mean()) if (short < q).any() else None for q in [300, 450, 600]}
                                  | {">=600": float(cb[short >= 600].mean()) if (short >= 600).any() else None},
        "worst_images": [{"path": paths[i], "cos": float(cb[i]), **props[i]} for i in worst],
    }
    out = {"n_images": len(paths), "variants": res, "best_variant": best,
           f"mean_cos_to_{ref}": between, "diagnostics": diag, "seconds": time.time() - t0}
    update_results("embedding_preprocessing", out)
    print(json.dumps({"between_variants": between, "diagnostics": {k: v for k, v in diag.items() if k != "worst_images"}}, indent=1))
    for w in diag["worst_images"][:8]:
        print(w)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(res) + 1.2))
    names = sorted(res, key=lambda k: res[k]["mean"])
    data = [(emb[k] * P).sum(1) for k in names]
    ax.boxplot(data, orientation="horizontal", showfliers=True, flierprops={"markersize": 2})
    ax.set_yticks(range(1, len(names) + 1))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("cosine(own embedding, NewsCLIPpings-provided embedding)", fontsize=8)
    ax.set_title(f"Image-embedding agreement by preprocessing variant ({len(paths)} VisualNews images)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_embedding_variants.png", dpi=150)
    plt.close(fig)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
