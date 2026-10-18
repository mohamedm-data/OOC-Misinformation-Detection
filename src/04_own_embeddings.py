"""Step 4: compute our own CLIP ViT-B/32 embeddings for VisualNews images from a partial origin.tar
and compare them to the NewsCLIPpings-provided embeddings.

Usage: python -I src/04_own_embeddings.py data/visualnews_partial/origin_head.tar
"""
import json
import pickle
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
NC = ROOT / "data" / "news_clippings"
OUT = ROOT / "outputs"
METRICS = OUT / "metrics"
EMB_OUT = OUT / "embeddings"
EXTRACT = ROOT / "data" / "visualnews_partial" / "extracted"
MODEL = "openai/clip-vit-base-patch32"
BATCH = 64


def extract_partial(tar_path):
    """Stream-extract a truncated tar; stop cleanly at the truncation point."""
    EXTRACT.mkdir(parents=True, exist_ok=True)
    members, err = [], None
    try:
        with tarfile.open(tar_path, mode="r|") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                try:
                    tf.extract(m, EXTRACT, filter="data")
                    members.append(m.name)
                except Exception as e:  # truncated last member
                    err = f"{type(e).__name__}: {e}"
                    break
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    # drop a possibly truncated final file
    if err and members:
        last = EXTRACT / members[-1]
        try:
            Image.open(last).verify()
        except Exception:
            members.pop()
    return members, err


def main(tar_path):
    t0 = time.time()
    members, err = extract_partial(tar_path)
    t_extract = time.time() - t0
    imgs = [m for m in members if m.lower().endswith((".jpg", ".jpeg", ".png"))]
    sources = sorted({m.split("/")[1] for m in members if m.count("/") > 1})
    other = [m for m in members if m not in imgs][:20]
    print(f"extracted {len(members)} files ({len(imgs)} images) in {t_extract:.1f}s; stop: {err}")

    # map tar path -> NewsCLIPpings id via metadata image_path ("visual_news/origin/<src>/images/..")
    path2id, meta_all = {}, {}
    for sp in ["train", "val", "test"]:
        meta = json.load(open(NC / "metadata" / f"{sp}.json"))
        for k, r in meta.items():
            p = r["image_path"].replace("visual_news/", "", 1)
            path2id[p] = int(k)
            meta_all[int(k)] = (sp, r["caption"])
        del meta
    img_emb, txt_emb = {}, {}
    for sp in ["train", "val", "test"]:
        img_emb.update(pickle.load(open(NC / "embeddings/clip_image_embeddings" / f"clip_image_embeddings_{sp}.pkl", "rb")))
        txt_emb.update(pickle.load(open(NC / "embeddings/clip_text_embeddings" / f"clip_text_embeddings_{sp}.pkl", "rb")))
    mapped = [(m, path2id[m]) for m in imgs if m in path2id]
    with_img_emb = [(m, i) for m, i in mapped if i in img_emb]
    print(f"{len(mapped)} images map to NewsCLIPpings metadata; {len(with_img_emb)} have provided image embeddings")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)

    # image embeddings for ALL extracted images (throughput), then compare on mapped subset
    own_img, bad = {}, []
    load_t, model_t = 0.0, 0.0
    # warm-up
    with torch.no_grad():
        model.get_image_features(**proc(images=[Image.new("RGB", (224, 224))], return_tensors="pt").to(device))
    for b in range(0, len(imgs), BATCH):
        names = imgs[b:b + BATCH]
        t1 = time.time()
        pil, ok = [], []
        for n in names:
            try:
                pil.append(Image.open(EXTRACT / n).convert("RGB"))
                ok.append(n)
            except Exception:
                bad.append(n)
        inp = proc(images=pil, return_tensors="pt").to(device)
        t2 = time.time()
        with torch.no_grad():
            f = model.get_image_features(**inp)
            if not torch.is_tensor(f):
                f = f.pooler_output
            f = torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy()
        if device == "mps":
            torch.mps.synchronize()
        t3 = time.time()
        load_t += t2 - t1
        model_t += t3 - t2
        own_img.update(dict(zip(ok, f)))
    n_img = len(own_img)

    def agree(pairs):
        c = np.array([float(np.dot(a, b / np.linalg.norm(b))) for a, b in pairs])
        if len(c) == 0:
            return None
        return {"n": int(len(c)), "mean": float(c.mean()), "min": float(c.min()), "p01": float(np.percentile(c, 1)),
                "median": float(np.median(c)), "frac_gt_0.99": float((c > 0.99).mean()), "frac_gt_0.95": float((c > 0.95).mean())}

    img_pairs = [(own_img[m], np.asarray(img_emb[i], dtype=np.float32)) for m, i in with_img_emb if m in own_img]
    img_agree = agree(img_pairs)

    # text: captions of mapped images (caption id == image id for the genuine VisualNews record)
    cap_ids = [i for _, i in mapped if i in txt_emb]
    caps = [meta_all[i][1] for i in cap_ids]
    own_txt, t_txt = [], time.time()
    for b in range(0, len(caps), 256):
        inp = proc(text=caps[b:b + 256], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            f = model.get_text_features(**inp)
            if not torch.is_tensor(f):
                f = f.pooler_output
            own_txt.append(torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy())
    if device == "mps":
        torch.mps.synchronize()
    t_txt = time.time() - t_txt
    own_txt = np.concatenate(own_txt) if own_txt else np.zeros((0, 512))
    txt_agree = agree([(own_txt[k], np.asarray(txt_emb[i], dtype=np.float32)) for k, i in enumerate(cap_ids)])

    # does own image-caption cosine track the provided one?
    pair_corr = None
    both = [(m, i) for m, i in with_img_emb if i in txt_emb and m in own_img]
    if both:
        idx = {i: k for k, i in enumerate(cap_ids)}
        own_c = np.array([np.dot(own_img[m], own_txt[idx[i]]) for m, i in both])
        prov_c = np.array([np.dot(img_emb[i], txt_emb[i]) / np.linalg.norm(img_emb[i]) / np.linalg.norm(txt_emb[i]) for m, i in both])
        pair_corr = {"n": len(both), "pearson": float(np.corrcoef(own_c, prov_c)[0, 1]),
                     "mean_abs_diff": float(np.abs(own_c - prov_c).mean())}

    res = {
        "tar_bytes": Path(tar_path).stat().st_size, "extract_seconds": t_extract, "extract_stop_reason": err,
        "files_extracted": len(members), "images_extracted": len(imgs), "sources_seen": sources,
        "non_image_files_sample": other, "images_unreadable": len(bad),
        "images_mapped_to_metadata": len(mapped), "images_with_provided_image_embedding": len(with_img_emb),
        "mapped_by_split": {sp: sum(meta_all[i][0] == sp for _, i in mapped) for sp in ["train", "val", "test"]},
        "image_embedding_agreement_cosine": img_agree, "text_embedding_agreement_cosine": txt_agree,
        "pair_cosine_own_vs_provided": pair_corr,
        "throughput": {"device": device, "batch": BATCH, "images": n_img,
                       "img_per_sec_model_only": n_img / model_t if model_t else None,
                       "img_per_sec_end_to_end": n_img / (model_t + load_t) if model_t else None,
                       "captions": len(caps), "captions_per_sec": len(caps) / t_txt if t_txt else None},
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__, "model": MODEL},
    }
    EMB_OUT.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    np.save(EMB_OUT / "own_image_embeddings.npy", np.stack([own_img[m] for m in sorted(own_img)]) if own_img else np.zeros((0, 512)))
    json.dump(sorted(own_img), open(EMB_OUT / "own_image_embeddings_paths.json", "w"))
    path = METRICS / "results.json"
    allres = json.load(open(path)) if path.exists() else {}
    allres["own_embeddings"] = res
    json.dump(allres, open(path, "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
