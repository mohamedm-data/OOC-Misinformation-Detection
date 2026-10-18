"""Step 2: data inventory + curated stratified subset of NewsCLIPpings merged_balanced.

Outputs:
  outputs/metrics/data_summary.json            counts, label balance, coverage, duplicates
  data/curated/{train,val,test}.parquet
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NC = ROOT / "data" / "news_clippings"
OUT = ROOT / "outputs" / "metrics"
CUR = ROOT / "data" / "curated"
OUT.mkdir(parents=True, exist_ok=True)
CUR.mkdir(parents=True, exist_ok=True)

SUBSETS = ["merged_balanced", "person_sbert_text_text", "scene_resnet_place",
           "semantics_clip_text_image", "semantics_clip_text_text"]
SPLITS = ["train", "val", "test"]
TARGET = {"train": 19200, "val": 2400, "test": 2400}  # 24,000 total, 80/10/10
SEED = 5500


def load_ann(subset, split):
    with open(NC / "data" / subset / f"{split}.json") as f:
        d = json.load(f)
    return d.get("source_datasets"), pd.DataFrame(d["annotations"])


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    summary = {"annotation_counts": {}, "example_annotation": None}
    source_map = None
    mb = {}
    for subset in SUBSETS:
        for split in SPLITS:
            src, df = load_ann(subset, split)
            if subset == "merged_balanced":
                source_map = src
                mb[split] = df
                if summary["example_annotation"] is None:
                    summary["example_annotation"] = json.loads(df.iloc[1].to_json())
            summary["annotation_counts"][f"{subset}/{split}"] = {
                "n": int(len(df)),
                "falsified": int(df["falsified"].sum()),
                "genuine": int((~df["falsified"]).sum()),
                "columns": list(df.columns),
                "dup_rows": int(df.duplicated(["id", "image_id"]).sum()),
                "missing_values": int(df.isna().sum().sum()),
            }
    summary["source_datasets"] = source_map

    # merged_balanced: per-method balance + embedding / metadata coverage
    summary["merged_balanced"] = {}
    meta_example = None
    for split in SPLITS:
        df = mb[split].copy()
        df["method"] = df["source_dataset"].astype(str).map(source_map)
        img_emb = load_pkl(NC / "embeddings/clip_image_embeddings" / f"clip_image_embeddings_{split}.pkl")
        txt_emb = load_pkl(NC / "embeddings/clip_text_embeddings" / f"clip_text_embeddings_{split}.pkl")
        if split == "train":
            k = next(iter(img_emb))
            summary["embedding_format"] = {
                "image_pkl_type": type(img_emb).__name__, "image_n": len(img_emb),
                "image_key_type": type(k).__name__, "image_value_shape": list(np.asarray(img_emb[k]).shape),
                "image_value_dtype": str(np.asarray(img_emb[k]).dtype),
            }
            k2 = next(iter(txt_emb))
            summary["embedding_format"].update({
                "text_pkl_type": type(txt_emb).__name__, "text_n": len(txt_emb),
                "text_key_type": type(k2).__name__, "text_value_shape": list(np.asarray(txt_emb[k2]).shape),
            })
        with open(NC / "metadata" / f"{split}.json") as f:
            meta = json.load(f)
        if meta_example is None:
            mk = next(iter(meta))
            meta_example = {"container": type(meta).__name__, "key_type": type(mk).__name__,
                            "example_key": mk, "example_record": meta[mk]}
        meta_keys = set(map(str, meta.keys()))
        img_keys = set(map(str, img_emb.keys()))
        txt_keys = set(map(str, txt_emb.keys()))
        ids, iids = df["id"].astype(str), df["image_id"].astype(str)
        s = {
            "n": int(len(df)),
            "label_balance": df["falsified"].value_counts().rename(str).to_dict(),
            "method_x_label": {f"{m}|{l}": int(c) for (m, l), c in df.groupby(["method", "falsified"]).size().items()},
            "unique_caption_ids": int(ids.nunique()), "unique_image_ids": int(iids.nunique()),
            "dup_pairs": int(df.duplicated(["id", "image_id"]).sum()),
            "genuine_with_id_ne_image_id": int(((~df["falsified"]) & (ids != iids)).sum()),
            "text_emb_coverage_id": float(ids.isin(txt_keys).mean()),
            "image_emb_coverage_image_id": float(iids.isin(img_keys).mean()),
            "meta_coverage_id": float(ids.isin(meta_keys).mean()),
            "meta_coverage_image_id": float(iids.isin(meta_keys).mean()),
            "n_image_emb": len(img_keys), "n_text_emb": len(txt_keys), "n_meta": len(meta_keys),
        }
        summary["merged_balanced"][split] = s
        print(split, json.dumps(s))

        # ---- curated stratified subset: equal cells over method x label ----
        rec = lambda i: meta.get(i) or meta.get(int(i)) or {}
        df["caption"] = ids.map(lambda i: rec(i).get("caption"))
        df["image_path"] = iids.map(lambda i: rec(i).get("image_path"))
        keep = ids.isin(txt_keys) & iids.isin(img_keys)
        df = df[keep & ~df.duplicated(["id", "image_id"])]
        cells = df.groupby(["method", "falsified"])
        per_cell = TARGET[split] // cells.ngroups
        parts = []
        for _, g in cells:
            parts.append(g.sample(n=min(per_cell, len(g)), random_state=SEED))
        cur = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
        cur = cur[["id", "image_id", "falsified", "method", "source_dataset", "similarity_score", "caption", "image_path"]]
        cur.to_parquet(CUR / f"{split}.parquet", index=False)
        summary.setdefault("curated", {})[split] = {
            "n": int(len(cur)), "per_cell_target": per_cell,
            "method_x_label": {f"{m}|{l}": int(c) for (m, l), c in cur.groupby(["method", "falsified"]).size().items()},
            "caption_missing": int(cur["caption"].isna().sum()),
        }
        del img_emb, txt_emb, meta

    summary["metadata_format"] = meta_example
    # split separation check
    sets = {sp: set(pd.read_parquet(CUR / f"{sp}.parquet")["id"]) for sp in SPLITS}
    summary["curated_id_overlap"] = {f"{a}&{b}": len(sets[a] & sets[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    with open(OUT / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps({k: summary[k] for k in ["curated", "curated_id_overlap", "metadata_format", "embedding_format"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
