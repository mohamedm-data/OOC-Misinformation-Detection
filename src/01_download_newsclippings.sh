#!/bin/bash
# Download NewsCLIPpings annotations, metadata and CLIP embeddings from HF model repo g-luo/news-clippings
set -e
cd "$(dirname "$0")/.."  # run from the project root
BASE=https://huggingface.co/g-luo/news-clippings/resolve/main
OUT=data/news_clippings
FILES=()
for s in merged_balanced person_sbert_text_text scene_resnet_place semantics_clip_text_image semantics_clip_text_text; do
  for sp in train val test; do FILES+=("data/$s/$sp.json"); done
done
for sp in train val test; do
  FILES+=("metadata/$sp.json" "embeddings/clip_image_embeddings/clip_image_embeddings_$sp.pkl" "embeddings/clip_text_embeddings/clip_text_embeddings_$sp.pkl")
done
for f in "${FILES[@]}"; do
  mkdir -p "$OUT/$(dirname $f)"
  curl -sSL --fail -C - --retry 4 -o "$OUT/$f" "$BASE/$f" && echo "ok $f $(stat -f%z $OUT/$f)"
done
