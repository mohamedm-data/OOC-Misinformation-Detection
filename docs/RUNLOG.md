# RUNLOG — DATA5500 capstone (NewsCLIPpings OOC detection), 2026-10-13

Machine: Apple M5 Pro, macOS 26.6.2. Python 3.11 venv at `.venv`.
Versions: torch 2.14.1 (MPS available), transformers 5.19.0, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1,
matplotlib 3.11.2, pillow 12.3.0, pyarrow 26.0.0, tqdm 4.70.1.

| Step | Command | Wall time |
|---|---|---|
| 1. Download NewsCLIPpings (HF model repo g-luo/news-clippings; 24 files, 2.6 GB) | `./src/01_download_newsclippings.sh` | 1m53s (~5.5 MB/s) |
| 2. Inventory + curated 24k subset | `.venv/bin/python src/02_prep_data.py` | 21s |
| 3. Cosine baseline + week-6 figures (PCA/t-SNE, AUC/F1 bars, bootstrap CI) | `.venv/bin/python src/03_baseline_similarity.py` | 36s |
| 4a. VisualNews partial tar (bytes 0–1,610,612,735 of 97,916,641,280; server sends Accept-Ranges: bytes) | `curl -r 0-1610612735 -o data/visualnews_partial/origin_head.tar https://www.cs.rice.edu/~vo9/visualnews/origin.tar` | 15m43s (~1.7 MB/s) |
| 4b. Own CLIP ViT-B/32 embeddings + agreement | `.venv/bin/python -I src/04_own_embeddings.py data/visualnews_partial/origin_head.tar` | 1m42s |
| 5. Logistic regression, C sweep, bootstrap CIs (2026-10-18) | `cd src && ../.venv/bin/python 05_logistic_regression.py` | 26s |
| 6. Embedding preprocessing variants, 10 variants x 2,641 images (2026-10-18; first run also downloads OpenAI ViT-B/32 weights, 338 MB) | `cd src && ../.venv/bin/python 06_embedding_preprocessing.py` | 3m27s |
| 7. MLP first draft, CPU, 12 epochs (best epoch 2) (2026-10-18) | `cd src && ../.venv/bin/python 07_mlp.py` | 36–45s |

Notes
- Subset `person_sbert_text_text` is a `data/` folder in the repo (do not filter it out with the "sbert" exclusion).
- Embedding pickles: dict int id -> float32[512], already L2-normalised. Text keyed by caption `id`, image by `image_id`.
- Metadata: dict str id -> record with `caption`, `image_path` (`visual_news/origin/<source>/images/<dir>/<n>.jpg`).
- Tar layout: `origin/<source>/images/<4-digit dir>/<n>.jpg`; first 1.5 GiB contains only `origin/bbc/images/*` (no data.json, no articles). Extraction stops at the truncated last member (ReadError), which is dropped.
- Mapping tar path -> NewsCLIPpings id via metadata `image_path` with the `visual_news/` prefix stripped.
- Seed 5500 everywhere.
- 2026-10-18: added torchvision 0.29.1, ftfy 6.3.1, and OpenAI `clip` 1.0 (from GitHub) for step 6 only. Steps 5 and 7 are deterministic: rerunning them reproduced identical numbers.
