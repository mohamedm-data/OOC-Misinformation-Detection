# Blocker log

Format: date, blocker, how it was found, resolution.

| Date | Blocker | How found | Resolution |
|---|---|---|---|
| 2026-10-05 | Cloud development environment could not reach huggingface.co or www.cs.rice.edu (proxy returned 403 on CONNECT). | Download attempts failed. | Resolved 2026-10-13: all downloads and the CLIP runs were done on my own laptop (Apple Silicon, PyTorch MPS), where both hosts are reachable. |
| 2026-10-05 | Expected VisualNews images to need an access request. | Checked https://www.cs.rice.edu/~vo9/visualnews/ | Not a blocker: it is an open directory with `origin.tar` (91 GB, images) and `articles.tar.gz` (4.2 GB). The real constraint is size. |
| 2026-10-05 | Open design question on handling dead image URLs. | Read the NewsCLIPpings annotation and metadata format. | Not applicable: annotations reference VisualNews ids, and metadata `image_path` points inside the VisualNews archive, not to web URLs. Removed from pending decisions. |
| 2026-10-13 | `origin.tar` downloads at about 1.7 MB/s (roughly 15 hours for 91 GB). | Measured while streaming. | The server supports HTTP range requests (`Accept-Ranges: bytes`), so only the first 1.5 GiB was streamed (19,478 images, all from `origin/bbc/`). Modeling uses the provided CLIP embeddings, so it does not depend on having every image. |
| 2026-10-13 | A truncated tar always ends with `ReadError: unexpected end of data`. | `tarfile` stream extraction. | `src/04_own_embeddings.py` stops cleanly at the truncation point and drops the truncated last file (20 other images were unreadable and skipped). |

## Known modeling caveat
The `semantics_clip_text_image` subset pairs captions with images chosen for high CLIP image-text similarity, so a raw CLIP-similarity threshold scores below chance there by construction (test ROC-AUC 0.458). This is reported as a per-method finding, not a bug.
