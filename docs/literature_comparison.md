# How the results compare with published NewsCLIPpings work

Published papers report **accuracy**, not ROC-AUC, on the official **Merged/Balanced test split**. The fair comparison is therefore our accuracy on the full official test split (7,264 pairs), with the decision threshold chosen on validation (from `results.json`, key `full_split.full_test_split`).

## Same setting: CLIP features, no external evidence

| Method | Features | Test accuracy, Merged/Balanced |
|---|---|---|
| Luo et al. 2021, CLIP ViT-B/32 classifier (Table 4) | CLIP ViT-B/32 | 0.602 |
| Luo et al. 2021, CLIP RN50 / RN101 (Table 4) | CLIP RN50 / RN101 | 0.616 / 0.660 |
| Luo et al. 2021, best fine-tuning variant, RN50-af (Table 8) | CLIP RN50 | 0.673 |
| Human annotators, 200-pair sample (Luo et al., Table 6) | n/a | 0.656 |
| **This project: logistic regression, full split** | frozen CLIP ViT-B/32 | **0.685** [0.674, 0.696] |
| **This project: MLP, full split (step 8)** | frozen CLIP ViT-B/32 | **0.714** [0.702, 0.724] |
| Papadopoulos et al. 2024, MUSE-MLP with image-text similarity only (Table 4) | CLIP | 0.807 |

These are per-subset accuracies for the same CLIP ViT-B/32 backbone (Luo et al. Table 4 vs. our step 8 MLP on the full test split):

| Subset | Luo et al. ViT-B/32 | Our MLP |
|---|---|---|
| Semantics, CLIP text-image | 0.670 | 0.613 |
| Semantics, CLIP text-text | 0.694 | 0.835 |
| Person, SBERT text-text | 0.610 | 0.686 |
| Scene, ResNet place | 0.682 | 0.722 |

**Reading:**
- With the same CLIP ViT-B/32 features, our frozen-embedding models beat the original paper's CLIP classifiers (0.714 vs. 0.602), and they also beat the human annotators' 0.656.
- The exception is the CLIP text-image swaps. There the original paper's model does better (0.670 vs. 0.613), which is why step 9 targets that subset.
- The no-evidence MUSE-MLP result (0.807) shows there is still room above our number using similarity features alone. Its exact setup is not given for every row, and it may use a larger CLIP model.

## Different setting: methods that retrieve external evidence

These methods search the web for related images and articles, then compare the pair against what they find:

| Method (as reported by Papadopoulos et al. 2024, Table 2) | Accuracy |
|---|---|
| CCN | 0.847 |
| SEN / ECENet / SNIFFER | 0.871 / 0.877 / 0.884 |
| RED-DOT (B/32 / L/14) | 0.878 / 0.903 |
| AITR (B/32 / L/14) | 0.898 / 0.933 |

They are not directly comparable to this project, which uses only the image and caption. Papadopoulos et al. also argue that much of their advantage comes from a shortcut: retrieval often returns the original article, so genuine pairs look very similar to their evidence. On a harder benchmark (VERITE, true vs. miscaptioned), the same kind of model drops to about 0.51.

## Sources
- Luo, Darrell, Rohrbach. *NewsCLIPpings: Automatic Generation of Out-of-Context Multimodal Media.* EMNLP 2021. https://arxiv.org/abs/2104.05893
- Papadopoulos et al. *Similarity over Factuality: Are we making progress on multimodal out-of-context misinformation detection?* 2024. https://arxiv.org/abs/2407.13488

The figures were taken from the papers' tables. Re-check them against the PDFs before quoting them in the final report.
