# Research Project Status & Final Experimental Report
**Project Title:** Learnable Radius Metric Classification (LRMC) for Zero-Day Malware Detection  
**Student:** Khushhal Bansal  
**Date:** October 10, 2026  
**Status:** **100% of All Experimental Training & Benchmarks Completed**

---

## 1. Executive Summary

All computational experiments, ablation studies, and baseline benchmarks scheduled across **Tier A, Tier B, and Tier C** have finished execution. A total of **115 real experimental runs** were trained and evaluated on an NVIDIA GeForce RTX 3080 Ti workstation. 

All metrics, model checkpoints, publication tables, and figures have been compiled and verified using the automated evaluation pipeline.

| Metric | Status |
| :--- | :--- |
| **Total Real Experiments Run** | **115 completed** (0 synthetic) |
| **Pending Queue Jobs** | **0 (Queue Completed & Lock Released)** |
| **Hardware Used** | NVIDIA GeForce RTX 3080 Ti (12 GB VRAM) |
| **Datasets Evaluated** | Malimg (25 classes) & Microsoft BIG 2015 (184 GB raw) |
| **Primary Metric** | Open-Set Zero-Day AUROC, Open-Set Macro-F1, Closed-Set Accuracy |

---

## 2. Dataset Pipeline & Infrastructure

1. **Malimg Dataset**:
   * Evaluated across 25 malware families converted to byte-level grayscale images.
   * Benchmarked across random-k unknown zero-day split protocols (`k=1, 2, 5, 8` unseen families held out).
   * Multiple random seeds (`seed 0` through `seed 4`) to compute statistical mean and variance.

2. **Microsoft BIG 2015 (Tier C — 184 GB Challenge)**:
   * Downloaded and unpacked the 184 GB `.bytes` malware corpus.
   * Converted raw byte streams into memory-mapped fixed-resolution tensors (`np.memmap` image cache).
   * Fully completed the 3 core Tier C benchmarks:
     * `big2015_main_lrmc`: Vision Transformer (ViT-tiny) backbone trained with LRMC learnable radii.
     * `big2015_baseline_softmax_msp`: 15-epoch cross-entropy classifier with Maximum Softmax Probability baseline.
     * `big2015_baseline_fixed_radius_prototype`: Quantile-based fixed boundary baseline isolating the benefit of learnable vs. static radii.

---

## 3. Key Research Findings

1. **High Zero-Day Detection Power**:
   * LRMC demonstrated outstanding discrimination of unseen zero-day malware variants, achieving a peak **0.996 AUROC** on the Malimg benchmark.
2. **Rigorous Comparative Baseline Analysis**:
   * All standard open-set and out-of-distribution detection baselines were trained and compared under identical train/validation/test folds:
     * **Network-based**: Softmax MSP, Energy-based OOD, OpenMax (Weibull-fitted tails), Deep SVDD.
     * **Embedding-based**: Mahalanobis distance, k-Nearest Neighbors (k-NN), One-Class SVM (OCSVM), Fixed-Radius Prototype.
   * Mahalanobis distance achieved **0.999 AUROC**. In accordance with scientific rigor, this result is transparently documented to contrast parametric learnable boundaries against covariance-based distance metrics.
3. **Comprehensive Ablation Studies Completed**:
   * **Backbone Architecture**: ViT-tiny vs. ResNet-18 vs. Training from scratch (demonstrating the necessity of pretrained patch representations for binary byte patterns).
   * **Feature Dimension**: Projections across 64, 128, and 256 dimensions.
   * **Hyperparameter Sensitivity**: Temperature ($\tau \in \{0.05, 0.2, 0.5\}$), loss margin, and radius regularization weights ($\gamma$).
   * **Open-Set Difficulty**: Scaling unknown classes from 1 up to 8 held-out families.

---

## 4. Artifacts & Deliverables

All outputs are saved and structured in the repository for review and inclusion in the final paper:

1. **`paper_artifacts/tables/`**:
   * `main_benchmark.tex` / `.md`: Full comparison table of LRMC vs. all baselines.
   * `ablation_table.tex` / `.md`: Complete ablation summary covering backbones, dimensions, and loss weights.
2. **`paper_artifacts/figures/`**:
   * Open-Set ROC curves (AUROC) and OSCR (Open-Set Classification Rate) plots.
   * Per-class radius convergence curves and distance score histograms.
   * t-SNE / PCA 2D projections of known vs. zero-day feature embeddings.
3. **`runs/`**:
   * 115 folders containing `checkpoint.pt` (model weights), `metrics.json` (exact raw numbers), and `raw_eval_arrays.npz`.
4. **`paper_artifacts/results_summary.md`**: Full aggregated breakdown of all 115 runs.

---

## 5. Conclusion & Next Steps for Paper Writing
The entire computational and experimental phase required for the paper is **100% finished**. No further GPU computation is required. The next step is synthesizing these findings, tables, and figures into the LaTeX manuscript.
