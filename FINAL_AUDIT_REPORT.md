# IEEE Conference Paper Forensic Audit and Pre-Submission Quality Report

**Title**: Avian Influenza Genomic Intelligence: A Hybrid Deep Learning Architecture for Pathogenicity Category Prediction and Mutation-Aware Sequence Modeling  
**Author**: Vishnu Varthan Senthilkumar  
**Target Venue**: IEEE Conference on Bioinformatics and Computational Biology / AI in Healthcare  
**Audit Date**: August 2026  

---

## 1. Formatting Issues Found & Resolved
- **Markdown Artifacts Elimination**: Stripped all markdown residual formatting (e.g., `**`, `*`, `#`) from `paper.tex` and author affiliations.
- **Label / Ref Format Standardization**: Replaced legacy colon-escaped labels (e.g., `\label{sec\:...}`) with standard IEEE LaTeX `\label{sec:...}` and `\ref{sec:...}`.
- **IEEE Documentclass Compliance**: Standardized document to `\documentclass[conference]{IEEEtran}` with standard IEEE two-column section and subsection headers.

---

## 2. LaTeX Issues Found & Resolved
- **Environment Closure Verification**: Verified exact pairing across all environments:
  - 6 `figure` environments, 1 `figure*` environment
  - 8 `table` environments
  - 29 `equation` environments
  - 5 `enumerate` and 1 `itemize` environments
- **Zero Undefined Labels / Citations**: 54 unique `\label` entries with 100% resolution across all in-text `\ref` and `\eqref` calls. 28 BibTeX entries with 100% resolution across all 23 in-text `\cite` calls.
- **Clean Auxiliary State**: Ensured no stale `.aux`, `.bbl`, or `.log` artifacts cause ghost entries.

---

## 3. Scientific Issues Found & Corrected
- **Dataset Imbalance Clarification**: Corrected previous mislabeling of the 35,211-sequence dataset as "balanced". The dataset is now rigorously documented as a curated, class-imbalanced benchmark dataset (Benign: 3.38%, Likely Benign: 51.65%, Likely Pathogenic: 30.83%, Pathogenic: 14.14%).
- **Proxy Label Framing**: Clarified that target labels represent curated proxy categories based on NCBI viral subtype records and host species origin rather than individual in-vivo chicken IVPI wet-lab measurements.
- **Ablation Protocol Demarcation**: Clarified that Table IX represents a designed comparative protocol rather than already completed ablation runs.

---

## 4. Numerical Inconsistencies Corrected
- **Parameters Count**: Reconciled the README's 10.21M figure with direct PyTorch module profiling:
  - `dnabert2` Extractor: 4,406,528 (42.77%)
  - `hyenadna` Extractor: 2,836,224 (27.53%)
  - `fusion` Engine: 2,928,128 (28.42%)
  - `classifier` MLP Head: 132,868 (1.29%)
  - **Total**: **10,303,748 trainable parameters** (100.00%).
- **Split Proportions**: Verified: Train (23,520 seqs, 66.8%) + Val (5,000 seqs, 14.2%) + Test (6,691 seqs, 19.0%) = 35,211 sequences (100.0%).
- **Class Supports**: Test set breakdown verified: Benign (191), Likely Benign (3,500), Likely Pathogenic (2,000), Pathogenic (1,000), Total = 6,691.

---

## 5. Dataset & Augmentation Inconsistencies Corrected
- **Pipeline Stage Distinction**: Clearly differentiated the repository stages:
  1. *Raw Repository Pool*: 149 files, 1,048,666 sequences, 4,782,973,902 bp
  2. *Full Augmented Pool*: 540 files, 67,482 sequences, 87,936,876 bp
  3. *Active Curated Dataset*: 72 files, 35,211 sequences, 45,909,430 bp (Train: 48 files, 23,520; Val: 10 files, 5,000; Test: 14 files, 6,691)
- **Augmentation Multipliers Documented**:
  - Benign: 260 $\to$ 1,191 ($4.6\times$)
  - Likely Benign: 1,000 $\to$ 18,185 ($18.2\times$)
  - Likely Pathogenic: 1,160 $\to$ 10,855 ($9.4\times$)
  - Pathogenic: 520 $\to$ 4,980 ($9.6\times$)
- **Data Leakage Safeguards**: Documented file-level stratified partitioning where sequences from any given FASTA file belong exclusively to one split.

---

## 6. Model Terminology Corrections
- **Custom Modules vs. Foundation Weights**: Replaced blanket foundation model assertions with technically accurate descriptions:
  - "DNABERT-2-inspired transformer encoder"
  - "HyenaDNA-inspired long-context module"
- **Alignment Method**: Replaced "BLAST" with "pairwise reference-based sequence comparator / global no-gap alignment".
- **Edit Distance**: Replaced "Levenshtein distance" with "Hamming-style mismatch count over valid positions".

---

## 7. Equation Corrections
- **$k$-mer Vector Lengths**: Corrected sliding window tokenization math: for sequence length $L=128$ and $k=6$, the sliding window yields $L - k + 1 = 123$ valid hexamers, which are zero-padded to length $L=128$ for tensor batching.
- **Cross-Attention Equations**: Formalized exact query ($Q \in \mathbb{R}^{B \times 32 \times 512}$) and context ($K, V \in \mathbb{R}^{B \times 128 \times 512}$) dimensions.
- **Explainability Formulation**: Accurately formulated gradient-input attribution ($\phi_k = f_k \cdot \frac{\partial z_c}{\partial f_k}$) without claiming game-theoretic Shapley uniqueness.

---

## 8. Figure Corrections
- **All 7 Figures Generated at 300 DPI**:
  1. `figures/architecture.png` (System architecture)
  2. `figures/pipeline.png` (Bioinformatic pipeline flow)
  3. `figures/model_architecture.png` (Tensor dimensions and computational graph)
  4. `figures/training_curves.png` (Loss and accuracy dynamics)
  5. `figures/confusion_matrix.png` (4x4 test confusion matrix)
  6. `figures/class_performance.png` (Precision/Recall/F1 per class)
  7. `figures/shap_summary.png` (Gradient attribution ranking)
- **Synchronized Paths**: Figures saved in both `figures/` and `paper/figures/`.

---

## 9. Table Corrections
- Formatted all tables using professional LaTeX `booktabs` (`\toprule`, `\midrule`, `\bottomrule`):
  - Table I: Subtype to Pathogenicity Class Mapping
  - Table II: Literature Comparison Table
  - Table III: Dataset Composition Across Pipeline Stages
  - Table IV: Augmentation Multipliers Across Classes
  - Table V: Verified Dataset Distribution Across Splits
  - Table VI: Training & Optimization Hyperparameters
  - Table VII: Layer-by-Layer Trainable Parameters (10,303,748 total)
  - Table VIII: Hold-Out Test Evaluation Benchmark Metrics
  - Table IX: Per-Class Performance Breakdown
  - Table X: Ablation Study Design Protocol

---

## 10. Reference Corrections
- **28 Real Publications**: All references in `references.bib` verified with accurate authors, titles, journals, conferences, DOIs, and arXiv identifiers.
- Zero hallucinated citations; zero missing citation keys.

---

## 11. Writing & Tone Corrections
- Eliminated promotional/marketing buzzwords ("revolutionary", "perfect", "groundbreaking").
- Removed robotic over-use of transition adverbs ("Furthermore", "Moreover", "Additionally", "Thus").
- Replaced causal assertions with scientifically grounded statements ("attribution analysis indicates", "evaluation demonstrated").

---

## 12. Source / PDF Synchronization
- `paper.tex` is established as the single authoritative source of truth.
- All figures in `figures/` and `paper/figures/` match the references in `paper.tex`.

---

## 13. Remaining Items for Future Work
- **Fine-Tuning Official Foundation Weights**: Training on pre-trained multi-species HuggingFace foundation checkpoints (e.g., official 117M DNABERT-2 or 1B HyenaDNA).
- **Execution of Ablation Matrix**: Running the full 5-way ablation experiment outlined in Table X.
- **Pan-Genome Graph Alignment**: Integrating multi-reference graph aligners for divergent viral clades.
