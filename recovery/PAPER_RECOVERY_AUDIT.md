# IEEE Research Paper Recovery Audit

**Manuscript Title**: *Avian Influenza Genomic Intelligence: A Hybrid Deep Learning Architecture for Pathogenicity Category Prediction and Mutation-Aware Sequence Modeling*  
**Format**: IEEE Conference Proceedings (`\documentclass[conference]{IEEEtran}`)  
**Audit Date**: October 15, 2026  
**Auditor**: Lead System Reconstruction Architect  

---

## 1. LaTeX Source Files & Integrity

| Source File | Location | Lines | Status | Validation Result |
|---|---|:---:|:---:|:---:|
| `paper.tex` | `paper/paper.tex` | 607 | `VERIFIED_PRESENT` | Compiles cleanly; standard IEEEtran |
| `paper.tex` (Root Mirror) | `./paper.tex` | 607 | `VERIFIED_PRESENT` | Byte-for-byte synchronized with `paper/paper.tex` |
| `references.bib` | `paper/references.bib` | 185 | `VERIFIED_PRESENT` | 28 entries; valid BibTeX format |
| `references.bib` (Root Mirror)| `./references.bib` | 185 | `VERIFIED_PRESENT` | Synchronized with `paper/references.bib` |
| `validate_paper.py` | `./validate_paper.py` | 85 | `VERIFIED_PRESENT` | 100% check pass (0 errors) |
| `scripts/check_ieee_format.py`| `scripts/check_ieee_format.py` | 120 | `VERIFIED_PRESENT` | 100% check pass (0 errors) |

---

## 2. Table Layout and IEEE Margin Compliance

All 10 tables were engineered to eliminate horizontal margin overflows into adjacent columns:

| Table # | LaTeX Label | Target Float | Width Constraint | Styling | Status |
|---|---|:---:|:---:|:---:|:---:|
| **Table I** | `tab:literature_comparison` | `table*` (Full Page) | `\textwidth` | `tabularx` (7 auto-wrapping columns) | `PASS` |
| **Table II** | `tab:subtype_mapping` | `table` (Single Column) | `\columnwidth` | `tabularx` (wrapped subtypes) | `PASS` |
| **Table III** | `tab:dataset_composition` | `table` (Single Column) | `\columnwidth` | `tabularx` (stage sequence/base counts) | `PASS` |
| **Table IV** | `tab:augmentation_summary` | `table` (Single Column) | `\columnwidth` | `tabularx` (multipliers & shares) | `PASS` |
| **Table V** | `tab:dataset_stats` | `table` (Single Column) | `\columnwidth` | `tabularx` (split class matrix) | `PASS` |
| **Table VI** | `tab:hyperparameters` | `table` (Single Column) | `\columnwidth` | `tabularx` (optimizer & layers) | `PASS` |
| **Table VII** | `tab:param_count` | `table` (Single Column) | `\columnwidth` | `tabularx` (verified 10.30M breakdown) | `PASS` |
| **Table VIII** | `tab:benchmark_metrics` | `table` (Single Column) | `\columnwidth` | `tabularx` (hold-out test scores) | `PASS` |
| **Table IX** | `tab:class_metrics` | `table` (Single Column) | `\columnwidth` | `tabularx` (per-class precision/recall/F1) | `PASS` |
| **Table X** | `tab:ablation_design` | `table` (Single Column) | `\columnwidth` | `tabularx` (architectural ablation matrix) | `PASS` |

---

## 3. High-Resolution Figure Inventory (All 300 DPI)

| Figure # | File Path | Layout | Dimension | Subject / Description |
|---|---|:---:|:---:|---|
| **Fig. 1** | `figures/architecture.png` | `figure*` | `0.95\textwidth` | Complete system architecture (7 clean stages, 10.30M parameters) |
| **Fig. 2** | `figures/pipeline.png` | `figure` | `\columnwidth` | Bioinformatic data processing flow (QC, standardization, alignment) |
| **Fig. 3** | `figures/model_architecture.png` | `figure` | `\columnwidth` | Neural network computational graph with exact tensor shapes |
| **Fig. 4** | `figures/training_curves.png` | `figure` | `\columnwidth` | Cross-entropy loss and accuracy optimization dynamics |
| **Fig. 5** | `figures/confusion_matrix.png` | `figure` | `0.85\columnwidth` | 4x4 hold-out test confusion matrix (N = 6,691) |
| **Fig. 6** | `figures/class_performance.png` | `figure` | `\columnwidth` | Class-wise precision, recall, and F1-score comparison |
| **Fig. 7** | `figures/shap_summary.png` | `figure` | `\columnwidth` | Gradient-based feature attribution ranking top functional loci |

---

## 4. Citation and Mathematical Verification
- **Total Math Equations**: 29 numbered equations. All wide equations (such as Eq. 19 for mutation vector concatenation) split using `\begin{aligned}` to prevent margin violations.
- **Sequential Numbering**: (1) through (29) verified with zero gaps.
- **Citations**: 23 unique citations in paper matching 28 BibTeX entries in `references.bib`. Zero undefined citations.
- **Labels and References**: 56 declared labels; all 17 `\ref{}` and 1 `\eqref{}` resolve cleanly.
