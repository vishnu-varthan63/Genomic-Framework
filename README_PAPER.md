# IEEE Conference Paper Submission Package & Audit Report

**Title**: Avian Influenza Genomic Intelligence: A Hybrid Deep Learning Architecture for Pathogenicity Phenotype Classification and Mutation-Aware Sequence Modeling  
**Author**: Vishnu Varthan Senthilkumar  
**Target Venue**: IEEE Conference on Bioinformatics / Computational Biology / AI in Healthcare  

---

## 1. Package File Manifest

| File | Purpose | Location |
|---|---|---|
| `paper.tex` | Complete, standalone IEEEtran LaTeX source code | `F:\vishnu\f\Projects\GEnomic-framework\paper.tex` & `paper\paper.tex` |
| `references.bib` | Fully verified BibTeX bibliography with real citations & DOIs | `F:\vishnu\f\Projects\GEnomic-framework\references.bib` & `paper\references.bib` |
| `README_PAPER.md` | Verification report, mathematical formulation audit, figure specs | `F:\vishnu\f\Projects\GEnomic-framework\README_PAPER.md` |

---

## 2. Experimental & Bioinformatic Audit Summary

### 2.1 Dataset Verification
- **Data Source**: NCBI Entrez viral genomic records (Influenza A virus segments) and host reference genomes.
- **Total Workspace FASTA Files**: 689 files
- **Raw Unaugmented Dataset**: 149 files, 1,048,666 sequences, 4,782,973,902 base pairs.
- **Active Curated Dataset (Balanced Experiment Set)**: **72 files, 35,211 sequences, 45,909,430 base pairs**
  - **Benign (Class 0)**: 3 files, 1,191 sequences (3.38%), 1,385,010 bases
  - **Likely Benign (Class 1)**: 37 files, 18,185 sequences (51.65%), 24,653,531 bases
  - **Likely Pathogenic (Class 2)**: 22 files, 10,855 sequences (30.83%), 13,546,658 bases
  - **Pathogenic (Class 3)**: 10 files, 4,980 sequences (14.14%), 6,324,231 bases
- **Splits**:
  - **Train (66.8%)**: 48 files, 23,520 sequences (Benign: 500, Likely Benign: 12,185, Likely Pathogenic: 7,355, Pathogenic: 3,480)
  - **Validation (14.2%)**: 10 files, 5,000 sequences (Benign: 500, Likely Benign: 2,500, Likely Pathogenic: 1,500, Pathogenic: 500)
  - **Hold-out Test (19.0%)**: 14 files, 6,691 sequences (Benign: 191, Likely Benign: 3,500, Likely Pathogenic: 2,000, Pathogenic: 1,000)

### 2.2 Model Parameter Audit (Direct Profiling)
Direct parameter measurement in PyTorch for `GenomicClassificationEngine`:
- **`dnabert2` Extractor**: 4,406,528 parameters (42.77%)
- **`hyenadna` Extractor**: 2,836,224 parameters (27.53%)
- **`fusion` Cross-Attention**: 2,928,128 parameters (28.42%)
- **`classifier` MLP Head**: 132,868 parameters (1.29%)
- **TOTAL**: **10,303,748 parameters** (10.30M)

### 2.3 Benchmark Evaluation Metrics (Hold-out Test: 6,691 sequences)
- **Classification Accuracy**: 86.03%
- **Macro ROC-AUC**: 89.47%
- **Weighted F1-Score**: 84.89%
- **Macro F1-Score**: 76.31%
- **Precision-Recall AUC (PR-AUC)**: 81.44%
- **Balanced Accuracy**: 75.29%
- **Matthews Correlation Coefficient (MCC)**: 0.7821
- **Cohen's Kappa ($\kappa$)**: 0.7795
- **Per-sequence Inference Latency**: 19.82 ms

---

## 3. Required Figure Asset Specifications

The LaTeX paper references the following figure files in the `figures/` subdirectory:

| Figure Tag in LaTeX | Referenced Path | Description |
|---|---|---|
| `Fig. 1` (`fig:architecture`) | `figures/fig1_architecture.png` | Complete system dataflow: QC $\to$ Tokenization $\to$ Dual Extractors $\to$ Cross-Attention $\to$ MLP $\to$ SHAP |
| `Fig. 2` (`fig:pipeline`) | `figures/fig2_pipeline.png` | Bioinformatic ingestion, QC scrubbing, and $k$-mer sliding window tokenization ($k=6$) |
| `Fig. 3` (`fig:mutation`) | `figures/fig3_mutation.png` | Pairwise alignment, mismatch extraction, spatial Gaussian prioritization, and $[32 \times 64]$ tensor encoding |
| `Fig. 4` (`fig:fusion`) | `figures/fig4_fusion.png` | Cross-attention query-key-value interaction ($Q=\text{Mutations}, K=V=\text{Sequence Context}$) |
| `Fig. 5` (`fig:loss`) | `figures/fig5_loss.png` | Training and validation cross-entropy loss convergence curve over 50 epochs |
| `Fig. 6` (`fig:accuracy`) | `figures/fig6_accuracy.png` | Training and validation accuracy curves showing learning dynamics |
| `Fig. 7` (`fig:confusion_matrix`) | `figures/fig7_confusion_matrix.png` | $4 \times 4$ normalized confusion matrix across the 4 pathogenicity classes |
| `Fig. 8` (`fig:class_metrics`) | `figures/fig8_class_metrics.png` | Bar chart comparing Class-wise Precision, Recall, and F1-Scores |
| `Fig. 9` (`fig:shap_global`) | `figures/fig9_shap_global.png` | Global SHAP feature importance across top 20 latent channels |
| `Fig. 10` (`fig:shap_local`) | `figures/fig10_shap_local.png` | Local waterfall attribution plot for a sample H5N1 pathogenic prediction |

---

## 4. Compilation Instructions

To compile the LaTeX source into an IEEE conference PDF:

```bash
# Method 1: pdflatex + bibtex
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex

# Method 2: latexmk
latexmk -pdf paper.tex
```
