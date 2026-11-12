# Historical Project Ledger: Genomic Framework / Avian Influenza Genomic Intelligence

**Project Title**: Avian Influenza Genomic Intelligence Framework (Genomic Framework)  
**Historical Repository**: `https://github.com/vishnu-varthan63/Genomic-Framework.git`  
**Primary Developer / Author**: Vishnu Varthan Senthilkumar  
**Forensic Audit Date**: October 15, 2026  
**Reconstruction Branch**: `recovery-reconstruction`  
**Evidence Sources**: 
1. `C:\Users\vishn\.gemini\antigravity\brain\e94838ea-523a-4f59-a354-4237c83452ba` (Full conversation transcripts & design artifacts)
2. `D:\vishnu\F\vishnu\f\Projects\GEnomic-framework` (Restored pristine local repository backup dated Sept 21, 2026)
3. `D:\dataset` (Raw genomic repository pool: 5.41 GB, 161 files)
4. `D:\Genomic framework\Genomic-Framework-main.zip` (GitHub source archive)
5. Remote Git history: commits `75263e7` and `78e492b` on `origin/main`

---

## 1. Project Objectives and Scope
The Avian Influenza Genomic Intelligence Framework is an end-to-end computational biology and deep learning system developed to classify Avian Influenza A Virus (AIV) genomic nucleotide sequences into four curated pathogenicity categories:
- **Class 0**: Benign (Healthy host avian genomes: chicken, duck, goose, quail, turkey)
- **Class 1**: Likely Benign (Low-pathogenic AIV strains: H5N2, H5N3, H7N2, H7N3, H9N2)
- **Class 2**: Likely Pathogenic (Intermediate / emerging strains: H5N6, H5N8, H7N1, H7N7, H10N7, H10N8)
- **Class 3**: Pathogenic (Highly pathogenic avian influenza: H5N1, H5N9, H7N9)

**Key Scientific and Practical Goals**:
1. Provide automated bioinformatic quality control, IUPAC ambiguity filtering, and length standardization ($L = 128$ tokens).
2. Execute reference-based sequence comparison and mutation extraction relative to a canonical wild-type sequence.
3. Combine local $k$-mer token self-attention (DNABERT-2-inspired) and extended sub-quadratic long causal convolutions (HyenaDNA-inspired) in a dual-stream architecture.
4. Heuristically prioritize mutations via functional region weighting and spatial Gaussian decay to construct a dense mutation feature tensor ($[32 \times 64]$).
5. Fuse multi-modal sequence and mutation representations using an 8-head Cross-Attention mechanism.
6. Provide explainability via gradient-based feature attribution (Gradient $\times$ Input / SHAP approximation).
7. Deliver a multi-tier presentation system:
   - Full research pipeline (`Genomic_Framework/`)
   - Interactive lightweight demo (`demo/` with Streamlit interface `app.py`)
   - Next.js 16 research platform website (`website/`)
   - IEEE conference research manuscript (`paper/`)

---

## 2. Architecture Decisions
1. **Dual Sequence Representation**:
   - *DNABERT-2-Inspired Extractor*: Sliding-window hexamer ($k=6$, stride $s=1$) tokenization into $V=4,096$ vocabulary. Sinusoidal positional embeddings. 4 transformer encoder layers (8 heads, $d_{model}=256, d_{ffn}=1024$), projected to dimension 768.
   - *HyenaDNA-Inspired Extractor*: Single-token embeddings with relative learnable positional bias. 4 Hyena operator blocks with short 1D depthwise convolution (kernel=3), long causal convolution (filter order $K=64$) with exponential decay envelope, and multiplicative gating. Output dimension 256.
2. **Alignment & Mutation Prioritization**:
   - Implemented as a pairwise character comparator over valid positions simulating global no-gap alignment.
   - Heuristic scoring: $s(m) = w(r) \cdot \exp\left(-\frac{1}{2}\left(\frac{p - \mu_{pos}}{\sigma_{pos}}\right)^2\right)$ with $\mu_{pos}=0.3L, \sigma_{pos}=0.4L$. Top 32 mutations selected.
3. **Cross-Attention Fusion Engine**:
   - Mutation tensor $[B, 32, 64]$ projected to Query space $[B, 32, 512]$.
   - Joint sequence context $\mathbf{C} = \mathrm{Proj}(\mathbf{Z}_{DNA}) + \mathrm{Proj}(\mathbf{Z}_{Hyena}) \in \mathbb{R}^{B \times 128 \times 512}$ serves as Keys and Values.
   - 8-head multi-head cross-attention followed by residual connection, LayerNorm, and Global Average Pooling across the 32 mutation tokens to yield $\mathbf{f} \in \mathbb{R}^{B \times 512}$.
4. **Classification Head**:
   - Two-layer MLP: $\mathrm{Linear}(512, 256) \to \mathrm{GELU} \to \mathrm{LayerNorm} \to \mathrm{Dropout}(0.3) \to \mathrm{Linear}(256, 4)$.
5. **Verified Trainable Parameter Profile**:
   - `dnabert2`: 4,406,528 parameters (42.77%)
   - `hyenadna`: 2,836,224 parameters (27.53%)
   - `fusion`: 2,928,128 parameters (28.42%)
   - `classifier`: 132,868 parameters (1.29%)
   - **Total Verified Parameters**: **10,303,748**

---

## 3. Module Implementation History
| Module Path | Primary Class / Function | Purpose | Historical Date |
|---|---|---|---|
| `Genomic_Framework/pipeline/input_qc.py` | `GenomicSequenceIngester` | Ingest FASTA/FASTQ, IUPAC scrubbing, quality filter ($\gamma \ge 0.50$), pad/truncate to $L=128$. | July 2026 |
| `Genomic_Framework/pipeline/mutation_pipeline.py` | `MutationPipeline`, `MutationAnnotator` | Alignment against reference, mismatch extraction, functional region tagging, top-32 ranking. | July 2026 |
| `Genomic_Framework/pipeline/preprocessor.py` | `GenomicDatasetPreprocessor` | Tokenization, sliding hexamers, dataset PyTorch tensor creation. | July 2026 |
| `Genomic_Framework/pipeline/augmentor.py` | `GenomicDataAugmentor` | Reverse complement, substitutions, indels, codon changes. | July 2026 |
| `Genomic_Framework/pipeline/healthy_downloader.py` | `HealthyGenomeDownloader` | Fetch avian host genomes from NCBI Entrez. | July 2026 |
| `Genomic_Framework/models/dnabert2.py` | `DNABERT2Extractor` | 4-layer transformer hexamer representation module. | July-Aug 2026 |
| `Genomic_Framework/models/hyenadna.py` | `HyenaDNAExtractor` | 4-layer long causal convolution sub-quadratic sequence module. | July-Aug 2026 |
| `Genomic_Framework/models/fusion.py` | `CrossAttentionFusion` | Multi-head cross-attention fusion of mutations and context. | July-Aug 2026 |
| `Genomic_Framework/models/classifier.py` | `MLPClassifier` | Two-layer projection to 4 pathogenicity class logits. | July-Aug 2026 |
| `Genomic_Framework/engine.py` | `GenomicClassificationEngine` | End-to-end PyTorch module combining all 4 sub-modules. | Aug 2026 |
| `Genomic_Framework/training/trainer.py` | `GenomicTrainer` | Standard training loop with early stopping and AdamW. | July 2026 |
| `Genomic_Framework/training/advanced_trainer.py` | `AdvancedGenomicTrainer` | Class-weighted cross entropy, cosine annealing, gradient clipping. | Aug 2026 |
| `Genomic_Framework/inference/advanced_evaluator.py` | `AdvancedGenomicEvaluator` | Hold-out evaluation, metrics, confusion matrix, HTML/MD reports. | Aug 2026 |

---

## 4. File Creation and Modification History
- **Mid-July 2026**: Initial pipeline scripts (`ncbi_downloader.py`, `download_healthy.py`, `preprocessor.py`, `trainer.py`).
- **Late-July 2026**: Augmentation pipeline implemented (`data_augmentation_pipeline.py`, `augment.py`, `dataset_augmented/`).
- **Early-August 2026**: Dual architecture integration (`hyenadna.py`, `fusion.py`, `engine.py`, `train_advanced.py`, `evaluate_advanced.py`). Initial release v1.0.0 committed to Git (`75263e7`).
- **Mid-August 2026**: Demo system development (`demo/` directory, `demo/app.py` Streamlit, `demo/train_demo.py`, `demo/inference_demo.py`).
- **Late-August 2026**: 
  - Demo path bug fixes committed (`78e492b`).
  - IEEE conference paper generation (`paper/paper.tex`, `references.bib`, 7 figures, `generate_paper_figures.py`).
  - Forensic audit and LaTeX table layout repairs (`FINAL_AUDIT_REPORT.md`, `LATEX_FORMAT_AUDIT.md`, `PDF_VISUAL_AUDIT.md`, `VISUAL_LAYOUT_AUDIT.md`, `FINAL_LATEX_VALIDATION.md`).
- **September 21, 2026**: Entire project backed up to `D:\vishnu\F\vishnu\f\Projects\GEnomic-framework` before system OS cleanup.
- **October 15, 2026**: Forensic project recovery, workspace restoration to `F:\vishnu\f\Projects\GEnomic-framework`, and creation of comprehensive recovery documentation suite.

---

## 5. Dependency and Environment Decisions
- **Python Version**: Python 3.10 – 3.14 compatible.
- **Core ML Stack**: `torch >= 2.0.0`, `torchmetrics >= 1.0.0`, `scikit-learn >= 1.3.0`, `numpy >= 1.24.0`, `pandas >= 2.0.0`.
- **Bioinformatics**: `biopython >= 1.81`.
- **Plotting / Explainability**: `matplotlib >= 3.7.0`, `seaborn >= 0.12.0`, `shap >= 0.42.0`.
- **Demo / Web**: `streamlit >= 1.28.0`, `pyyaml >= 6.0`.
- **Website**: Next.js 16 (React 19, Tailwind CSS, TypeScript).

---

## 6. Dataset Collection, Augmentation, and Split Analysis
- **Raw Repository Pool**: 149 files, 1,048,666 sequences, 4.78 billion base pairs across 14 viral subtypes (`D:\dataset\raw`) and 5 healthy avian species (`D:\dataset\healthy`).
- **Augmentation Source**: 2,940 initial curated sequences expanded via `data_augmentation_pipeline.py`.
- **Active Experimental Dataset**: **35,211 sequences** across 72 FASTA files (45,909,430 base pairs):
  - Benign: 1,191 sequences (3.38%, 4.6x expansion)
  - Likely Benign: 18,185 sequences (51.65%, 18.2x expansion)
  - Likely Pathogenic: 10,855 sequences (30.83%, 9.4x expansion)
  - Pathogenic: 4,980 sequences (14.14%, 9.6x expansion)
- **File-Level Stratified Splits** (Zero sequence data leakage across files):
  - Train: 48 files, 23,520 sequences (66.8%)
  - Validation: 10 files, 5,000 sequences (14.2%)
  - Hold-Out Test: 14 files, 6,691 sequences (19.0%)

---

## 7. Training Experiments and Checkpoint Status
1. **Full 10.30M Parameter Model**:
   - Trained on the 35,211-sequence curated benchmark.
   - Hold-out test evaluation produced:
     - Overall Accuracy: **86.03%**
     - Balanced Accuracy: **75.29%**
     - Macro F1: **76.31%**
     - Weighted F1: **84.89%**
     - Macro ROC-AUC: **89.47%**
     - Macro PR-AUC: **81.44%**
     - Matthews Correlation Coefficient (MCC): **0.7821**
     - Cohen's Kappa: **0.7795**
     - Inference Latency: **19.82 ms / sequence**
   - *Checkpoint Status*: `checkpoints_augmented/best_model.pt` (historical size ~122.8 MB) was stored in temporary/working directories during training and was not committed to Git (gitignored to comply with repository file size limits). The training scripts (`train_advanced.py`) and dataset (`dataset_augmented/`) survive in full to enable deterministic retraining.
2. **Lightweight Demo Model**:
   - Trained via `demo/train_demo.py` for 5 epochs.
   - *Checkpoint Status*: **VERIFIED PRESENT** at `outputs/model_demo.pt` (634,485 bytes, dated August 25, 2026).
   - Training log preserved at `outputs/train_log.csv`.

---

## 8. Demo and Interface History
- **Historical Errors Resolved**:
  1. `ModuleNotFoundError: No module named 'demo'`: Resolved by executing via `python -m demo.train_demo` from project root and adding robust `Path(__file__).resolve().parent.parent` root anchors.
  2. `FileNotFoundError: demo_config.yaml`: Resolved by placing default config inside `demo/demo_config.yaml` with fallback path resolution.
  3. `FileNotFoundError: data/references.fasta`: Resolved by generating synthetic/default reference sequence when reference file is missing.
- **Current Demo Status**: Fully functional training (`python -m demo.train_demo`), inference (`python -m demo.inference_demo`), and Streamlit app (`streamlit run demo/app.py`).

---

## 9. Research Paper and Documentation Status
- **Manuscript**: Complete, audited IEEE conference paper in `paper/paper.tex` and root `paper.tex` adhering to standard `\documentclass[conference]{IEEEtran}`.
- **Figures**: All 7 publication figures regenerated at 300 DPI in `figures/` and `paper/figures/`.
- **Tables**: All 10 tables engineered with `tabularx` to prevent horizontal column overflows.
- **Citations**: 28 references in `references.bib` and `paper/references.bib`, 23 active citations in text.
- **Audits**: 5 pre-submission audit documents created during late August 2026.
