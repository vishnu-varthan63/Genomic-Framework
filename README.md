# 🧬 Avian Influenza Genomic Intelligence System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16.0-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()

> **End-to-End Hybrid Deep Learning Framework for Avian Influenza Sequence Classification, Mutation Alignment, and Pathogenicity Prediction.**

---

## 📌 Executive Summary & Abstract

Avian influenza viruses (e.g., H5N1, H5N8, H7N9) present significant global pandemic risks, requiring fast and reliable bioinformatic surveillance. Traditional wet-lab serological and in-vivo pathogenicity assays are costly and time-consuming. 

The **Avian Influenza Genomic Intelligence System** introduces a hybrid deep learning architecture (**`GenomicClassificationEngine`**, **10.21 Million Parameters**) that integrates:
1. **DNABERT-2 Extractor (4.41M Params)**: Captures localized $k$-mer token patterns ($k=6$, vocabulary size 4,096).
2. **HyenaDNA Extractor (2.74M Params)**: Implicit sub-quadratic long-convolution operators to model long-range structural context.
3. **Cross-Attention Feature Fusion (2.93M Params)**: Dynamically aligns pairwise mutation matrices against multi-modal sequence representations.
4. **MLP Classifier Head (0.13M Params)**: Predicts phenotypes across 4 risk categories (*Benign*, *Likely Benign*, *Likely Pathogenic*, *Pathogenic*).

Tested on a hold-out test set of **6,691 FASTA sequences** (from an active balanced dataset of **35,211 sequence records**), the model achieves **86.03% Accuracy**, **89.47% ROC-AUC**, **84.89% Weighted F1**, and **0.7821 Matthew's Correlation Coefficient (MCC)**.

---

## 🏗️ System Pipeline Architecture

```text
                               ┌─────────────────────────────────────────┐
                               │     Raw Genomic FASTA Surveillance      │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │   Quality Control & Length Scrubbing    │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │     Pairwise Alignment & Mutation QC    │
                               └────────────────────┬────────────────────┘
                                                    │
                                         ┌──────────┴──────────┐
                                         │                     │
                                         ▼                     ▼
┌──────────────────────────────────────────┐                 ┌──────────────────────────────────────────┐
│      DNABERT-2 Extractor (4.41M)         │                 │       HyenaDNA Extractor (2.74M)         │
│  - k-mer Hexamer Tokenization (k=6)      │                 │  - Implicit Long Convolutions (Order 64) │
│  - 4-Layer Self-Attention Transformer    │                 │  - 4 Stacked Hyena Operator Blocks       │
└────────────────────┬─────────────────────┘                 └────────────────────┬─────────────────────┘
                     │                                                            │
                     └──────────────────────────┬─────────────────────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────────────┐
                               │   Cross-Attention Feature Fusion (2.93M)│
                               │  Queries = Mutation Tensors [B, 32, 64] │
                               │  Keys/Vals = Sequence Context [B, L, 512]│
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │    MLP Classifier & Report Engine       │
                               │  Outputs 4 Logits + Softmax Probabilities│
                               └─────────────────────────────────────────┘
```

---

## 📊 Benchmark Evaluation Metrics

| Metric | Hold-Out Test Benchmark Score |
|---|:---:|
| **Classification Accuracy** | **86.03%** |
| **ROC-AUC (Macro Average)** | **89.47%** |
| **Weighted F1-Score** | **84.89%** |
| **Macro F1-Score** | **76.31%** |
| **PR-AUC (Precision-Recall)** | **81.44%** |
| **Balanced Accuracy** | **75.29%** |
| **Matthew's Correlation Coefficient (MCC)** | **0.7821** |
| **Cohen's Kappa ($\kappa$)** | **0.7795** |
| **Inference Latency** | **19.82 ms / sequence** |

---

## 🧮 Model Parameter Distribution (10.21M Total)

| Sub-Module | Layer Purpose | Trainable Parameters | % Contribution |
|---|---|:---:|:---:|
| **`dnabert2`** | Token Embedding & 4x Transformer Encoder Blocks | **4,406,528** | **43.18%** |
| **`hyenadna`** | Nucleotide Embedding & 4x Hyena Long-Conv Blocks | **2,737,920** | **26.83%** |
| **`fusion`** | Feature Projections & 8-Head Cross-Attention Engine | **2,928,128** | **28.69%** |
| **`classifier`** | MLP Dense Layers & 4-Class Logit Head | **132,868** | **1.30%** |
| **GRAND TOTAL** | **GenomicClassificationEngine** | **10,205,444** | **100.00%** |

---

## 📁 Repository Directory Structure

```text
DNA_AvianFluenza/
├── configs/
│   └── train_config.yaml         # Central YAML configuration (hyperparameters, paths)
├── Genomic_Framework/            # Core Python Deep Learning Package
│   ├── engine.py                 # Master GenomicClassificationEngine orchestrator
│   ├── main.py                   # CLI entrypoint for pipeline invocation
│   ├── inference/                # Evaluation & advanced reporting module
│   │   └── advanced_evaluator.py
│   ├── models/                   # Neural network sub-modules
│   │   ├── classifier.py         # Module 11: MLP Classifier Head
│   │   ├── dnabert2.py           # Module 4: DNABERT-2 Extractor
│   │   ├── fusion.py             # Module 10: Cross-Attention Fusion
│   │   └── hyenadna.py           # Module 5: HyenaDNA Extractor
│   ├── pipeline/                 # Data ingester, QC, alignment, mutation parser
│   │   ├── input_qc.py
│   │   ├── mutation_pipeline.py
│   │   └── preprocessor.py
│   └── training/                 # PyTorch trainers & optimization hooks
│       ├── advanced_trainer.py
│       └── trainer.py
├── docs/                         # Technical Architecture Specifications
│   ├── pipeline_modules_and_data_flow.md
│   └── software_design_document.md
├── tests/                        # Automated Unit & Integration Tests
│   ├── test_advanced_pipeline.py
│   ├── test_augmentor.py
│   ├── test_healthy_downloader.py
│   └── test_training_modules.py
├── website/                      # Next.js 16 Production Research Platform
│   ├── app/                      # 17 Static pre-rendered routes
│   └── components/
├── augment.py                    # Standalone biological data augmentor CLI
├── data_augmentation_pipeline.py  # Hexamer mutation & sequence generation engine
├── download_healthy.py           # Healthy reference genome fetcher
├── evaluate_advanced.py          # Standalone evaluation & plot generator CLI
├── generate_dataset_statistics.py# Bioinformatic dataset auditor & JSON generator
├── hyperparameter_search.py      # Optuna-style search runner
├── ncbi_downloader.py            # NCBI Entrez viral sequence downloader
├── train_advanced.py             # Production training launcher CLI
├── requirements.txt              # Pinned Python package dependencies
├── LICENSE                       # MIT Open-Source License
└── README.md                     # Main Project Documentation
```

---

## ⚡ Quick Start & Setup Guide

### 1. Prerequisites & Environment Setup

Ensure Python **3.11+** is installed on your system.

```bash
# Clone the repository
git clone https://github.com/vishnuvarthansenthilkumar/DNA_AvianFluenza.git
cd DNA_AvianFluenza

# Create virtual environment
python -m venv .venv

# Activate environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Dataset Acquisition & Setup

> [!NOTE]
> To keep this repository lightweight and suitable for GitHub hosting (avoiding multi-gigabyte binary sequence transfers), raw sequence datasets and model checkpoints are excluded from Git via `.gitignore`. You can automatically acquire and generate the complete dataset using the provided bioinformatic tools.

#### Step 2.1: Download Raw Sequences
Run the automated downloader scripts to fetch viral influenza sequences from NCBI Entrez and healthy host reference genomes:

```bash
# 1. Download viral FASTA sequences from NCBI Entrez into dataset/raw
python ncbi_downloader.py

# 2. Download healthy host reference genomes into dataset/healthy
python download_healthy.py
```

#### Step 2.2: Biological Data Augmentation & Dataset Balancing
Run the augmentation pipeline to produce the balanced sequence partitions:

```bash
# Performs hexamer mutation & sequence balancing -> outputs to dataset_augmented/balanced
python augment.py
```

#### Step 2.3: Verify Bioinformatic Dataset Statistics
Scan all FASTA header records (`>`) to generate dataset statistics JSON:

```bash
python generate_dataset_statistics.py
```

---

### 3. Model Training & Local Checkpoint Generation

Train the **10.21M parameter model** on the complete sequence dataset:

```bash
# Start advanced training using train_config.yaml
python train_advanced.py --config configs/train_config.yaml
```

*Training Safety Guards Enabled:*
- **NaN Loss Detection**: Automatically skips steps if corrupt loss occurs.
- **Gradient Explosion Protection**: Automatic mixed precision (AMP) + gradient clipping ($L_2 \le 1.0$).
- **Dynamic LR Scheduler**: Linear warmup (3 epochs) followed by Cosine Annealing.
- **Checkpoints**: Saved locally to `output_advanced/checkpoints/best_model.pt`.

---

### 4. Model Evaluation & Benchmark Report Generation

Run evaluation on the hold-out test dataset:

```bash
python evaluate_advanced.py \
  --checkpoint output_advanced/checkpoints/best_model.pt \
  --dataset-root dataset_augmented \
  --output-dir eval_results_fixed \
  --config configs/train_config.yaml \
  --batch-size 16
```

This generates:
- `metrics/evaluation_metrics.json`
- `metrics/classification_report.csv`
- `plots/confusion_matrix.png`, `roc_curves.png`, `precision_recall_curves.png`
- `reports/evaluation_report.md` & `evaluation_report.html`

---

### 5. Running Automated Unit Tests

Run the complete test suite:

```bash
python -m unittest discover -s tests -p "*.py"
```

---

### 6. Launching the Next.js Research Web Platform

The repository includes a production-grade **Next.js 16 App Router** research web app:

```bash
cd website

# Install Node dependencies
npm install

# Start development server (http://localhost:3000)
npm run dev

# Or build static production bundle
npm run build
```

---

## 📖 Citation

If you use this framework or model architecture in your research, please cite:

```bibtex
@article{senthilkumar2026avian,
  title={Avian Influenza Genomic Intelligence: Hybrid Deep Learning Framework for Pathogenicity Classification and Mutation Alignment},
  author={Senthilkumar, Vishnu Varthan},
  journal={IEEE Transactions on Artificial Intelligence / Bioinformatics},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
