# Avian Influenza Genomic Intelligence Framework: Complete System Documentation & Developer Guide

**Document Version**: 2.0.0 (Forensic Restoration & Production Edition)  
**Author / Principal Investigator**: Vishnu Varthan Senthilkumar  
**Repository**: `https://github.com/vishnu-varthan63/Genomic-Framework.git`  
**Target Audience**: Bioinformaticians, Machine Learning Engineers, MLOps Engineers, and Computational Biology Researchers  

---

## 1. System Architecture & High-Level Design

The Avian Influenza Genomic Intelligence Framework is an end-to-end biological deep learning system designed for the automated classification of Avian Influenza A Virus (AIV) genome sequences into four clinically and epidemiologically curated pathogenicity categories:

```
[FASTA/FASTQ Input Sequence]
             │
             ▼
[Module 1-2: QC & Standardization]
  • Scrub non-canonical IUPAC characters to {A, C, G, T, N}
  • Filter sequences with valid fraction γ < 0.50
  • Pad or truncate to fixed uniform length L = 128
             │
             ├───────────────────────────────────┬──────────────────────────────────┐
             ▼                                   ▼                                  ▼
[Module 4: DNABERT-2-Inspired]        [Module 5: HyenaDNA-Inspired]        [Module 3, 6-9: Mutation Engine]
  • Sliding hexamer k-mer (k=6)         • Single-token embeddings            • Pairwise comparator vs. reference
  • 4x Transformer Encoder layers       • 4x Hyena Operator blocks           • Identify mismatch loci M
  • Multi-head self-attention (8 heads) • Depthwise Conv3 + LongConv64       • Functional region annotation
  • Sinusoidal positional embeddings    • Learnable relative bias            • Gaussian spatial decay scoring
  • Output: [B, 128, 768] (4.41M params)• Output: [B, 128, 256] (2.84M params)• Output: [B, 32, 64] tensor
             │                                   │                                  │
             └───────────────────┬───────────────┘                                  │
                                 ▼                                                  ▼
                   [Joint Sequence Context C]                             [Mutation Query Tensor Q]
                   Proj(DNA) + Proj(Hyena)                                Linear(64 -> 512)
                   C ∈ R^{B x 128 x 512}                                  Q ∈ R^{B x 32 x 512}
                                 │                                                  │
                                 └───────────────────────┬──────────────────────────┘
                                                         ▼
                                       [Module 10: Cross-Attention Fusion Engine]
                                         • Queries: Mutation Tokens Q [B, 32, 512]
                                         • Keys & Values: Context C [B, 128, 512]
                                         • 8-Head Multi-Head Cross-Attention
                                         • Residual + LayerNorm + FFN
                                         • Global Average Pooling across 32 mutation tokens
                                         • Output: Fused representation f ∈ R^{B x 512} (2.93M params)
                                                         │
                                                         ▼
                                       [Module 11: MLP Classification Head]
                                         • Linear(512 -> 256) + LayerNorm + GELU + Dropout(0.3)
                                         • Linear(256 -> 4) -> Logits z ∈ R^{B x 4}
                                         • Softmax -> Probabilities over 4 Pathogenicity Categories
                                                         │
                                                         ▼
                                       [Module 12-13: Explainability & Reports]
                                         • Gradient-Input Product Attribution (Saliency Ranking)
                                         • Automated Multi-Format Reporting (JSON / CSV / HTML / Markdown)
```

---

## 2. Pathogenicity Classification Taxonomy

The model maps sequences to four discrete categories based on curated viral subtype virulence and host origin:

| Class Index | Pathogenicity Class | Definition & Biological Grounding | Associated Strains / Host Origins |
|:---:|---|---|---|
| **0** | **Avirulent** | Non-viral healthy host avian genomic sequences serving as negative control baselines. | Gallus gallus (chicken), Anas platyrhynchos (duck), Anser cygnoides (goose), quail, turkey. |
| **1** | **Low Pathogenicity** | Low-Pathogenic Avian Influenza (LPAI) strains. Typically cause asymptomatic or mild infection in wild waterfowl. | H5N2, H5N3, H7N2, H7N3, H9N2. |
| **2** | **Moderate-High Pathogenicity** | Intermediate or emerging avian influenza lineages possessing virulence markers or mammalian transmission potential. | H5N6, H5N8, H7N1, H7N7, H10N7, H10N8. |
| **3** | **Highly Pathogenic** | Highly Pathogenic Avian Influenza (HPAI) clades causing severe systemic disease, flock mortality, and human zoonotic fatalities. | H5N1, H5N9, H7N9. |

---

## 3. Detailed Component Specifications

### 3.1. Quality Control & Bioinformatic Preprocessor
- **Source**: `Genomic_Framework/pipeline/input_qc.py`, `preprocessor.py`
- **Scrubbing**: Restricts input alphabets to $\Sigma = \{A, C, G, T, N\}$. Non-canonical characters (e.g., $R, Y, S, W, K, M, B, D, H, V$) are replaced with $N$.
- **Validation**: Rejects sequences where valid canonical nucleotide fraction $\gamma < 0.50$.
- **Standardization**: Enforces uniform sequence length $L = 128$ bases via right-padding with $N$ or centered/head truncation.

### 3.2. Reference Alignment & Mutation Prioritization
- **Source**: `Genomic_Framework/pipeline/mutation_pipeline.py`
- **Mismatch Identification**: Compares query sequence $S$ against wild-type reference $R$ over valid non-$N$ positions:
  $$\mathcal{M} = \{i \in \{1, \dots, L\} \mid S[i] \neq R[i] \land S[i] \neq 'N' \land R[i] \neq 'N'\}$$
- **Regional Heuristics**: Assigns biological priority weights $w(r)$ across 8 functional genomic regions ($w(\text{coding}) = 1.00, w(\text{splice\_site}) = 0.95, \dots, w(\text{non\_coding}) = 0.30$).
- **Spatial Gaussian Decay**: Emphasizes mutations proximate to critical catalytic centers ($\mu_{pos} = 0.3L, \sigma_{pos} = 0.4L$):
  $$s(m) = w(r) \cdot \exp\left(-\frac{1}{2}\left(\frac{p - \mu_{pos}}{\sigma_{pos}}\right)^2\right)$$
- **Encoding**: Ranks variants by $s(m)$ and encodes the top $M = 32$ variants into 64-dimensional feature vectors:
  $$\mathbf{v}_m = \left[\frac{p}{L} \;\parallel\; \mathbf{e}_{ref} \;\parallel\; \mathbf{e}_{query} \;\parallel\; \mathbf{e}_{type} \;\parallel\; \mathbf{e}_{region} \;\parallel\; s(m) \;\parallel\; \frac{\text{rank}}{M} \;\parallel\; \mathbf{0}_{40}\right]$$
  yielding mutation tensor $\mathbf{M} \in \mathbb{R}^{B \times 32 \times 64}$.

### 3.3. Dual Sequence Representation
- **DNABERT-2-Inspired Extractor** (`Genomic_Framework/models/dnabert2.py`):
  - Sliding hexamer window ($k=6, s=1$) producing 123 tokens padded to 128 tokens.
  - Vocabulary size $V = 4,096$ ($4^6$).
  - 4 Transformer layers, 8 attention heads, $d_{model}=256, d_{ffn}=1024$, projected to $\mathbf{Z}_{DNA} \in \mathbb{R}^{B \times 128 \times 768}$.
  - **4,406,528 trainable parameters**.
- **HyenaDNA-Inspired Extractor** (`Genomic_Framework/models/hyenadna.py`):
  - Single-token resolution with learnable relative positional bias.
  - 4 Hyena operator blocks utilizing short Conv3 depthwise convolutions, long causal convolutions of order $K=64$ with exponential decay envelopes, and multiplicative gating.
  - Generates sub-quadratic long-range representation $\mathbf{Z}_{Hyena} \in \mathbb{R}^{B \times 128 \times 256}$.
  - **2,836,224 trainable parameters**.

### 3.4. Multi-Modal Cross-Attention Fusion
- **Source**: `Genomic_Framework/models/fusion.py`
- Sequence streams projected to $d_h = 512$: $\mathbf{C} = \mathrm{Proj}(\mathbf{Z}_{DNA}) + \mathrm{Proj}(\mathbf{Z}_{Hyena}) \in \mathbb{R}^{B \times 128 \times 512}$.
- Mutation tensor projected to Query space: $\mathbf{Q} \in \mathbb{R}^{B \times 32 \times 512}$.
- 8-head multi-head cross-attention allows discrete mutation queries to dynamically extract contextual evidence from joint sequence channels.
- Global average pooling across mutation tokens yields fused representation $\mathbf{f} \in \mathbb{R}^{B \times 512}$.
- **2,928,128 trainable parameters**.

### 3.5. MLP Classifier Head
- **Source**: `Genomic_Framework/models/classifier.py`
- Architecture: $\mathrm{Linear}(512 \to 256) \to \mathrm{LayerNorm} \to \mathrm{GELU} \to \mathrm{Dropout}(0.3) \to \mathrm{Linear}(256 \to 4)$.
- **132,868 trainable parameters**.
- **Total Engine Footprint**: Exactly **10,303,748 trainable parameters**.

---

## 4. Dataset Organization & Augmentation Strategy

### 4.1. Dataset Composition
- **Raw Pool (`dataset/raw/`, `dataset/healthy/`)**: 149 files, 1,048,666 sequences, 4.78 billion base pairs.
- **Augmented Pool (`dataset_augmented/`)**: 540 files, 67,482 sequences, 87.9M base pairs.
- **Curated Experimental Benchmark**: 72 FASTA files, 35,211 sequences, 45,909,430 base pairs.
  - **Train**: 48 files, 23,520 sequences (66.8%)
  - **Validation**: 10 files, 5,000 sequences (14.2%)
  - **Hold-Out Test**: 14 files, 6,691 sequences (19.0%)

### 4.2. Biological Augmentation Strategies (`augment.py`, `data_augmentation_pipeline.py`)
1. **Reverse Complement**: Simulates negative-sense RNA genome strands.
2. **Point Substitutions**: Models viral polymerase transcription error rates ($1.0 \times 10^{-4}$).
3. **Insertions & Deletions (Indels)**: Simulates reading frame and loop perturbations.
4. **Codon Variations**: Synonymous codon wobble substitutions preserving translated amino acids.
5. **Sliding Windows**: Simulates fragmentary next-generation sequencing contigs.

---

## 5. Developer Guide: Commands, Testing & Workflows

### 5.1. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core dependencies
pip install -r requirements.txt
```

### 5.2. Running Automated Tests
```powershell
# Run the full unit and integration test suite
python -m unittest discover -s tests -p "test_*.py" -v
```

### 5.3. Running the Lightweight Interactive Demo
```powershell
# Method 1: Automated PowerShell launcher
powershell -ExecutionPolicy Bypass -File run_demo.ps1

# Method 2: Command-line demo inference
python -m demo.inference_demo --fasta demo/data/demo_sequences.fasta

# Method 3: Streamlit interactive UI
streamlit run demo/app.py
```

### 5.4. Full Model Training & Evaluation
```powershell
# Train the 10.30M parameter model on the 35,211 sequence benchmark
python train_advanced.py --config configs/train_config.yaml

# Evaluate model and generate plots/reports
python evaluate_advanced.py --checkpoint checkpoints_augmented/best_model.pt --dataset-root dataset_augmented --output-dir eval_results
```

### 5.5. Compiling and Verifying the IEEE LaTeX Paper
```powershell
# Validate IEEE table formatting and citation references
python scripts/check_ieee_format.py
python validate_paper.py

# Re-render 300 DPI publication figures
python generate_paper_figures.py
```

### 5.6. Launching the Next.js 16 Web Research Platform
```powershell
cd website
npm install
npm run dev
# Access local research portal at http://localhost:3000
```
