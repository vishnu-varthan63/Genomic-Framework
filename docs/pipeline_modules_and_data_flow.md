# Genomic Sequence Classification Framework
## Complete Module Work Summaries, End-to-End Data Travel Journey, and Parameter Matrix

---

## Part 1: One-Line Work Summary for Every Module (Start to End)

| Module # | Module Class / Component Name | One-Line Work Summary |
| :---: | :--- | :--- |
| **Module 1** | `GenomicSequenceIngester` | Ingests raw FASTA files and enforces biological quality control (length 100–1500bp, GC content 38%–62%, N-fraction $\le 5\%$, homopolymers $\le 20$, and stop codons). |
| **Module 2** | `GenomicTokenizer` | Decomposes clean nucleotide strings into overlapping 6-mer k-mer sequences and encodes them into fixed-length integer token ID vectors (`[B, 128]`). |
| **Module 3** | `SequenceAligner` | Performs semi-global Needleman-Wunsch alignment between query sequence and subtype reference sequence to extract positional mutation records. |
| **Module 4** | `DNABERT2Extractor` | Extracts local k-mer semantic context representations using 4-layer multi-head self-attention transformer blocks (`[B, 128, 256]`). |
| **Module 5** | `HyenaDNAExtractor` | Extracts ultra-long-range context representations via 4-layer sub-quadratic implicit long convolutions (`[B, 128, 256]`). |
| **Module 6** | `DataAugmentationPipeline` | Expands dataset volume by $>10\times$ and balances classes using a 5-tier biological pipeline (host adaptation, reassortment, silent mutations, SMOTE, reverse complement, sliding windows, and noise). |
| **Module 7** | `GenomicDatasetBuilder` | Discovers FASTA dataset files, calculates inverse-frequency class weights, splits data (70% train / 15% val / 15% test), and constructs PyTorch DataLoaders. |
| **Module 8** | `AdvancedTrainer` | Executes multi-epoch model training using PyTorch AMP mixed precision, AdamW optimizer, Cosine Annealing scheduler, gradient accumulation, and automatic best-model selection. |
| **Module 9** | `AdvancedEvaluator` | Executes zero-gradient batch inference, computes 8 research metrics (Accuracy, F1, ROC-AUC, PR-AUC, MCC, Kappa), and exports predictions to CSV, JSON, and Excel. |
| **Module 10** | `MutationProcessor` | Encodes Needleman-Wunsch substitution, insertion, and deletion positional alignment records into fixed 3D mutation tensors (`[B, 32, 64]`). |
| **Module 11** | `GenomicClassificationEngine` & `MLPClassifier` | Fuses local and long-range representations via Cross-Attention (`[B, 512]`), concatenates mutation vectors (`[B, 576]`), and predicts pathogenicity logits (`[B, 4]`). |
| **Module 12** | `ExplainabilityEngine` | Computes Integrated Gradients attribution maps and perturbation saliency scores to highlight nucleotide positions driving pathogenicity predictions. |
| **Module 13** | `GenomicReportGenerator` | Renders comprehensive executive training and evaluation reports in Markdown and styled HTML formats with embedded visualization plots. |

---

## Part 2: Step-by-Step End-to-End Data Travel Journey (Input FASTA to Final Prediction)

```text
[Input FASTA File / Nucleotide String]
                 │
                 ▼
 ┌─────────────────────────────────────────────────────────┐
 │ STEP 1: Quality Control Filtering                       │
 │ Input : Raw Nucleotide Sequence (e.g. "ATGCGATCG...")   │
 │ Module: Module 1 (GenomicSequenceIngester)              │
 │ Checks: Length [100, 1500], GC [0.38, 0.62], N-frac, Stop│
 └─────────────────────────┬───────────────────────────────┘
                           │ Valid Sequence String (Length L)
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ STEP 2: Tokenization & Sequence Alignment               │
 │ Input : Valid Nucleotide String                         │
 │ Module: Module 2 (Tokenizer) & Module 3 (Aligner)       │
 │ Output: Token ID Vector [B, 128]                        │
 │         Alignment Mutation Records                      │
 └─────────────┬─────────────────────────────┬─────────────┘
               │ Token IDs [B, 128]          │ Mutation Records
               ▼                             ▼
 ┌───────────────────────────┐ ┌───────────────────────────┐
 │ STEP 3A: Local Extraction │ │ STEP 3B: Long Extraction  │
 │ Module: Module 4          │ │ Module: Module 5          │
 │         (DNABERT-2)       │ │         (HyenaDNA)        │
 │ Output: Tensor            │ │ Output: Tensor            │
 │         [B, 128, 256]     │ │         [B, 128, 256]     │
 └─────────────┬─────────────┘ └─────────────┬─────────────┘
               │                             │
               └──────────────┬──────────────┘
                              │ Dual Feature Tensors [B, 128, 256]
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │ STEP 4: Cross-Attention Feature Fusion                  │
 │ Input : Local Tensor [B, 128, 256] + Long [B, 128, 256] │
 │ Module: Module 11 (CrossAttentionFusion)                │
 │ Process: Bi-directional Cross-Attn -> LayerNorm -> Pool │
 │ Output: Fused Feature Vector [B, 512]                   │
 └─────────────────────────┬───────────────────────────────┘
                           │ Fused Vector [B, 512]
                           │
 ┌─────────────────────────┴───────────────────────────────┐
 │ STEP 5: Mutation Tensor Processing                      │
 │ Input : Needleman-Wunsch Alignment Mutation Records     │
 │ Module: Module 10 (MutationProcessor)                   │
 │ Output: Mutation Matrix [B, 32, 64] -> Vector [B, 64]   │
 └─────────────────────────┬───────────────────────────────┘
                           │ Mutation Vector [B, 64]
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ STEP 6: Feature Concatenation & MLP Classification      │
 │ Input : Fused Vector [B, 512] + Mutation Vector [B, 64] │
 │ Tensor: Concatenated Feature Vector [B, 576]            │
 │ Module: Module 11 (MLPClassifier)                       │
 │ Layers: Linear(576->256) -> GELU -> Dropout -> Linear(4)│
 │ Output: Raw Pathogenicity Logits z [B, 4]               │
 └─────────────────────────┬───────────────────────────────┘
                           │ Logits [B, 4]
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ STEP 7: Softmax Probabilities & Confidence Scoring     │
 │ Formula: P(Y=c) = Softmax(z_c)                         │
 │ Output : Probabilities [prob_Benign, prob_Likely_Benign,│
 │          prob_Likely_Pathogenic, prob_Pathogenic]       │
 │          Predicted Class & Confidence Score             │
 └─────────────┬─────────────────────────────┬─────────────┘
               │                             │
               ▼                             ▼
 ┌───────────────────────────┐ ┌───────────────────────────┐
 │ STEP 8A: Explainability   │ │ STEP 8B: Testing Export   │
 │ Module: Module 12         │ │ Module: Module 9 & 13     │
 │         (Integrated Grad) │ │ Output: CSV, JSON, Excel, │
 │ Output: Saliency Map      │ │         ROC/PR PNG Plots, │
 │         Nucleotide Heatmap│ │         HTML & MD Reports │
 └───────────────────────────┘ └───────────────────────────┘
```

---

## Part 3: Complete Parameters & Tensor Transformation Matrix

### 3.1 Network Layer Parameters & Model Dimensions

| Stage / Component | Class / Module Name | Parameter Name | Value / Dimension | Description |
| :--- | :--- | :--- | :---: | :--- |
| **Input Sequence** | `GenomicSequenceIngester` | `seq_len` ($L$) | **128** | Sequence length constraint in base pairs |
| **k-mer Tokenizer** | `GenomicTokenizer` | `k_mer_size` ($k$) | **6** | Overlapping 6-mer window size |
| **Vocabulary Size** | `GenomicTokenizer` | `vocab_size` ($|V|$) | **4096** | $|V| = 4^6 = 4096$ unique 6-mers + special tokens |
| **DNABERT-2 Embedding** | `DNABERT2Extractor` | `embed_dim` | **256** | Input token embedding dimension |
| **DNABERT-2 Projection** | `DNABERT2Extractor` | `output_dim` | **768** | Internal transformer layer projection |
| **DNABERT-2 Layers** | `DNABERT2Extractor` | `num_layers` | **4** | Number of transformer self-attention blocks |
| **DNABERT-2 Heads** | `DNABERT2Extractor` | `num_heads` | **8** | Multi-head attention count per block |
| **DNABERT-2 Params** | `DNABERT2Extractor` | `total_params` | **4.41M** | Total parameters in local extractor branch |
| **HyenaDNA Model Dim** | `HyenaDNAExtractor` | `d_model` | **256** | Hidden state dimension of long-range encoder |
| **HyenaDNA Order** | `HyenaDNAExtractor` | `filter_order` | **64** | Order of implicit long convolution filter |
| **HyenaDNA Layers** | `HyenaDNAExtractor` | `num_layers` | **4** | Number of Hyena recurrence blocks |
| **HyenaDNA Params** | `HyenaDNAExtractor` | `total_params` | **2.74M** | Total parameters in long-range extractor branch |
| **Cross-Attn Heads** | `CrossAttentionFusion` | `num_heads` | **8** | Bi-directional cross-attention heads |
| **Cross-Attn Dim** | `CrossAttentionFusion` | `hidden_dim` | **512** | Fused cross-attention output dimension |
| **Cross-Attn Params** | `CrossAttentionFusion` | `total_params` | **2.93M** | Total parameters in cross-attention fusion module |
| **Mutation Tensor** | `MutationProcessor` | `shape` | `[B, 32, 64]` | 32 max mutations $\times$ 64-dim mutation vector |
| **Mutation Summary** | `MutationProcessor` | `flat_dim` | **64** | Mean-pooled positional mutation vector |
| **MLP Input Vector** | `MLPClassifier` | `input_dim` | **576** | $512 \text{ (Fused Features)} + 64 \text{ (Mutation Vector)}$ |
| **MLP Intermediate** | `MLPClassifier` | `hidden_dim` | **256** | GELU intermediate hidden representation |
| **MLP Output Classes** | `MLPClassifier` | `num_classes` | **4** | Benign (0), Likely Benign (1), Likely Pathogenic (2), Pathogenic (3) |
| **MLP Params** | `MLPClassifier` | `total_params` | **132,868** | Parameters in final classification head |
| **TOTAL MODEL ENGINE** | `GenomicClassificationEngine` | **Total Parameters** | **10.21M** | **Complete Network Model Size (10,210,500 parameters)** |

---

### 3.2 Tensor Shape Transformation Flow

| Step | Tensor Description | Tensor Shape | Output Memory Type |
| :---: | :--- | :---: | :---: |
| **1** | Raw Batch Token IDs | `[B, 128]` | Integer LongTensor |
| **2** | DNABERT-2 Output Representation | `[B, 128, 256]` | Float32 Tensor |
| **3** | HyenaDNA Output Representation | `[B, 128, 256]` | Float32 Tensor |
| **4** | Cross-Attention Bi-directional Fusion | `[B, 128, 512]` | Float32 Tensor |
| **5** | Fused Pooled Feature Vector | `[B, 512]` | Float32 Tensor |
| **6** | Needleman-Wunsch Mutation Array | `[B, 32, 64]` | Float32 Tensor |
| **7** | Flattened Mutation Vector | `[B, 64]` | Float32 Tensor |
| **8** | Combined Classifier Input Vector | `[B, 576]` | Float32 Tensor |
| **9** | MLP Hidden Activation Vector | `[B, 256]` | Float32 Tensor |
| **10** | Pathogenicity Logits Vector ($z$) | `[B, 4]` | Float32 Tensor |
| **11** | Softmax Class Probabilities ($P$) | `[B, 4]` | Float32 Probability Vector |
