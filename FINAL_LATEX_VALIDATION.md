# Final IEEE LaTeX Pre-Submission Validation Report

**Document**: *Avian Influenza Genomic Intelligence: A Hybrid Deep Learning Architecture for Pathogenicity Category Prediction and Mutation-Aware Sequence Modeling*  
**Format**: IEEE Conference Proceedings (`\documentclass[conference]{IEEEtran}`)  
**Validation Suite**: `scripts/check_ieee_format.py` & `validate_paper.py`  
**Status**: **100% Passed (0 Errors, 0 Warnings)**  

---

## 1. Compliance Checklist

| Item | Requirement | Verification Result | Status |
|---|---|---|:---:|
| **Document Class** | `\documentclass[conference]{IEEEtran}` | Clean IEEEtran configuration | **PASS** |
| **Margin Override Check** | No `geometry`, `fullpage`, or manual `\textwidth` overrides | Native IEEEtran margin management | **PASS** |
| **Two-Column Balance** | Symmetric left/right margins ($0.625\text{ in}$) and column separation ($1\text{ pc}$) | Natural, balanced column layout | **PASS** |
| **Table Layout** | All 10 tables fit strictly within `\columnwidth` or `\textwidth` | 10/10 tables converted to `tabularx` | **PASS** |
| **Figure Clarity** | High-resolution, readable font size, restrained color palette | All 7 figures re-rendered at 300 DPI | **PASS** |
| **Equation Formatting** | Aligned multiline equations, standard sequential numbering (1)-(29) | No overflow; all equations numbered sequentially | **PASS** |
| **Reference Integrity** | Zero undefined `\ref`, `\eqref`, or `\cite` calls | 56 labels resolved; 28 BibTeX citations mapped | **PASS** |
| **Scientific Consistency** | Exact preservation of all parameters, sequence counts, and metrics | 35,211 sequences; 10.30M params; 86.03% acc | **PASS** |

---

## 2. Verified Mathematical Equations Inventory

- $\text{Eq.}~\eqref{eq:qc_fraction}$: Valid nucleotide fraction $\gamma$
- $\text{Eq.}~\eqref{eq:pad_truncate}$: Sequence length standardization ($L=128$)
- $\text{Eq.}~\eqref{eq:mismatch_set}$: Mismatch loci set $\mathcal{M}$
- $\text{Eq.}~\eqref{eq:edit_distance}$: Hamming-style mismatch count $D_E(S,R)$
- $\text{Eq.}~\eqref{eq:kmer}$: Hexamer sliding window ($123$ valid $k$-mers)
- $\text{Eq.}~\eqref{eq:token_hash}$: Deterministic token hash modulo $V=4,096$
- $\text{Eq.}~\eqref{eq:dnabert_emb}$: Dense embedding with sinusoidal PE
- $\text{Eq.}~\eqref{eq:pe_sin}-\eqref{eq:pe_cos}$: Sinusoidal positional encodings
- $\text{Eq.}~\eqref{eq:mha_step}-\eqref{eq:ffn_step}$: Multi-head self-attention and FFN transformations
- $\text{Eq.}~\eqref{eq:dna_out}$: DNABERT-2-inspired output projection $\mathbf{Z}_{\text{DNA}} \in \mathbb{R}^{L \times 768}$
- $\text{Eq.}~\eqref{eq:hyena_input}$: Hyena input embedding with relative positional bias
- $\text{Eq.}~\eqref{eq:short_conv}-\eqref{eq:hyena_gate}$: Short convolution, long causal filter (Order 64), and multiplicative gating
- $\text{Eq.}~\eqref{eq:hyena_block_out}$: Hyena block output $\mathbf{Z}_{\text{Hyena}} \in \mathbb{R}^{L \times 256}$
- $\text{Eq.}~\eqref{eq:mutation_score}$: Mutation importance score with Gaussian spatial decay
- $\text{Eq.}~\eqref{eq:mut_encoding}$: Dense 64-dimensional mutation vector encoding (multiline aligned)
- $\text{Eq.}~\eqref{eq:proj_dna}-\eqref{eq:proj_mut}$: Sequence and mutation query latent projections ($d_h=512$)
- $\text{Eq.}~\eqref{eq:cross_attn}$: 8-head cross-attention mechanism
- $\text{Eq.}~\eqref{eq:global_pool}$: Global average pooling over mutation tokens
- $\text{Eq.}~\eqref{eq:mlp_h1}-\eqref{eq:logits}$: MLP classification head mapping to logits $\mathbf{z} \in \mathbb{R}^4$
- $\text{Eq.}~\eqref{eq:softmax}$: Softmax class probability computation
- $\text{Eq.}~\eqref{eq:loss}$: Class-weighted cross-entropy loss function
- $\text{Eq.}~\eqref{eq:shap_grad}$: Gradient-input product attribution score $\phi_k$

---

## 3. Verified Experimental Metrics Preserved

- **Dataset**: 35,211 curated sequences across 72 files (Train: 23,520; Val: 5,000; Test: 6,691).
- **Model Parameters**: Exactly 10,303,748 trainable parameters.
- **Hold-Out Test Performance**:
  - Test Accuracy: **86.03%**
  - Macro ROC-AUC: **89.47%**
  - Weighted F1-Score: **84.89%**
  - Macro F1-Score: **76.31%**
  - Matthews Correlation Coefficient (MCC): **0.7821**
  - Cohen's Kappa ($\kappa$): **0.7795**
  - Average Inference Latency: **19.82 ms / sequence**
