# Model Architecture & Checkpoint Audit

**Project**: Avian Influenza Genomic Intelligence Framework  
**Audit Date**: October 15, 2026  
**Auditor**: Lead System Reconstruction Architect  

---

## 1. Verified Model Parameter Profile

Direct computational profiling of each constituent module within `GenomicClassificationEngine` (`Genomic_Framework/engine.py`) confirms the exact layer-by-layer parameter counts:

| Module Identifier | Component Description | Sub-Components | Trainable Parameters | Parameter Share (%) |
|---|---|---|:---:|:---:|
| `dnabert2` | DNABERT-2-Inspired Transformer Extractor | Token Embedding ($V=4096, d=256$), 4x MHA Layers (8 heads, $d_{ffn}=1024$), Output Linear Projection ($256 \to 768$) | **4,406,528** | 42.77% |
| `hyenadna` | HyenaDNA-Inspired Long-Convolution Extractor | Token Embedding ($V=4096, d=256$), 4x Hyena Operator Blocks (Conv3, Order-64 Causal Filter, Gating) | **2,836,224** | 27.53% |
| `fusion` | Cross-Attention Multi-Modal Fusion Engine | Context Projections ($768 \to 512, 256 \to 512$), Mutation Query Projection ($64 \to 512$), 8-Head Cross-Attention, Pooling | **2,928,128** | 28.42% |
| `classifier` | MLP Classification Head | Linear ($512 \to 256$), LayerNorm, GELU, Dropout (0.3), Linear ($256 \to 4$) | **132,868** | 1.29% |
| **Total** | **GenomicClassificationEngine** | **Complete Integrated Architecture** | **10,303,748** | **100.00%** |

---

## 2. Model Checkpoint Inventory

### A. Recovered Checkpoints (Verified on Disk)
| File Path | File Size | Date Created | Architecture | Training Details | Status |
|---|:---:|:---:|---|---|:---:|
| `outputs/model_demo.pt` | 634,485 bytes | 2026-08-25 22:39 | `demo.models.demo_model.DemoModel` | 5 epochs on `demo/data/demo_sequences.fasta` via `demo/train_demo.py` | `VERIFIED_WORKING` |

### B. Full Research Model Checkpoint Status
- **Historical Location**: `checkpoints_augmented/best_model.pt` (historical size ~122.8 MB).
- **Physical Presence**: **Not present in recovered repository files**.
- **Root Cause of Absence**: Standard Git hygiene practice. `.gitignore` explicitly excludes `checkpoints/`, `checkpoints_*/`, and `*.pt` files to avoid exceeding GitHub repository size quotas.
- **Reproducibility Assessment**:
  - The model code (`Genomic_Framework/models/`) is 100% recovered and compiles without error.
  - The active dataset (`dataset_augmented/` with 35,211 sequences) is 100% recovered on disk.
  - The exact training configuration (`configs/train_config.yaml`) and random seed (`seed = 42`) are fully preserved.
  - The full model can be reproducibly retrained at any time using:
    ```bash
    python train_advanced.py --config configs/train_config.yaml
    ```
- **Integrity Rule**: We explicitly acknowledge that the 122.8 MB checkpoint is absent from disk rather than inventing or substituting an untrained dummy model as the full trained checkpoint.

---

## 3. Demo Model Checkpoint Verification
To verify the integrity of `outputs/model_demo.pt`, the checkpoint was inspected for tensor state keys:
- State dict keys include: `sequence_encoder`, `mutation_encoder`, `cross_attention`, `classifier`.
- Model loads into `demo/models/demo_model.py` and successfully runs inference via `python -m demo.inference_demo`.
- The demo model is kept strictly distinct from the full 10.30M research model.
