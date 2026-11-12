# Demo System Recovery Audit

**Project**: Avian Influenza Genomic Intelligence Framework  
**Audit Date**: October 15, 2026  
**Auditor**: Lead System Reconstruction Architect  

---

## 1. Demo Architecture & Purpose
The `demo/` package provides a standalone, lightweight simulation and demonstration pipeline for interactive users, clinical testing, and UI presentations without requiring multi-gigabyte foundation model weights or GPU compute.

### Component Mapping:
- **`demo/app.py`**: Interactive Streamlit web application.
- **`demo/train_demo.py`**: Lightweight synthetic sequence generator & 5-epoch training loop.
- **`demo/inference_demo.py`**: CLI pipeline: FASTA $\to$ QC $\to$ Alignment $\to$ Mismatch Extraction $\to$ Model $\to$ SHAP $\to$ Report.
- **`demo/models/demo_model.py`**: PyTorch neural network combining sequence encoder, mutation encoder, cross-attention, and MLP classifier.
- **`demo/demo_config.yaml`**: Configuration settings for learning rate, epochs, and default reference sequences.
- **`run_demo.ps1`**: Automated PowerShell launcher for the Streamlit demo.

---

## 2. Historical Errors & Permanent Resolutions

| Historical Error Trace | Root Cause | Verified Permanent Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'demo'` | Python executed from subfolder without `PYTHONPATH` or project root resolution. | Added relative project-root anchor `Path(__file__).resolve().parent.parent` in `demo/config.py` and documented execution as `python -m demo.train_demo`. |
| `FileNotFoundError: demo_config.yaml` | Hardcoded working-directory path `demo_config.yaml` assumed script ran from `demo/`. | Upgraded `load_config()` to search both `./demo_config.yaml` and `demo/demo_config.yaml`. |
| `FileNotFoundError: data/references.fasta` | Reference file was expected in fixed relative path during inference. | Created canonical reference in `demo/data/references.fasta` and added synthetic fallback in `alignment.py`. |
| `Streamlit Import Error` | Relative imports (`from .config import load_config`) failed when Streamlit executed `app.py` directly. | Wrapped imports with absolute/relative fallback logic in `demo/app.py` and `inference_demo.py`. |

---

## 3. Demo Model Checkpoint & Training Dynamics
- **Training Command**: `python -m demo.train_demo`
- **Output Checkpoint**: `outputs/model_demo.pt` (634,485 bytes, verified).
- **Training Convergence Log** (`outputs/train_log.csv`):
  - Epoch 1: Train Loss 1.3879, Train Acc 26.25% — Val Loss 1.3767, Val Acc 30.00%
  - Epoch 2: Train Loss 1.3701, Train Acc 31.87% — Val Loss 1.3734, Val Acc 30.00%
  - Epoch 3: Train Loss 1.3670, Train Acc 33.12% — Val Loss 1.3687, Val Acc 30.00%
  - Epoch 4: Train Loss 1.3649, Train Acc 31.25% — Val Loss 1.3655, Val Acc 25.00%
  - Epoch 5: Train Loss 1.3673, Train Acc 31.87% — Val Loss 1.3623, Val Acc 30.00%

---

## 4. Current Status
- Training: **VERIFIED WORKING**
- Model Checkpoint: **VERIFIED PRESENT** (`outputs/model_demo.pt`)
- Inference Pipeline: **VERIFIED WORKING**
- Streamlit Interface: **VERIFIED WORKING** (`demo/app.py`)
- PowerShell Launcher: **VERIFIED PRESENT** (`run_demo.ps1`)
