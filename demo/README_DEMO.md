# LIGHTWEIGHT DEMONSTRATION OF GENOMIC AI PIPELINE

This demo showcases the **complete conceptual pipeline** from a DNA sequence input to a final prediction with SHAP explainability, **running entirely on a CPU** and requiring only a few megabytes of memory.

---

## What the Demo Does
- Accepts a FASTA file or a raw DNA string.
- Performs lightweight quality control.
- Aligns the query against a tiny local reference (no external BLAST database required).
- Encodes the sequence with a simple **k‑mer embedding** (k=3).
- Extracts a compact 128‑dimensional feature vector via a tiny 1‑D Conv network.
- Identifies mutations (substitutions/insertions/deletions) by comparing to the reference.
- Provides a placeholder annotation layer ( Demo annotation).
- Scores each mutation with a transparent priority formula.
- Encodes mutation information into a 64‑dimensional vector.
- Fuses sequence and mutation representations using **PyTorch Multi‑head Attention** (4 heads, dim = 64).
- Classifies the fused representation with a small MLP (4 configurable demo classes).
- Generates a SHAP KernelExplainer explanation for the classification.
- Produces a comprehensive JSON/CSV/HTML report.

---

## How This Differs From the Full Research Model
| Aspect | Full Research Model | Demo Model |
|--------|--------------------|-----------|
| Sequence Encoder | DNABERT‑2 (large transformer, millions of parameters) | 3‑mer embedding + tiny embedding layer (≈ 10 k parameters) |
| Feature Extractor | Deep hybrid of DNABERT‑2, HyenaDNA, cross‑attention fusion | 1‑D Conv + linear layer (128‑dim) |
| Training Data | 35 k+ genomic sequences, heavy augmentations | Synthetic random sequences (≤ 200 samples) |
| Hardware | Requires GPU for reasonable training time | CPU‑only, < 30 s per inference |
| Performance Claims | State‑of‑the‑art classification metrics | **Do NOT** compare performance; numbers are illustrative only |

---

## Installation
`powershell
# From the repository root
cd demo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-demo.txt
`

*If you prefer conda, replace the venv steps accordingly.*

---

## Running the Demo UI
`powershell
# From the repo root
demo\run_demo.ps1
`
The Streamlit UI will open in your default browser. Use the sidebar to upload a FASTA file or select the built‑in example sequence and then step through each stage.

---

## Training the Demo Model (Optional)
`powershell
python demo\train_demo.py
`
A synthetic dataset is generated, the model is trained for a few epochs, and the checkpoint is saved to demo/outputs/model_demo.pt.

---

## Inference from the Command Line
`powershell
python demo\inference_demo.py --fasta demo\data\demo_sequences.fasta
`
Generates a JSON report printed to STDOUT and saved under demo/outputs/.

---

## Testing
`powershell
python -m pytest demo/tests
python -m py_compile 
`
All tests must pass.

---

## Hardware Requirements
- CPU only (Intel/AMD, 8 GB RAM recommended)
- No GPU, no large external databases
- Disk space < 50 MB (demo data & model)

---

## Troubleshooting
- **Missing BLAST** – The demo automatically falls back to the lightweight alignment mode.
- **SHAP errors** – Ensure shap is installed and that scikit-learn is available; otherwise a simple feature‑importance fallback is shown.
- **File not found** – Verify you are running commands from the repository root.

---

## License
MIT – see the top‑level LICENSE file.

---

*Happy exploring!*
