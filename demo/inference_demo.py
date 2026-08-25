import argparse
import yaml
import torch
from pathlib import Path
import json

# Import demo components
from demo.models.demo_model import DemoModel
from demo.pipeline.qc import quality_control
from demo.pipeline.alignment import align_sequence
from demo.pipeline.mutation import identify_mutations
from demo.pipeline.annotation import annotate_mutations
from demo.pipeline.prioritization import prioritize_mutations
from demo.pipeline.report import build_report, save_report
try:
    import shap
except ImportError:
    shap = None
import numpy as np

# Load config
from .config import load_config
config = load_config()

def load_model(checkpoint_path: Path) -> DemoModel:
    model = DemoModel()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def run_shap(model: DemoModel, background_seqs, sample_seq, mut_pos, mut_type):
    # Create a wrapper that accepts a list of sequences and returns class probs
    def model_wrapper(seqs):
        probs = []
        for seq in seqs:
            prob = model(seq, mut_pos, mut_type)
            probs.append(prob.detach().numpy())
        return np.vstack(probs)
    explainer = shap.KernelExplainer(model_wrapper, background_seqs)
    shap_vals = explainer.shap_values([sample_seq], nsamples=100)
    # shap_vals is list per class; we flatten to a single array for simplicity
    return shap_vals

def main():
    parser = argparse.ArgumentParser(description="Run lightweight demo inference on a FASTA file.")
    parser.add_argument("--fasta", required=True, help="Path to input FASTA file (single sequence).")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / "outputs" / "model_demo.pt"), help="Path to trained demo model checkpoint.")
    args = parser.parse_args()

    # Load sequence from FASTA (first sequence only)
    with open(args.fasta, "r") as fh:
        lines = [l.strip() for l in fh if not l.startswith('>')]
    raw_seq = ''.join(lines)

    # Step 1: QC
    cleaned_seq = quality_control(raw_seq)

    # Step 2: Alignment (demo mode)
    alignment_stats = align_sequence(cleaned_seq)

    # Load reference for mutation identification
    ref_path = Path(__file__).resolve().parent / "data" / "references.fasta"
    with open(ref_path, "r") as rh:
        ref_seq = ''.join([l.strip() for l in rh if not l.startswith('>')])

    # Step 3: Mutation identification
    mut_positions, mut_types = identify_mutations(cleaned_seq, ref_seq)

    # Step 4: Annotation
    annotations = annotate_mutations(mut_types)

    # Step 5: Prioritization
    priorities = []
    if mut_positions.numel() > 0:
        priorities = prioritize_mutations(mut_positions, mut_types).tolist()

    # Load model
    model = load_model(Path(args.model))

    # Step 6: Prediction
    with torch.no_grad():
        probs = model(cleaned_seq, mut_positions.unsqueeze(0), mut_types.unsqueeze(0))
        probs = probs.squeeze(0).tolist()

    # Step 7: SHAP explainability (optional, fallback if fails)
    shap_vals = None
    try:
        # Background: take first 10 synthetic sequences if they exist in outputs folder
        bg_path = Path(__file__).resolve().parents[1] / "outputs" / "train_log.csv"
        # Use a simple dummy background of random sequences (since synthetic data not saved)
        dummy_bg = ["A"*len(cleaned_seq) for _ in range(5)]
        shap_vals = run_shap(model, dummy_bg, cleaned_seq, mut_positions, mut_types)
    except Exception as e:
        print(f"[WARN] SHAP explanation failed: {e}. Continuing without SHAP.")

    # Build report
    mutation_dict = {
        "positions": mut_positions.squeeze(1).tolist() if mut_positions.numel() > 0 else [],
        "types": ["substitution" if t[0]==1 else "insertion" if t[1]==1 else "deletion" if t[2]==1 else "unknown" for t in mut_types.tolist()] if mut_types.numel() > 0 else [],
    }
    report = build_report(
        input_seq=raw_seq,
        cleaned_seq=cleaned_seq,
        alignment_stats=alignment_stats,
        mutations=mutation_dict,
        annotations=annotations,
        priorities=priorities,
        prediction_probs=probs,
        shap_values=shap_vals,
    )
    json_path = save_report(report)
    print(f"Report saved to {json_path}")

if __name__ == "__main__":
    main()
