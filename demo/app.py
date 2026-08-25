import streamlit as st
import yaml
from pathlib import Path
import torch

# Import demo components using package-relative imports
from demo.models.demo_model import DemoModel
from demo.pipeline.qc import quality_control
from demo.pipeline.alignment import align_sequence
from demo.pipeline.mutation import identify_mutations
from demo.pipeline.annotation import annotate_mutations
from demo.pipeline.prioritization import prioritize_mutations
from demo.pipeline.report import build_report, save_report

# Load configuration (demo_config.yaml resides in demo folder)
from demo.config import load_config
config = load_config()

@st.cache_resource
def load_model() -> DemoModel:
    """Load the trained demo model checkpoint (if present)."""
    ckpt_path = Path(__file__).resolve().parents[1] / "outputs" / "model_demo.pt"
    model = DemoModel()
    if ckpt_path.is_file():
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        st.warning("Model checkpoint not found; using randomly initialized model.")
    model.eval()
    return model

model = load_model()

st.title("Genomic Demo – End‑to‑End Pipeline")
st.write("Upload a DNA sequence (FASTA) or paste it below to run the lightweight demo pipeline.")

seq_input = st.text_area("DNA sequence (FASTA format)", height=200)
if st.button("Run Demo"):
    if not seq_input:
        st.error("Please provide a DNA sequence.")
    else:
        # Extract raw sequence (ignore FASTA header lines)
        raw_seq = "".join([line.strip() for line in seq_input.splitlines() if not line.startswith('>')])
        try:
            # Step 1: QC
            cleaned_seq = quality_control(raw_seq)
            # Step 2: Alignment (demo mode)
            alignment_stats = align_sequence(cleaned_seq)
            # Load reference for mutation identification
            ref_path = Path(__file__).resolve().parent / "data" / "references.fasta"
            with open(ref_path, "r") as rh:
                ref_seq = "".join([l.strip() for l in rh if not l.startswith('>')])
            # Step 3: Mutation identification
            mut_positions, mut_types = identify_mutations(cleaned_seq, ref_seq)
            # Step 4: Annotation
            annotations = annotate_mutations(mut_types)
            # Step 5: Prioritization
            priorities = []
            if mut_positions.numel() > 0:
                priorities = prioritize_mutations(mut_positions, mut_types).tolist()
            # Step 6: Prediction
            with torch.no_grad():
                probs = model(cleaned_seq, mut_positions.unsqueeze(0), mut_types.unsqueeze(0))
                probs = probs.squeeze(0).tolist()
            # Build and display report
            mutation_dict = {
                "positions": mut_positions.squeeze(1).tolist() if mut_positions.numel() > 0 else [],
                "types": [
                    "substitution" if t[0]==1 else "insertion" if t[1]==1 else "deletion" if t[2]==1 else "unknown"
                    for t in mut_types.tolist()
                ] if mut_types.numel() > 0 else [],
            }
            report = build_report(
                input_seq=raw_seq,
                cleaned_seq=cleaned_seq,
                alignment_stats=alignment_stats,
                mutations=mutation_dict,
                annotations=annotations,
                priorities=priorities,
                prediction_probs=probs,
                shap_values=None,
            )
            st.subheader("Report")
            st.json(report)
        except Exception as e:
            st.error(f"An error occurred during the pipeline: {e}")
