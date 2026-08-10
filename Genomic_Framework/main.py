"""
main.py
=======
End-to-End Demonstration Script — Genomic Sequence Classification Framework

This script exercises the **entire** pipeline from raw DNA string input
through QC → alignment → mutation processing → feature extraction →
cross-attention fusion → classification → explainability → report.

Run with:
    python main.py

All processing is self-contained and uses synthetic/dummy data so no
external datasets, pre-trained weights, or internet access is required.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

# ---------------------------------------------------------------------------
# Make sure the repo root is on the Python path when run directly
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# ---------------------------------------------------------------------------
# Framework imports
# ---------------------------------------------------------------------------
from Genomic_Framework.pipeline.input_qc import GenomicSequenceIngester
from Genomic_Framework.pipeline.mutation_pipeline import (
    SequenceAligner,
    MutationProcessor,
)
from Genomic_Framework.engine import (
    GenomicClassificationEngine,
    ExplainabilityEngine,
    GenomicReportGenerator,
    CLASS_LABELS,
    # Configuration constants
    SEQ_LEN,
    BATCH_SIZE,
    VOCAB_SIZE,
    HYENA_DIM,
    HIDDEN_DIM,
    NUM_CLASSES,
    MAX_MUTATIONS,
    MUTATION_DIM,
    run_verification,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic sequence generator
# ---------------------------------------------------------------------------

def generate_synthetic_sequences(
    n: int = 4,
    min_len: int = 80,
    max_len: int = 600,
    seed: int = 42,
) -> List[str]:
    """
    Generate *n* random ACGTN sequences of varying lengths.

    A small fraction of characters are intentionally set to invalid symbols
    to exercise the QC scrubbing logic.
    """
    rng = random.Random(seed)
    bases = ["A", "C", "G", "T", "N"]
    invalid = ["X", "Z", "!", "1", " "]
    seqs: List[str] = []
    for i in range(n):
        length = rng.randint(min_len, max_len)
        chars = [
            rng.choice(invalid) if rng.random() < 0.05 else rng.choice(bases)
            for _ in range(length)
        ]
        seqs.append("".join(chars))
    return seqs


# ---------------------------------------------------------------------------
# Dummy tokeniser (character-level k-mer simulation)
# ---------------------------------------------------------------------------

def tokenise_sequences(
    sequences: List[str],
    seq_len: int = SEQ_LEN,
    vocab_size: int = VOCAB_SIZE,
    seed: int = 0,
) -> torch.Tensor:
    """
    Convert cleaned sequence strings to integer token tensors.

    In production this would use the DNABERT-2 / HyenaDNA tokeniser.
    Here we simulate it with a deterministic character hash into [1, vocab_size).

    Parameters
    ----------
    sequences : list[str]
        Cleaned sequences of equal length (output of QC stage).
    seq_len : int
        Target token length (sequences are truncated / padded to this).

    Returns
    -------
    torch.Tensor
        Shape ``[len(sequences), seq_len]``.
    """
    base_map = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
    token_matrix: List[List[int]] = []
    for seq in sequences:
        tokens = []
        for ch in seq[:seq_len]:
            base_tok = base_map.get(ch, 5)
            # Scatter into larger vocabulary via simple hash
            tok = (base_tok * 811 + 37) % (vocab_size - 1) + 1
            tokens.append(tok)
        # Pad to seq_len if needed
        tokens += [0] * (seq_len - len(tokens))
        token_matrix.append(tokens)
    return torch.tensor(token_matrix, dtype=torch.long)


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Execute the full end-to-end genomic classification pipeline.
    """
    t_global_start = time.perf_counter()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 0 — Structural integrity check
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "+" + "=" * 70 + "+")
    print("|  GENOMIC SEQUENCE CLASSIFICATION FRAMEWORK -- FULL PIPELINE RUN  |")
    print("+" + "=" * 70 + "+")

    print("\n[STEP 0] Running structural integrity verification...")
    run_verification()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1 — Generate synthetic raw sequences
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[STEP 1] Generating synthetic raw DNA sequences...")
    raw_sequences = generate_synthetic_sequences(n=BATCH_SIZE, seed=99)
    for i, seq in enumerate(raw_sequences):
        print(f"  Raw[{i}] length={len(seq):>4}  preview={seq[:30]}...")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2 — QC / Ingestion (Modules 1–2)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[STEP 2] Running QC & Sequence Ingestion (Modules 1-2)...")
    ingester = GenomicSequenceIngester(
        max_seq_len=SEQ_LEN * 4,  # allow longer strings before tokenisation
        strict_validation=True,
        min_valid_fraction=0.50,
    )
    cleaned_seqs, qc_log = ingester.clean_and_validate(raw_sequences)
    print(f"  QC accepted: {qc_log['n_accepted']} / {qc_log['n_input']} sequences")
    print(f"  QC rejected: {qc_log['n_rejected']} sequences")
    for i, seq in enumerate(cleaned_seqs):
        print(f"  Cleaned[{i}] length={len(seq)}  preview={seq[:30]}...")

    # Guard: if too few sequences survive QC, pad with synthetic
    while len(cleaned_seqs) < BATCH_SIZE:
        cleaned_seqs.append("ACGT" * (SEQ_LEN // 4))

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3 — Reference Alignment (Module 3)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[STEP 3] Reference Alignment (Module 3)...")
    aligner = SequenceAligner(seed=42)
    alignment_maps = [
        aligner.compute_alignment_map(seq) for seq in cleaned_seqs
    ]
    for i, amap in enumerate(alignment_maps):
        print(
            f"  Alignment[{i}] - mismatches={len(amap.mismatch_positions):>3}  "
            f"edit_dist={amap.edit_distance:>3}  "
            f"alignment[:20]={amap.alignment_string[:20]}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4 — Mutation Processing (Modules 6–9)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[STEP 4] Mutation Identification -> Annotation -> Prioritisation -> Encoding (Modules 6-9)...")
    processor = MutationProcessor(
        max_mutations=MAX_MUTATIONS,
        mutation_dim=MUTATION_DIM,
    )

    # Process each sample and stack into a batch tensor
    mut_tensors = [
        processor.process(amap, batch_size=1)  # [1, M, mut_dim]
        for amap in alignment_maps
    ]
    mutation_tensor = torch.cat(mut_tensors, dim=0)  # [B, M, mut_dim]
    print(f"  Mutation tensor shape: {list(mutation_tensor.shape)}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5 — Tokenisation (bridge between text pipeline and neural models)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[STEP 5] Tokenising cleaned sequences...")
    token_ids = tokenise_sequences(cleaned_seqs, seq_len=SEQ_LEN)
    print(f"  Token IDs shape : {list(token_ids.shape)}")
    print(f"  Sample tokens   : {token_ids[0, :10].tolist()}...")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6 — Neural Engine Forward Pass (Modules 4, 5, 10, 11)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[STEP 6] Neural Classification Engine forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = GenomicClassificationEngine(
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        hyena_dim=HYENA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        max_mutations=MAX_MUTATIONS,
        mutation_dim=MUTATION_DIM,
    ).to(device)
    engine.eval()

    token_ids      = token_ids.to(device)
    mutation_tensor = mutation_tensor.to(device)

    with torch.no_grad():
        logits, fused = engine(token_ids, mutation_tensor)

    print(f"  Token IDs        -> {list(token_ids.shape)}")
    print(f"  Mutation tensor  -> {list(mutation_tensor.shape)}")
    print(f"  Fused vector     -> {list(fused.shape)}")
    print(f"  Logits           -> {list(logits.shape)}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7 — Explainability (Module 12)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[STEP 7] Computing SHAP-style attributions (Module 12)...")
    explain_engine = ExplainabilityEngine(model=engine, n_steps=30)
    fused_grad = fused.detach().requires_grad_(True)
    attributions = explain_engine.compute_attributions(
        fused_vector=fused_grad,
        target_class=0,
    )
    print(f"  Attribution tensor -> {list(attributions.shape)}")
    top_feats = explain_engine.top_k_features(attributions, k=5)
    print(f"  Top-5 feature indices : {top_feats['indices']}")
    print(f"  Top-5 SHAP scores     : {[round(s, 5) for s in top_feats['scores']]}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 8 — Report Generation (Module 13)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[STEP 8] Generating classification reports (Module 13)...")
    reporter = GenomicReportGenerator(class_labels=CLASS_LABELS, top_shap_k=5)
    sample_ids = [f"PATIENT_{i:04d}" for i in range(len(cleaned_seqs))]

    reports = reporter.generate(
        logits=logits,
        attributions=attributions,
        sample_ids=sample_ids,
        metadata={
            "pipeline_version": "1.0.0",
            "reference_genome": "GRCh38",
            "model": "GenomicClassificationEngine v1",
        },
    )

    # Save reports
    report_path = _repo_root / "genomic_reports.json"
    reporter.save_report(reports, str(report_path))
    print(f"\n  Reports saved -> {report_path}")

    # Pretty-print first report
    print("\n  --- Sample Report (first sample) ---")
    print(json.dumps(reports[0], indent=4))

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    t_elapsed = time.perf_counter() - t_global_start
    print(f"\n{'=' * 72}")
    print(f"  PIPELINE COMPLETE  |  Total wall time: {t_elapsed:.2f}s")
    print(f"{'=' * 72}\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
