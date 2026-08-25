import yaml
import os
import subprocess
import shutil
from pathlib import Path
from difflib import SequenceMatcher
import pandas as pd

# Load config
from ..config import load_config
config = load_config()

def _load_reference() -> str:
    """Load the first sequence from the reference FASTA file in demo/data/."""
    ref_path = Path(__file__).resolve().parents[1] / "data" / "references.fasta"
    with open(ref_path, "r") as fh:
        lines = [l.strip() for l in fh if not l.startswith('>')]
    return ''.join(lines)

def _run_blast(query_seq: str) -> dict:
    """Run external BLAST+ if configured.
    Returns a dict with keys: 'identity', 'alignment_length', 'mismatches', 'gaps'.
    """
    blast_path = config.get('blast_path')
    blast_db = config.get('blast_db')
    if not blast_path or not blast_db:
        raise RuntimeError('BLAST path or database not configured')
    # Write temporary query FASTA
    tmp_dir = Path(__file__).resolve().parents[2] / "outputs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    query_path = tmp_dir / "tmp_query.fasta"
    with open(query_path, "w") as fq:
        fq.write(f">query\n{query_seq}\n")
    # Run blastn
    cmd = [blast_path, '-query', str(query_path), '-db', str(blast_db), '-outfmt', '6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore']
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"BLAST execution failed: {result.stderr}")
    # Parse first line
    fields = result.stdout.strip().split('\t')
    if not fields:
        raise RuntimeError('BLAST returned no hits')
    pident = float(fields[2])
    length = int(fields[3])
    mismatches = int(fields[4])
    gaps = int(fields[5])
    return {'identity': pident / 100.0, 'alignment_length': length, 'mismatches': mismatches, 'gaps': gaps}

def align_sequence(query_seq: str) -> dict:
    """Align `query_seq` against the reference.
    Returns a dictionary with alignment statistics.
    If BLAST is configured and the executable is found, it is used;
    otherwise a lightweight difflib based alignment is performed.
    """
    # Detect if BLAST is usable
    blast_path = config.get('blast_path')
    if blast_path and shutil.which(blast_path):
        try:
            return _run_blast(query_seq)
        except Exception as e:
            # Fallback to demo mode on any error
            print(f"[WARN] BLAST failed ({e}), falling back to demo alignment.")
    # Demo alignment using difflib
    ref_seq = _load_reference()
    matcher = SequenceMatcher(None, query_seq, ref_seq)
    ratio = matcher.ratio()  # similarity between 0 and 1
    # Compute mismatches & gaps roughly
    matches = sum(1 for a, b in zip(query_seq, ref_seq) if a == b)
    mismatches = sum(1 for a, b in zip(query_seq, ref_seq) if a != b)
    gaps = abs(len(query_seq) - len(ref_seq))
    return {
        'identity': ratio,
        'alignment_length': min(len(query_seq), len(ref_seq)),
        'mismatches': mismatches,
        'gaps': gaps,
    }
