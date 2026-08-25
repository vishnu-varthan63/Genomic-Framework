import yaml
import os
from ..config import load_config
config = load_config()

VALID_BASES = set("ACGT")

def quality_control(seq: str) -> str:
    """Simple QC for a DNA sequence.
    - Removes whitespace and makes uppercase.
    - Checks that only A/C/G/T are present.
    - Enforces length bounds from demo_config.yaml.
    Raises:
        ValueError: if any check fails.
    Returns the cleaned sequence.
    """
    clean_seq = seq.strip().upper().replace("\n", "").replace(" ", "")
    if not set(clean_seq).issubset(VALID_BASES):
        raise ValueError("Sequence contains invalid characters; only A/C/G/T are allowed.")
    min_len = config.get("min_seq_len", 50)
    max_len = config.get("max_seq_len", 5000)
    if not (min_len <= len(clean_seq) <= max_len):
        raise ValueError(f"Sequence length {len(clean_seq)} out of bounds [{min_len}, {max_len}].")
    return clean_seq
