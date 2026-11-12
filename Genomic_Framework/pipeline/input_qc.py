"""
pipeline/input_qc.py
====================
Module 1 — Raw DNA Input Ingestion
Module 2 — Quality Control & Sequence Standardisation

Responsibilities
----------------
* Accept raw genetic text (plain strings, FASTA/FASTQ simulation).
* Scrub every character that is not in the canonical IUPAC set {A, C, G, T, N}.
* Pad short sequences with 'N' and truncate long sequences to ``max_seq_len``.
* Return a clean list of sequences **and** a detailed execution-status log.

Design Notes
------------
All logic is self-contained in :class:`GenomicSequenceIngester`.  No external
bioinformatics libraries (Biopython, pysam …) are required; the class is
intentionally dependency-free so the full framework can run in any Python ≥ 3.9
environment without extra setup.
"""

from __future__ import annotations

import re
import time
import logging
from typing import Dict, List, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VALID_NUCLEOTIDES: frozenset[str] = frozenset("ACGTN")
_FASTA_HEADER_RE = re.compile(r"^>.*$", re.MULTILINE)
_FASTQ_QUALITY_RE = re.compile(r"^\+.*$", re.MULTILINE)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _strip_fasta(raw: str) -> str:
    """Remove FASTA header lines (lines starting with '>') and return the
    concatenated nucleotide body."""
    lines = raw.strip().splitlines()
    seq_lines = [ln.strip() for ln in lines if not ln.startswith(">")]
    return "".join(seq_lines)


def _strip_fastq(raw: str) -> str:
    """Parse a single FASTQ record and return only the sequence line.
    Expects the canonical 4-line FASTQ layout."""
    lines = [ln.strip() for ln in raw.strip().splitlines()]
    # FASTQ: line 0 = @header, line 1 = sequence, line 2 = +, line 3 = quality
    if len(lines) >= 2:
        return lines[1]
    return lines[0]  # fallback — treat entire content as sequence


def _detect_format(raw: str) -> str:
    """Heuristically detect whether *raw* is plain, FASTA, or FASTQ."""
    stripped = raw.strip()
    if stripped.startswith(">"):
        return "fasta"
    if stripped.startswith("@"):
        return "fastq"
    return "plain"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GenomicSequenceIngester:
    """
    Raw DNA input ingestion and quality-control pipeline stage.

    Parameters
    ----------
    max_seq_len : int
        Target sequence length after padding / truncation.  All output
        sequences will be **exactly** ``max_seq_len`` characters long.
    pad_char : str
        Single character used for right-padding when a sequence is shorter
        than ``max_seq_len``.  Defaults to ``'N'`` (unknown nucleotide).
    strict_validation : bool
        When ``True``, sequences that yield fewer than ``min_valid_fraction``
        valid nucleotides after scrubbing are **rejected** and excluded from
        the output.  When ``False``, they are cleaned and kept regardless.
    min_valid_fraction : float
        Minimum fraction (0 – 1) of valid nucleotides that must survive the
        scrubbing step for a sequence to be retained in strict mode.

    Examples
    --------
    >>> ingester = GenomicSequenceIngester(max_seq_len=128)
    >>> seqs, log = ingester.clean_and_validate(["ACGTACGT", "ACGT!XYZ"])
    >>> print(seqs)
    ['ACGTACGT' + 'N' * 120, 'ACGT' + 'N' * 124]
    """

    def __init__(
        self,
        max_seq_len: int = 512,
        pad_char: str = "N",
        strict_validation: bool = True,
        min_valid_fraction: float = 0.50,
    ) -> None:
        if len(pad_char) != 1 or pad_char not in _VALID_NUCLEOTIDES:
            raise ValueError(
                f"pad_char must be a single valid nucleotide; got '{pad_char}'."
            )
        if not (0.0 < min_valid_fraction <= 1.0):
            raise ValueError("min_valid_fraction must be in (0, 1].")

        self.max_seq_len: int = max_seq_len
        self.pad_char: str = pad_char
        self.strict_validation: bool = strict_validation
        self.min_valid_fraction: float = min_valid_fraction

        logger.info(
            "GenomicSequenceIngester initialised — max_seq_len=%d, "
            "strict=%s, min_valid_fraction=%.2f",
            max_seq_len,
            strict_validation,
            min_valid_fraction,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean_and_validate(
        self,
        sequences: Union[str, List[str]],
    ) -> Tuple[List[str], Dict]:
        """
        Ingest, clean, and validate a batch of raw genomic sequences.

        Parameters
        ----------
        sequences : str | list[str]
            A single sequence string or a list of raw sequence strings.
            Each entry may be plain nucleotides, a FASTA record, or a
            FASTQ record — the format is auto-detected.

        Returns
        -------
        cleaned_sequences : list[str]
            Standardised sequences of length exactly ``max_seq_len``.
        status_log : dict
            Execution-status dictionary containing:
            ``n_input``, ``n_accepted``, ``n_rejected``, ``rejected_indices``,
            ``per_sequence_stats``, ``wall_time_s``.

        Raises
        ------
        TypeError
            If *sequences* is neither a string nor a list.
        """
        t_start = time.perf_counter()

        if isinstance(sequences, str):
            sequences = [sequences]
        if not isinstance(sequences, list):
            raise TypeError(
                f"sequences must be str or list[str]; got {type(sequences)}."
            )

        cleaned: List[str] = []
        per_seq_stats: List[Dict] = []
        rejected_indices: List[int] = []

        for idx, raw in enumerate(sequences):
            stat, processed = self._process_single(idx, raw)
            per_seq_stats.append(stat)

            if stat["accepted"]:
                cleaned.append(processed)
            else:
                rejected_indices.append(idx)
                logger.warning(
                    "Sequence #%d rejected — valid_fraction=%.3f < %.3f",
                    idx,
                    stat["valid_fraction_before_pad"],
                    self.min_valid_fraction,
                )

        wall_time = time.perf_counter() - t_start

        status_log: Dict = {
            "n_input": len(sequences),
            "n_accepted": len(cleaned),
            "n_rejected": len(rejected_indices),
            "rejected_indices": rejected_indices,
            "per_sequence_stats": per_seq_stats,
            "wall_time_s": round(wall_time, 6),
        }

        logger.debug(
            "QC complete — accepted=%d / %d in %.4f s",
            len(cleaned),
            len(sequences),
            wall_time,
        )
        return cleaned, status_log

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_single(
        self,
        idx: int,
        raw: str,
    ) -> Tuple[Dict, str]:
        """
        Process a single raw sequence string.

        Returns
        -------
        stat : dict
            Per-sequence statistics dictionary.
        cleaned_seq : str
            Padded / truncated sequence of length ``max_seq_len`` (empty
            string if rejected).
        """
        # --- 1. Format detection and header stripping ---
        fmt = _detect_format(raw)
        if fmt == "fasta":
            body = _strip_fasta(raw)
        elif fmt == "fastq":
            body = _strip_fastq(raw)
        else:
            body = raw.strip()

        # --- 2. Uppercase normalisation ---
        body_upper = body.upper()
        original_len = len(body_upper)

        # --- 3. Character scrubbing — keep only {A, C, G, T, N} ---
        scrubbed = "".join(ch for ch in body_upper if ch in _VALID_NUCLEOTIDES)
        n_valid = len(scrubbed)
        valid_fraction = n_valid / original_len if original_len > 0 else 0.0

        # --- 4. Rejection check (strict mode) ---
        accepted = True
        if self.strict_validation and valid_fraction < self.min_valid_fraction:
            accepted = False

        # --- 5. Pad / truncate to max_seq_len ---
        if accepted:
            if len(scrubbed) < self.max_seq_len:
                cleaned_seq = scrubbed.ljust(self.max_seq_len, self.pad_char)
            else:
                cleaned_seq = scrubbed[: self.max_seq_len]
        else:
            cleaned_seq = ""

        stat: Dict = {
            "index": idx,
            "detected_format": fmt,
            "original_len": original_len,
            "scrubbed_len": n_valid,
            "valid_fraction_before_pad": round(valid_fraction, 4),
            "final_len": len(cleaned_seq),
            "accepted": accepted,
            "truncated": original_len > self.max_seq_len,
            "padded": n_valid < self.max_seq_len and accepted,
        }
        return stat, cleaned_seq

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"GenomicSequenceIngester("
            f"max_seq_len={self.max_seq_len}, "
            f"pad_char='{self.pad_char}', "
            f"strict={self.strict_validation})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test (module-level)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_sequences: List[str] = [
        "ACGTACGTNNACGT",                          # plain — short, will be padded
        ">chr1\nACGTACGTACGTACGT",                  # FASTA record
        "@read_001\nACGTNNACGT\n+\nIIIIIIIIII",    # FASTQ record
        "ACGT!!!XYZXYZ",                            # mostly invalid — may be rejected
        "A" * 600,                                  # long — will be truncated
    ]

    ingester = GenomicSequenceIngester(
        max_seq_len=32,
        strict_validation=True,
        min_valid_fraction=0.50,
    )
    cleaned_seqs, status = ingester.clean_and_validate(sample_sequences)

    print("\n=== GenomicSequenceIngester Smoke Test ===")
    for i, seq in enumerate(cleaned_seqs):
        print(f"  Cleaned seq {i}: length={len(seq)!s:>4}  preview={seq[:20]}…")
    print(f"\nStatus log summary:")
    for k, v in status.items():
        if k != "per_sequence_stats":
            print(f"  {k}: {v}")
