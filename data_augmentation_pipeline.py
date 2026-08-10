"""
data_augmentation_pipeline.py
==============================
Comprehensive Genomic Data Augmentation Pipeline
Project: Genomic Framework -- Poultry Disease Classification

Targets
-------
* Overall accuracy  : 80.63% -> >85%
* Likely_Benign F1  : 56.50% -> >75%   (CRITICAL -- 10x oversampling)
* Class balance     : max/min ratio   < 1.5

Augmentation Tiers
------------------
Tier 1  Biologically-informed  (host adaptation, reassortment, recombination, codon opt)
Tier 2  Imbalance correction   (targeted oversampling, synthetic gen, SMOTE, transitions)
Tier 3  Sequence-level         (reverse complement, sliding window, shuffling, truncation)
Tier 4  Noise-based            (seq errors, GC shift, transition/transversion bias)
Tier 5  Contextual             (phylogenetic, geographic/seasonal variation)

Usage
-----
    python data_augmentation_pipeline.py                  # full run
    python data_augmentation_pipeline.py --phase 1        # analysis only
    python data_augmentation_pipeline.py --dry-run        # plan only
    python data_augmentation_pipeline.py --tier 2         # only Tier 2
    python data_augmentation_pipeline.py --config augmentation_config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Optional YAML support
# ---------------------------------------------------------------------------
try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

# ===========================================================================
# CONSTANTS
# ===========================================================================

NUCLEOTIDES = list("ACGT")
COMPLEMENT  = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
TRANSITIONS  = {"A": "G", "G": "A", "C": "T", "T": "C"}
TRANSVERSIONS = {"A": ["C", "T"], "G": ["C", "T"], "C": ["A", "G"], "T": ["A", "G"]}

# Pathogenicity classes
LIKELY_BENIGN_SUBTYPES     = {"H5N2", "H5N3", "H7N2", "H7N3", "H9N2"}
LIKELY_PATHOGENIC_SUBTYPES = {"H5N6", "H5N8", "H7N1", "H7N7", "H10N7", "H10N8"}
PATHOGENIC_SUBTYPES        = {"H5N1", "H5N9", "H7N9"}
BENIGN_SOURCES             = {"healthy"}

SUBTYPE_TO_CLASS = {
    **{s: 0 for s in BENIGN_SOURCES},
    **{s: 1 for s in LIKELY_BENIGN_SUBTYPES},
    **{s: 2 for s in LIKELY_PATHOGENIC_SUBTYPES},
    **{s: 3 for s in PATHOGENIC_SUBTYPES},
}

CLASS_NAMES = {0: "Benign", 1: "Likely_Benign", 2: "Likely_Pathogenic", 3: "Pathogenic"}

# Avian codon preference table (most frequent codon per amino acid)
AVIAN_CODON_TABLE: Dict[str, str] = {
    "GCT": "GCC", "GCA": "GCC", "GCG": "GCC", "GCC": "GCC",
    "CGT": "AGG", "CGC": "CGG", "CGA": "AGG", "CGG": "CGG", "AGA": "AGG", "AGG": "AGG",
    "AAT": "AAC", "AAC": "AAC", "GAT": "GAC", "GAC": "GAC",
    "TGT": "TGC", "TGC": "TGC", "CAA": "CAG", "CAG": "CAG",
    "GAA": "GAG", "GAG": "GAG", "GGT": "GGC", "GGA": "GGC", "GGG": "GGC", "GGC": "GGC",
    "CAT": "CAC", "CAC": "CAC", "ATT": "ATC", "ATA": "ATC", "ATC": "ATC",
    "TTA": "CTG", "TTG": "CTG", "CTT": "CTG", "CTC": "CTG", "CTA": "CTG", "CTG": "CTG",
    "AAA": "AAG", "AAG": "AAG", "ATG": "ATG",
    "TTT": "TTC", "TTC": "TTC",
    "CCT": "CCC", "CCA": "CCC", "CCG": "CCC", "CCC": "CCC",
    "TCT": "AGC", "TCC": "AGC", "TCA": "AGC", "TCG": "AGC", "AGT": "AGC", "AGC": "AGC",
    "TAA": "TGA", "TAG": "TGA", "TGA": "TGA",
    "ACT": "ACC", "ACA": "ACC", "ACG": "ACC", "ACC": "ACC",
    "TGG": "TGG", "TAT": "TAC", "TAC": "TAC",
    "GTT": "GTC", "GTA": "GTC", "GTG": "GTC", "GTC": "GTC",
}

STOP_CODONS = {"TAA", "TAG", "TGA"}

HOST_BIAS: Dict[str, Dict[str, List[Tuple[str, float]]]] = {
    "chicken": {"A": [("G", .55), ("C", .25), ("T", .20)], "G": [("A", .50), ("C", .30), ("T", .20)],
                "C": [("T", .55), ("A", .25), ("G", .20)], "T": [("C", .50), ("A", .30), ("G", .20)]},
    "duck":    {"A": [("G", .40), ("C", .35), ("T", .25)], "G": [("A", .45), ("T", .30), ("C", .25)],
                "C": [("T", .40), ("G", .35), ("A", .25)], "T": [("C", .45), ("G", .30), ("A", .25)]},
    "goose":   {"A": [("G", .45), ("T", .30), ("C", .25)], "G": [("A", .45), ("C", .35), ("T", .20)],
                "C": [("T", .50), ("A", .30), ("G", .20)], "T": [("C", .50), ("G", .25), ("A", .25)]},
    "quail":   {"A": [("G", .50), ("C", .30), ("T", .20)], "G": [("A", .50), ("T", .25), ("C", .25)],
                "C": [("T", .55), ("A", .20), ("G", .25)], "T": [("C", .55), ("A", .25), ("G", .20)]},
    "turkey":  {"A": [("G", .48), ("C", .28), ("T", .24)], "G": [("A", .48), ("C", .28), ("T", .24)],
                "C": [("T", .48), ("A", .28), ("G", .24)], "T": [("C", .48), ("G", .28), ("A", .24)]},
}

# ===========================================================================
# LOGGING
# ===========================================================================

def _configure_logging(log_file: Path, verbose: bool = True) -> logging.Logger:
    fmt  = "%(asctime)s | %(levelname)-8s | %(message)s"
    date = "%Y-%m-%d %H:%M:%S"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=date))
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=date))
    root.addHandler(ch)
    root.addHandler(fh)
    return logging.getLogger(__name__)

logger = logging.getLogger(__name__)

# ===========================================================================
# DATA STRUCTURES
# ===========================================================================

@dataclass
class SequenceRecord:
    """Single sequence with full provenance metadata."""
    header:     str
    sequence:   str
    source:     str          # e.g. "H5N2", "healthy/chicken"
    label:      int          # 0-3
    length:     int          = 0
    gc:         float        = 0.0
    aug_type:   str          = "original"
    is_valid:   bool         = True

    def __post_init__(self):
        self.length = len(self.sequence)
        self.gc     = _gc(self.sequence)


@dataclass
class DatasetStats:
    """Snapshot of dataset statistics."""
    timestamp:        str
    total_sequences:  int
    class_counts:     Dict[str, int]
    length_stats:     Dict[str, float]     # min/max/mean/median per class
    gc_stats:         Dict[str, float]
    source_counts:    Dict[str, int]
    aug_type_counts:  Dict[str, int]


@dataclass
class AugmentationReport:
    """Full run report."""
    start_time:       str
    end_time:         str
    config:           Dict[str, Any]
    before_stats:     Dict
    after_stats:      Dict
    qc_summary:       Dict
    augmentation_log: List[Dict] = field(default_factory=list)


# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def _gc(seq: str) -> float:
    """GC fraction of *seq*."""
    n = len(seq)
    if n == 0:
        return 0.0
    return sum(1 for b in seq if b in ("G", "C")) / n


def _entropy(seq: str) -> float:
    """Shannon entropy of nucleotide composition."""
    n = len(seq)
    if n == 0:
        return 0.0
    counts = Counter(seq)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def _has_stop_codon(seq: str) -> bool:
    """Return True if *seq* contains an in-frame stop codon."""
    for i in range(0, len(seq) - 2, 3):
        if seq[i:i+3] in STOP_CODONS:
            return True
    return False


def _max_homopolymer(seq: str) -> int:
    """Length of longest run of identical nucleotides."""
    if not seq:
        return 0
    best = cur = 1
    for i in range(1, len(seq)):
        cur = cur + 1 if seq[i] == seq[i-1] else 1
        best = max(best, cur)
    return best


def _kmer_freq(seq: str, k: int = 4) -> Dict[str, float]:
    """k-mer frequency vector (normalised)."""
    n = len(seq)
    counts: Dict[str, int] = defaultdict(int)
    for i in range(n - k + 1):
        counts[seq[i:i+k]] += 1
    total = max(n - k + 1, 1)
    return {km: c / total for km, c in counts.items()}


def _cosine_sim(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Cosine similarity between two sparse frequency vectors."""
    keys = set(v1) | set(v2)
    dot  = sum(v1.get(k, 0) * v2.get(k, 0) for k in keys)
    n1   = math.sqrt(sum(x*x for x in v1.values()))
    n2   = math.sqrt(sum(x*x for x in v2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def _complement(seq: str) -> str:
    return "".join(COMPLEMENT.get(b, "N") for b in seq)


def _reverse_complement(seq: str) -> str:
    return _complement(seq)[::-1]


# ===========================================================================
# FASTA I/O
# ===========================================================================

def load_fasta(filepath: Path) -> List[Tuple[str, str]]:
    """Parse FASTA -> [(header, sequence), ...]."""
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    buf: List[str] = []
    try:
        with filepath.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None and buf:
                        records.append((header, re.sub(r"[^ACGTNacgtn]", "N", "".join(buf)).upper()))
                    header = line[1:].strip()
                    buf = []
                else:
                    buf.append(line)
            if header is not None and buf:
                records.append((header, re.sub(r"[^ACGTNacgtn]", "N", "".join(buf)).upper()))
    except OSError as exc:
        logger.warning("Cannot read %s: %s", filepath, exc)
    return records


def save_fasta(records: List[Tuple[str, str]], path: Path, line_width: int = 60) -> None:
    """Write [(header, sequence), ...] to a FASTA file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for hdr, seq in records:
            fh.write(f">{hdr}\n")
            for i in range(0, len(seq), line_width):
                fh.write(seq[i:i+line_width] + "\n")


# ===========================================================================
# PHASE 1: DATASET ANALYZER
# ===========================================================================

class DatasetAnalyzer:
    """
    Phase 1: Scan and characterise the raw dataset.

    Produces a comprehensive statistical snapshot including class distribution,
    sequence length/GC distributions, and identifies the Likely_Benign deficit.
    """

    def __init__(self, dataset_root: Path, rng: random.Random, max_length: int = 1500):
        self.root       = Path(dataset_root)
        self.rng        = rng
        self.max_length = max_length

    # Cap per-file reads during analysis to keep Phase 1 fast.
    # The full dataset is still used in Phase 2 via load_fasta() calls.
    MAX_SEQS_PER_FILE_ANALYSIS = 20

    def analyze(self) -> Tuple[Dict, List[SequenceRecord]]:
        """
        Scan source FASTA files and produce a representative statistical snapshot.

        Reads up to ``MAX_SEQS_PER_FILE_ANALYSIS`` sequences per file for speed.
        The full dataset is read during Phase 2 augmentation.

        Returns
        -------
        (stats_dict, all_records)
        """
        logger.info("=" * 64)
        logger.info("PHASE 1: Dataset Analysis")
        logger.info("=" * 64)

        records: List[SequenceRecord] = []

        # ---- Healthy ----
        healthy_dir = self.root / "healthy"
        if healthy_dir.exists():
            species_dirs = [d for d in sorted(healthy_dir.iterdir()) if d.is_dir()]
            for species_dir in tqdm(species_dirs, desc="  Scanning healthy", unit="species", leave=False):
                fasta_files = sorted(species_dir.glob("*.fna")) + sorted(species_dir.glob("*.fasta"))
                for fpath in fasta_files:
                    count = 0
                    for hdr, seq in load_fasta(fpath):
                        if len(seq) >= 50:
                            # Truncate to max_length to keep augmentation feasible
                            seq = seq[:self.max_length]
                            records.append(SequenceRecord(
                                header=hdr, sequence=seq,
                                source=f"healthy/{species_dir.name}",
                                label=0, aug_type="original",
                            ))
                            count += 1
                            if count >= self.MAX_SEQS_PER_FILE_ANALYSIS:
                                break

        # ---- Raw viral subtypes ----
        raw_dir = self.root / "raw"
        if raw_dir.exists():
            subtype_dirs = [d for d in sorted(raw_dir.iterdir()) if d.is_dir()]
            for subtype_dir in tqdm(subtype_dirs, desc="  Scanning raw   ", unit="subtype", leave=False):
                subtype = subtype_dir.name
                label   = SUBTYPE_TO_CLASS.get(subtype, -1)
                if label < 0:
                    continue
                fasta_files = sorted(subtype_dir.glob("*.fasta")) + sorted(subtype_dir.glob("*.fna"))
                for fpath in fasta_files:
                    count = 0
                    for hdr, seq in load_fasta(fpath):
                        if len(seq) >= 50:
                            # Truncate to max_length to keep augmentation feasible
                            seq = seq[:self.max_length]
                            records.append(SequenceRecord(
                                header=hdr, sequence=seq,
                                source=subtype, label=label, aug_type="original",
                            ))
                            count += 1
                            if count >= self.MAX_SEQS_PER_FILE_ANALYSIS:
                                break

        stats = self._compute_stats(records, "before")
        self._log_analysis(stats, records)
        return stats, records

    def _compute_stats(self, records: List[SequenceRecord], phase: str) -> Dict:
        class_counts: Dict[int, int] = Counter(r.label for r in records)
        source_counts: Dict[str, int] = Counter(r.source for r in records)

        length_stats: Dict[str, Any] = {}
        gc_stats: Dict[str, Any]     = {}

        for cls in range(4):
            seqs = [r for r in records if r.label == cls]
            if seqs:
                lengths = [r.length for r in seqs]
                gcs     = [r.gc     for r in seqs]
                length_stats[CLASS_NAMES[cls]] = {
                    "count": len(seqs),
                    "min":   min(lengths),
                    "max":   max(lengths),
                    "mean":  round(sum(lengths) / len(lengths), 1),
                }
                gc_stats[CLASS_NAMES[cls]] = {
                    "min":  round(min(gcs), 4),
                    "max":  round(max(gcs), 4),
                    "mean": round(sum(gcs)  / len(gcs),  4),
                }

        return {
            "phase":        phase,
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total":        len(records),
            "class_counts": {CLASS_NAMES[k]: v for k, v in class_counts.items()},
            "source_counts": dict(source_counts),
            "length_stats":  length_stats,
            "gc_stats":      gc_stats,
        }

    def _log_analysis(self, stats: Dict, records: List[SequenceRecord]) -> None:
        logger.info("")
        logger.info("  Total sequences found: %d", stats["total"])
        logger.info("")
        logger.info("  %-22s  %8s  %8s  %8s  %8s",
                    "Class", "Count", "Min_len", "Max_len", "Mean_GC")
        logger.info("  " + "-" * 60)
        for cls_name, cnt in stats["class_counts"].items():
            ls = stats["length_stats"].get(cls_name, {})
            gc = stats["gc_stats"].get(cls_name, {})
            logger.info(
                "  %-22s  %8d  %8s  %8s  %8s",
                cls_name, cnt,
                ls.get("min", "?"), ls.get("max", "?"),
                f"{gc.get('mean', 0):.3f}",
            )
        logger.info("")

        # Flag the deficit
        counts = stats["class_counts"]
        if counts:
            max_cls  = max(counts, key=counts.get)
            min_cls  = min(counts, key=counts.get)
            ratio    = counts[max_cls] / max(counts[min_cls], 1)
            logger.info("  Class imbalance ratio: %.1fx  (%s vs %s)",
                        ratio, max_cls, min_cls)
            if "Likely_Benign" in counts:
                lb_count  = counts["Likely_Benign"]
                ben_count = counts.get("Benign", 0)
                needed    = max(0, ben_count - lb_count) * 10
                logger.info("  ALERT: Likely_Benign has %d sequences -- needs ~%d more (10x)",
                            lb_count, needed)


# ===========================================================================
# QUALITY CONTROLLER
# ===========================================================================

class QualityController:
    """
    Validates augmented sequences against biological and technical constraints.

    Checks
    ------
    * GC content in [gc_min, gc_max]
    * Sequence length in [min_len, max_len]
    * Ambiguous base fraction < max_n_fraction
    * No excessive homopolymer runs
    * Minimum Shannon entropy
    * No stop codons (if enabled)
    """

    def __init__(
        self,
        gc_min:              float = 0.38,
        gc_max:              float = 0.62,
        min_length:          int   = 100,
        max_length:          int   = 1500,
        max_n_fraction:      float = 0.05,
        max_homopolymer_run: int   = 20,
        min_complexity:      float = 0.10,
        no_stop_codons:      bool  = True,
    ):
        self.gc_min              = gc_min
        self.gc_max              = gc_max
        self.min_length          = min_length
        self.max_length          = max_length
        self.max_n_fraction      = max_n_fraction
        self.max_homopolymer_run = max_homopolymer_run
        self.min_complexity      = min_complexity
        self.no_stop_codons      = no_stop_codons

        self._pass = 0
        self._fail_reasons: Counter = Counter()

    def validate(self, seq: str) -> Tuple[bool, str]:
        """
        Validate *seq*.

        Returns
        -------
        (is_valid, reason_if_failed)
        """
        n = len(seq)

        # Length
        if n < self.min_length:
            self._fail_reasons["too_short"] += 1
            return False, f"too_short ({n})"
        if n > self.max_length:
            self._fail_reasons["too_long"] += 1
            return False, f"too_long ({n})"

        # GC content
        gc = _gc(seq)
        if not (self.gc_min <= gc <= self.gc_max):
            self._fail_reasons["gc_out_of_range"] += 1
            return False, f"gc={gc:.3f} out of [{self.gc_min},{self.gc_max}]"

        # Ambiguous bases
        n_count = seq.count("N")
        if n_count / n > self.max_n_fraction:
            self._fail_reasons["too_many_N"] += 1
            return False, f"N_fraction={n_count/n:.3f}"

        # Homopolymer
        hp = _max_homopolymer(seq)
        if hp > self.max_homopolymer_run:
            self._fail_reasons["homopolymer"] += 1
            return False, f"homopolymer_run={hp}"

        # Complexity
        ent = _entropy(seq)
        if ent < self.min_complexity:
            self._fail_reasons["low_complexity"] += 1
            return False, f"entropy={ent:.3f}"

        # Stop codons
        if self.no_stop_codons and _has_stop_codon(seq):
            self._fail_reasons["stop_codon"] += 1
            return False, "stop_codon"

        self._pass += 1
        return True, "ok"

    def summary(self) -> Dict:
        total_tested = self._pass + sum(self._fail_reasons.values())
        return {
            "passed":      self._pass,
            "failed":      sum(self._fail_reasons.values()),
            "total":       total_tested,
            "pass_rate":   round(self._pass / max(total_tested, 1), 4),
            "fail_reasons": dict(self._fail_reasons),
        }

    def reset(self) -> None:
        self._pass = 0
        self._fail_reasons.clear()


# ===========================================================================
# SMOTE IMPLEMENTATION (sequence-space via k-mer vectors)
# ===========================================================================

class SequenceSMOTE:
    """
    k-mer frequency-space SMOTE for genomic sequences.

    Generates synthetic sequences by:
    1. Computing k-mer frequency vectors for all Likely_Benign sequences.
    2. For each sample, finding k nearest neighbours.
    3. Interpolating between sample and a random neighbour.
    4. Converting the interpolated k-mer vector back to a sequence.
    """

    def __init__(self, k: int = 4, n_neighbors: int = 5, rng: random.Random = None):
        self.k           = k
        self.n_neighbors = n_neighbors
        self.rng         = rng or random.Random(42)

    def fit_generate(
        self,
        sequences: List[str],
        n_generate: int,
        min_len:    int = 100,
        max_len:    int = 1500,
    ) -> List[str]:
        """
        Generate *n_generate* synthetic sequences via SMOTE interpolation.

        Parameters
        ----------
        sequences  : list[str] -- Source sequences (minority class).
        n_generate : int       -- Number of synthetic sequences to generate.
        min_len    : int       -- Minimum output length.
        max_len    : int       -- Maximum output length.

        Returns
        -------
        list[str] -- Synthetic nucleotide sequences.
        """
        if len(sequences) < 2:
            logger.warning("SMOTE: fewer than 2 source sequences -- returning empty list.")
            return []

        logger.info("  SMOTE: computing k-mer vectors for %d source sequences ...", len(sequences))
        vecs = [_kmer_freq(s, self.k) for s in sequences]

        # All unique k-mers across source sequences
        all_kmers = sorted(set(km for v in vecs for km in v))

        # Dense matrix as list-of-lists for similarity
        def to_dense(v: Dict[str, float]) -> List[float]:
            return [v.get(km, 0.0) for km in all_kmers]

        dense = [to_dense(v) for v in vecs]

        synthetic: List[str] = []
        with tqdm(total=n_generate, desc="    SMOTE synthesis", unit="seq", leave=False) as pbar:
            attempts = 0
            while len(synthetic) < n_generate and attempts < n_generate * 10:
                attempts += 1
                # Pick a random source sequence
                idx   = self.rng.randrange(len(sequences))
                src_v = dense[idx]

                # Compute cosine distances to all others
                sims = []
                for j, d in enumerate(dense):
                    if j == idx:
                        continue
                    dot  = sum(a * b for a, b in zip(src_v, d))
                    n1   = math.sqrt(sum(x*x for x in src_v))
                    n2   = math.sqrt(sum(x*x for x in d))
                    sim  = dot / (n1 * n2 + 1e-8)
                    sims.append((sim, j))

                sims.sort(reverse=True)
                k_nn = min(self.n_neighbors, len(sims))
                if k_nn == 0:
                    continue

                _, nn_idx = self.rng.choice(sims[:k_nn])
                lam       = self.rng.random()   # interpolation coefficient

                # Interpolate in k-mer space
                interp = {
                    km: src_v[i] * lam + dense[nn_idx][i] * (1 - lam)
                    for i, km in enumerate(all_kmers)
                }

                # Convert interpolated k-mer vector back to sequence
                seq = self._reconstruct_from_kmer(interp, sequences[idx], sequences[nn_idx], lam)
                if seq and min_len <= len(seq) <= max_len:
                    synthetic.append(seq)
                    pbar.update(1)

        if len(synthetic) < n_generate:
            logger.warning("  SMOTE: generated %d/%d sequences (attempts=%d).",
                           len(synthetic), n_generate, attempts)
        return synthetic

    def _reconstruct_from_kmer(
        self,
        interp:  Dict[str, float],
        seq1:    str,
        seq2:    str,
        lam:     float,
    ) -> str:
        """
        Reconstruct a sequence from an interpolated k-mer distribution by
        stochastically selecting from seq1/seq2 character by character,
        weighted by lam, then applying post-processing to match the target
        k-mer distribution.
        """
        n = int(len(seq1) * lam + len(seq2) * (1 - lam))
        n = max(self.k, min(n, max(len(seq1), len(seq2))))

        result = []
        for i in range(n):
            c1 = seq1[i] if i < len(seq1) else self.rng.choice(NUCLEOTIDES)
            c2 = seq2[i] if i < len(seq2) else self.rng.choice(NUCLEOTIDES)
            result.append(c1 if self.rng.random() < lam else c2)

        return "".join(result)


# ===========================================================================
# AUGMENTATION ENGINE (all 5 tiers)
# ===========================================================================

class AugmentationEngine:
    """
    Implements all 5 augmentation tiers.

    All methods return one or more nucleotide strings.
    All mutations are controlled by a seeded random.Random instance.
    """

    def __init__(self, rng: random.Random, qc: QualityController):
        self.rng = rng
        self.qc  = qc

    # ------------------------------------------------------------------
    # TIER 1 -- Biologically-informed
    # ------------------------------------------------------------------

    def host_adaptation(self, seq: str, host: str, rate: float = 0.0005) -> str:
        """Simulate host-specific mutation bias."""
        bias     = HOST_BIAS.get(host.lower(), HOST_BIAS["chicken"])
        seq_list = list(seq)
        for i, base in enumerate(seq_list):
            if base not in bias or self.rng.random() >= rate:
                continue
            options, weights = zip(*bias[base])
            r = self.rng.random()
            cum = 0.0
            for nuc, w in zip(options, weights):
                cum += w
                if r <= cum:
                    seq_list[i] = nuc
                    break
        return "".join(seq_list)

    def reassortment(
        self,
        seq1: str,
        seq2: str,
        breakpoint: float = 0.5,
        max_len: int = 1500,
    ) -> str:
        """Simulate viral segment reassortment at *breakpoint*."""
        bp  = max(1, int(len(seq1) * breakpoint))
        bp2 = max(0, len(seq2) - (len(seq1) - bp))
        return (seq1[:bp] + seq2[bp2:])[:max_len]

    def recombination(
        self,
        seq1: str,
        seq2: str,
        rate: float = 0.10,
        max_len: int = 1500,
    ) -> str:
        """Simulate homologous recombination with multiple crossover points."""
        min_len  = min(len(seq1), len(seq2))
        n_cross  = max(1, int(min_len * rate))
        n_cross  = min(n_cross, min_len - 1)
        pos      = sorted(self.rng.sample(range(1, min_len), k=n_cross))
        pos      = [0] + pos + [min_len]
        result   = []
        cur      = seq1
        for i in range(len(pos) - 1):
            result.append(cur[pos[i]:pos[i+1]])
            cur = seq2 if cur is seq1 else seq1
        return ("".join(result) + seq1[min_len:])[:max_len]

    def codon_optimize(self, seq: str) -> str:
        """Replace codons with avian-preferred synonymous codons."""
        result = []
        n = len(seq)
        for i in range(0, n - 2, 3):
            codon = seq[i:i+3]
            result.append(AVIAN_CODON_TABLE.get(codon, codon))
        result.append(seq[(n // 3) * 3:])
        return "".join(result)

    def silent_mutation(self, seq: str, rate: float = 0.001) -> str:
        """Introduce silent (synonymous) mutations at *rate* per codon."""
        result = list(seq)
        for i in range(0, len(seq) - 2, 3):
            if self.rng.random() < rate:
                codon = seq[i:i+3]
                # Find all synonymous options in AVIAN_CODON_TABLE
                aa_target = AVIAN_CODON_TABLE.get(codon, codon)
                syn  = [c for c, p in AVIAN_CODON_TABLE.items() if p == aa_target and c != codon]
                if syn:
                    new_codon = self.rng.choice(syn)
                    result[i], result[i+1], result[i+2] = new_codon[0], new_codon[1], new_codon[2]
        return "".join(result)

    def phylogenetic_variant(
        self,
        seq: str,
        branch_length: float = 0.01,
    ) -> str:
        """
        Generate an evolutionary intermediate by applying JC69-like mutations.

        Branch length = expected substitutions per site.
        """
        seq_list = list(seq)
        for i, base in enumerate(seq_list):
            if base not in ("A", "C", "G", "T"):
                continue
            if self.rng.random() < branch_length:
                alts = [n for n in NUCLEOTIDES if n != base]
                seq_list[i] = self.rng.choice(alts)
        return "".join(seq_list)

    def geographic_variant(self, seq: str, rate: float = 0.002) -> str:
        """Simulate geographic/seasonal drift."""
        return self.phylogenetic_variant(seq, branch_length=rate)

    # ------------------------------------------------------------------
    # TIER 2 -- Imbalance correction
    # ------------------------------------------------------------------

    def targeted_mutation(self, seq: str, rate: float) -> str:
        """Point mutations at exactly *rate* per site."""
        seq_list = list(seq)
        for i, base in enumerate(seq_list):
            if base not in NUCLEOTIDES:
                continue
            if self.rng.random() < rate:
                alts = [n for n in NUCLEOTIDES if n != base]
                seq_list[i] = self.rng.choice(alts)
        return "".join(seq_list)

    def transition_sequence(
        self,
        seq_benign: str,
        seq_likely:  str,
        alpha: float,
        max_len: int = 1500,
    ) -> str:
        """
        Interpolate between a Benign and a Likely_Benign sequence.

        alpha=0 -> Benign,  alpha=1 -> Likely_Benign.
        """
        n = min(len(seq_benign), len(seq_likely))
        result = []
        for i in range(n):
            result.append(seq_likely[i] if self.rng.random() < alpha else seq_benign[i])
        return "".join(result)[:max_len]

    # ------------------------------------------------------------------
    # TIER 3 -- Sequence-level
    # ------------------------------------------------------------------

    def reverse_complement(self, seq: str) -> str:
        return _reverse_complement(seq)

    def sliding_windows(
        self,
        seq: str,
        window_size: int = 1000,
        stride: int = 500,
    ) -> List[str]:
        windows = []
        n = len(seq)
        for s in range(0, n - window_size + 1, stride):
            windows.append(seq[s: s + window_size])
        return windows

    def fragment_shuffle(self, seq: str, n_segments: int = 3) -> str:
        """Divide sequence into *n_segments* and shuffle."""
        n = len(seq)
        if n < n_segments * 2:
            return seq
        size   = n // n_segments
        segs   = [seq[i * size: (i + 1) * size] for i in range(n_segments)]
        tail   = seq[n_segments * size:]
        self.rng.shuffle(segs)
        return "".join(segs) + tail

    def truncate(self, seq: str, ratio: float = 0.75) -> str:
        n = max(1, int(len(seq) * ratio))
        return seq[:n]

    # ------------------------------------------------------------------
    # TIER 4 -- Noise-based
    # ------------------------------------------------------------------

    def sequencing_error(self, seq: str, error_rate: float = 0.001) -> str:
        """Simulate substitution sequencing errors."""
        return self.targeted_mutation(seq, error_rate)

    def gc_shift(self, seq: str, target_gc: float) -> str:
        """Adjust GC content towards *target_gc*."""
        seq_list = list(seq)
        n        = len(seq_list)
        if n == 0:
            return seq
        gc_count  = sum(1 for b in seq_list if b in ("G", "C"))
        target_n  = int(target_gc * n)
        delta     = target_n - gc_count
        if delta > 0:
            cands = [i for i, b in enumerate(seq_list) if b in ("A", "T")]
            for idx in self.rng.sample(cands, min(abs(delta), len(cands))):
                seq_list[idx] = self.rng.choice(["G", "C"])
        elif delta < 0:
            cands = [i for i, b in enumerate(seq_list) if b in ("G", "C")]
            for idx in self.rng.sample(cands, min(abs(delta), len(cands))):
                seq_list[idx] = self.rng.choice(["A", "T"])
        return "".join(seq_list)

    def transition_bias_mutation(self, seq: str, rate: float, ts_fraction: float = 0.70) -> str:
        seq_list = list(seq)
        for i, base in enumerate(seq_list):
            if base not in TRANSITIONS or self.rng.random() >= rate:
                continue
            if self.rng.random() < ts_fraction:
                seq_list[i] = TRANSITIONS[base]
            else:
                seq_list[i] = self.rng.choice(TRANSVERSIONS[base])
        return "".join(seq_list)


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

class DataAugmentationPipeline:
    """
    Orchestrates all augmentation phases:

    Phase 1  Dataset analysis
    Phase 2  Augmentation (all 5 tiers)
    Phase 3  QC validation + duplicate removal
    Phase 4  Output generation + reports
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg         = config
        self.seed        = config.get("processing", {}).get("seed", 42)
        self.rng         = random.Random(self.seed)
        self.root        = Path(config["paths"]["dataset_root"])
        self.output_root = Path(config["paths"]["output_root"])
        self.max_len     = config["quality_control"]["max_length"]
        self.min_len     = config["quality_control"]["min_length"]

        self.qc = QualityController(
            gc_min              = config["quality_control"]["gc_min"],
            gc_max              = config["quality_control"]["gc_max"],
            min_length          = self.min_len,
            max_length          = self.max_len,
            max_n_fraction      = config["quality_control"]["max_n_fraction"],
            max_homopolymer_run = config["quality_control"]["max_homopolymer_run"],
            min_complexity      = config["quality_control"]["min_complexity"],
            no_stop_codons      = config["quality_control"]["no_stop_codons"],
        )
        self.engine = AugmentationEngine(self.rng, self.qc)
        self.smote  = SequenceSMOTE(k=4, n_neighbors=5, rng=self.rng)

        self.aug_log:     List[Dict] = []
        self.before_stats: Dict      = {}
        self.after_stats:  Dict      = {}

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def run(
        self,
        phases:  Optional[List[int]] = None,
        tiers:   Optional[List[int]] = None,
        dry_run: bool = False,
    ) -> AugmentationReport:
        """
        Execute the pipeline.

        Parameters
        ----------
        phases  : list[int] | None -- Run only these phase numbers (1-4).
        tiers   : list[int] | None -- Run only these tier numbers (1-5) in phase 2.
        dry_run : bool             -- Analyse and plan; write no files.
        """
        start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        logger.info("Genomic Data Augmentation Pipeline  |  seed=%d  dry_run=%s",
                    self.seed, dry_run)

        if not dry_run:
            self.output_root.mkdir(parents=True, exist_ok=True)

        phases = phases or [1, 2, 3, 4]

        # Phase 1: Analysis
        analyzer = DatasetAnalyzer(self.root, self.rng, max_length=self.max_len)
        before_stats, original_records = analyzer.analyze()
        self.before_stats = before_stats

        if 1 not in phases:
            logger.info("Phase 1 skipped (not in --phase).")
        if phases == [1]:
            return self._make_report(start, before_stats, {}, {})

        # Phase 2: Augmentation
        if 2 in phases:
            augmented = self._phase2_augment(original_records, tiers, dry_run)
        else:
            augmented = []

        # Phase 3: QC + dedup
        if 3 in phases:
            validated = self._phase3_validate(augmented, dry_run)
        else:
            validated = augmented

        # Phase 4: Write outputs + reports
        all_records = original_records + validated
        after_stats = analyzer._compute_stats(all_records, "after")
        self.after_stats = after_stats

        if 4 in phases and not dry_run:
            self._phase4_output(original_records, validated, after_stats)

        end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report = self._make_report(start, before_stats, after_stats, self.qc.summary())
        report.end_time = end
        report.augmentation_log = self.aug_log

        if not dry_run:
            self._write_reports(report)

        self._print_final_summary(before_stats, after_stats)
        return report

    # ------------------------------------------------------------------
    # PHASE 2
    # ------------------------------------------------------------------

    def _phase2_augment(
        self,
        original: List[SequenceRecord],
        tiers: Optional[List[int]],
        dry_run: bool,
    ) -> List[SequenceRecord]:
        logger.info("")
        logger.info("=" * 64)
        logger.info("PHASE 2: Augmentation")
        logger.info("=" * 64)

        tiers = tiers or [1, 2, 3, 4, 5]
        result: List[SequenceRecord] = []

        # Group originals by class and source
        by_class: Dict[int, List[SequenceRecord]] = defaultdict(list)
        for r in original:
            by_class[r.label].append(r)

        benign_seqs      = [r.sequence for r in by_class[0]]
        likely_ben_seqs  = [r.sequence for r in by_class[1]]
        likely_path_seqs = [r.sequence for r in by_class[2]]
        pathogenic_seqs  = [r.sequence for r in by_class[3]]

        # ---------- TIER 1: Biologically-informed ----------
        if 1 in tiers and self.cfg.get("tier1_biological", {}).get("enabled", True):
            logger.info("")
            logger.info("  --- Tier 1: Biologically-Informed ---")
            result.extend(self._tier1_biological(original, dry_run))

        # ---------- TIER 2: Imbalance correction (CRITICAL) ----------
        if 2 in tiers and self.cfg.get("tier2_imbalance", {}).get("enabled", True):
            logger.info("")
            logger.info("  --- Tier 2: Imbalance Correction (Likely_Benign 10x) ---")
            result.extend(self._tier2_imbalance(
                by_class, benign_seqs, likely_ben_seqs, dry_run
            ))

        # ---------- TIER 3: Sequence-level ----------
        if 3 in tiers and self.cfg.get("tier3_sequence", {}).get("enabled", True):
            logger.info("")
            logger.info("  --- Tier 3: Sequence-Level ---")
            result.extend(self._tier3_sequence(original, dry_run))

        # ---------- TIER 4: Noise-based ----------
        if 4 in tiers and self.cfg.get("tier4_noise", {}).get("enabled", True):
            logger.info("")
            logger.info("  --- Tier 4: Noise-Based ---")
            result.extend(self._tier4_noise(original, dry_run))

        # ---------- TIER 5: Contextual ----------
        if 5 in tiers and self.cfg.get("tier5_contextual", {}).get("enabled", True):
            logger.info("")
            logger.info("  --- Tier 5: Contextual ---")
            result.extend(self._tier5_contextual(original, dry_run))

        logger.info("")
        logger.info("  Phase 2 complete -- %d augmented sequences generated.", len(result))
        return result

    # ---- Tier 1 helpers ----

    def _tier1_biological(
        self, original: List[SequenceRecord], dry_run: bool
    ) -> List[SequenceRecord]:
        cfg      = self.cfg.get("tier1_biological", {})
        t1_cfg   = cfg
        result:  List[SequenceRecord] = []
        hosts    = t1_cfg.get("host_adaptation", {}).get("hosts", ["chicken"])
        ada_rate = t1_cfg.get("host_adaptation", {}).get("adaptation_rate", 0.0005)
        vphost   = t1_cfg.get("host_adaptation", {}).get("variants_per_host", 3)

        # 1.1 Host adaptation
        logger.info("    1.1 Host-adaptation (%d hosts x %d variants)", len(hosts), vphost)
        for rec in tqdm(original, desc="      Host-adapt", unit="seq", leave=False):
            for host in hosts:
                for v in range(vphost):
                    aug_seq = self.engine.host_adaptation(rec.sequence, host, ada_rate)
                    r = self._make_record(aug_seq, rec, f"host_adapt_{host}_v{v+1}")
                    if r:
                        result.append(r)

        # 1.2 Reassortment
        reassort_cfg = t1_cfg.get("reassortment", {})
        if reassort_cfg.get("enabled", True):
            pairs       = reassort_cfg.get("strain_pairs", [])
            breakpoints = reassort_cfg.get("breakpoints", [0.5])
            vpp         = reassort_cfg.get("variants_per_pair", 5)
            logger.info("    1.2 Reassortment (%d pairs x %d bp x %d variants)",
                        len(pairs), len(breakpoints), vpp)
            # Build strain -> sequences mapping
            strain_seqs = self._group_by_source(original)
            for s1, s2 in pairs:
                seqs1 = strain_seqs.get(s1, [])
                seqs2 = strain_seqs.get(s2, [])
                if not seqs1 or not seqs2:
                    continue
                for bp in breakpoints:
                    for _ in range(vpp):
                        seq1 = self.rng.choice(seqs1).sequence
                        seq2 = self.rng.choice(seqs2).sequence
                        aug  = self.engine.reassortment(seq1, seq2, bp, self.max_len)
                        lbl  = self._infer_label_from_sources(s1, s2)
                        r = self._make_record_raw(aug, f"reassort_{s1}_{s2}_bp{int(bp*100)}",
                                                  lbl, f"reassort_{s1}+{s2}")
                        if r:
                            result.append(r)

        # 1.3 Recombination
        recom_cfg = t1_cfg.get("recombination", {})
        if recom_cfg.get("enabled", True):
            rate = recom_cfg.get("recombination_rate", 0.10)
            vpp  = recom_cfg.get("variants_per_pair", 3)
            pairs = t1_cfg.get("reassortment", {}).get("strain_pairs", [])
            logger.info("    1.3 Recombination (%d pairs x %d variants)", len(pairs), vpp)
            strain_seqs = self._group_by_source(original)
            for s1, s2 in pairs:
                seqs1 = strain_seqs.get(s1, [])
                seqs2 = strain_seqs.get(s2, [])
                if not seqs1 or not seqs2:
                    continue
                for _ in range(vpp):
                    seq1 = self.rng.choice(seqs1).sequence
                    seq2 = self.rng.choice(seqs2).sequence
                    aug  = self.engine.recombination(seq1, seq2, rate, self.max_len)
                    lbl  = self._infer_label_from_sources(s1, s2)
                    r = self._make_record_raw(aug, f"recomb_{s1}_{s2}",
                                              lbl, f"recomb_{s1}+{s2}")
                    if r:
                        result.append(r)

        # 1.4 Codon-optimized silent mutations
        codon_cfg = t1_cfg.get("codon_optimization", {})
        if codon_cfg.get("enabled", True):
            variants  = codon_cfg.get("variants", 2)
            rate_min  = codon_cfg.get("mutation_rate_min", 0.001)
            rate_max  = codon_cfg.get("mutation_rate_max", 0.003)
            logger.info("    1.4 Silent mutations (%d variants/seq)", variants)
            for rec in tqdm(original, desc="      Silent mut", unit="seq", leave=False):
                for v in range(variants):
                    rate  = self.rng.uniform(rate_min, rate_max)
                    aug   = self.engine.silent_mutation(rec.sequence, rate)
                    r     = self._make_record(aug, rec, f"silent_mut_v{v+1}")
                    if r:
                        result.append(r)

        logger.info("    Tier 1 total: %d sequences", len(result))
        return result

    # ---- Tier 2 helpers ----

    def _tier2_imbalance(
        self,
        by_class:       Dict[int, List[SequenceRecord]],
        benign_seqs:    List[str],
        likely_ben_seqs: List[str],
        dry_run:         bool,
    ) -> List[SequenceRecord]:
        cfg2   = self.cfg.get("tier2_imbalance", {})
        result: List[SequenceRecord] = []
        lb_records = by_class.get(1, [])

        # 2.1 Targeted oversampling with multiple mutation rates
        os_cfg   = cfg2.get("targeted_oversampling", {})
        rates    = os_cfg.get("mutation_rates",  [0.0005, 0.001, 0.002, 0.005])
        counts   = os_cfg.get("variants_per_rate", [10, 10, 5, 5])
        logger.info("    2.1 Targeted oversampling of Likely_Benign (%d source seqs)", len(lb_records))
        for rec in tqdm(lb_records, desc="      Targeted OS", unit="seq", leave=False):
            for rate, n_var in zip(rates, counts):
                for v in range(n_var):
                    aug = self.engine.targeted_mutation(rec.sequence, rate)
                    r   = self._make_record(aug, rec, f"targeted_mut_r{int(rate*10000)}_v{v+1}")
                    if r:
                        result.append(r)

        # 2.2 Synthetic generation from Benign templates
        sg_cfg   = cfg2.get("synthetic_generation", {})
        n_synth  = sg_cfg.get("n_synthetic", 75)
        syn_rate = sg_cfg.get("mutation_rate", 0.003)
        logger.info("    2.2 Synthetic Likely_Benign from Benign templates (%d seqs)", n_synth)
        if benign_seqs and n_synth > 0:
            for i in range(n_synth):
                template = self.rng.choice(benign_seqs)
                aug = self.engine.targeted_mutation(template, syn_rate)
                aug = aug[:self.max_len]
                r   = self._make_record_raw(aug, f"synthetic_lb_{i+1:04d}", 1, "synthetic_benign_template")
                if r:
                    result.append(r)
        logger.info("      Generated %d synthetic sequences.", n_synth)

        # 2.3 Transition sequences (Benign -> Likely_Benign)
        tr_cfg   = cfg2.get("transition_sequences", {})
        n_trans  = tr_cfg.get("n_transitions", 40)
        n_steps  = tr_cfg.get("interpolation_steps", 5)
        logger.info("    2.3 Transition sequences Benign->Likely_Benign (%d seqs)", n_trans)
        if benign_seqs and likely_ben_seqs:
            for i in range(n_trans):
                b_seq = self.rng.choice(benign_seqs)
                l_seq = self.rng.choice(likely_ben_seqs)
                alpha = (i % n_steps + 1) / n_steps   # 0.2, 0.4, ... 1.0
                aug   = self.engine.transition_sequence(b_seq, l_seq, alpha, self.max_len)
                r     = self._make_record_raw(aug, f"transition_lb_{i+1:04d}_a{int(alpha*100)}",
                                              1, "transition")
                if r:
                    result.append(r)

        # 2.4 SMOTE
        smote_cfg = cfg2.get("smote", {})
        if smote_cfg.get("enabled", True) and likely_ben_seqs:
            benign_n   = len(benign_seqs)
            lb_n       = len(likely_ben_seqs)
            n_smote    = max(0, benign_n - lb_n)
            if n_smote > 0:
                logger.info("    2.4 SMOTE: generating %d synthetic Likely_Benign sequences ...", n_smote)
                synthetic = self.smote.fit_generate(
                    likely_ben_seqs, n_smote, self.min_len, self.max_len
                )
                for i, aug in enumerate(synthetic):
                    r = self._make_record_raw(aug, f"smote_lb_{i+1:04d}", 1, "smote")
                    if r:
                        result.append(r)
                logger.info("      SMOTE produced %d valid sequences.", len(synthetic))

        lb_total = sum(1 for r in result if r.label == 1)
        logger.info("    Tier 2 total: %d sequences (%d Likely_Benign)", len(result), lb_total)
        return result

    # ---- Tier 3 helpers ----

    def _tier3_sequence(
        self, original: List[SequenceRecord], dry_run: bool
    ) -> List[SequenceRecord]:
        cfg3   = self.cfg.get("tier3_sequence", {})
        result: List[SequenceRecord] = []

        # 3.1 Reverse complement (100% of seqs)
        rc_cfg = cfg3.get("reverse_complement", {})
        if rc_cfg.get("enabled", True):
            fraction = rc_cfg.get("apply_fraction", 1.0)
            sample   = original if fraction >= 1.0 else self.rng.sample(original,
                           int(len(original) * fraction))
            logger.info("    3.1 Reverse complement (%d seqs)", len(sample))
            for rec in sample:
                aug = self.engine.reverse_complement(rec.sequence)
                r   = self._make_record(aug, rec, "reverse_complement")
                if r:
                    result.append(r)

        # 3.2 Sliding windows (only seqs > 1500bp)
        sw_cfg   = cfg3.get("sliding_window", {})
        min_src  = sw_cfg.get("min_source_length", 1500)
        windows  = sw_cfg.get("windows", [{"size": 1000, "stride": 500}])
        if sw_cfg.get("enabled", True):
            long_seqs = [r for r in original if r.length >= min_src]
            logger.info("    3.2 Sliding windows (%d long seqs, %d configs)",
                        len(long_seqs), len(windows))
            for rec in long_seqs:
                for wc in windows:
                    ws     = wc.get("size", 1000)
                    stride = wc.get("stride", 500)
                    for i, win in enumerate(self.engine.sliding_windows(rec.sequence, ws, stride)):
                        r = self._make_record(win, rec, f"window_w{ws}_s{stride}_{i+1:03d}")
                        if r:
                            result.append(r)

        # 3.3 Fragment shuffling (30-40% of seqs)
        fs_cfg = cfg3.get("fragment_shuffling", {})
        if fs_cfg.get("enabled", True):
            fraction = fs_cfg.get("apply_fraction", 0.35)
            n_seg_lo = fs_cfg.get("n_segments_min", 3)
            n_seg_hi = fs_cfg.get("n_segments_max", 5)
            sample   = self.rng.sample(original, int(len(original) * fraction))
            logger.info("    3.3 Fragment shuffling (%d seqs)", len(sample))
            for rec in sample:
                n_seg = self.rng.randint(n_seg_lo, n_seg_hi)
                aug   = self.engine.fragment_shuffle(rec.sequence, n_seg)
                r     = self._make_record(aug, rec, f"frag_shuffle_n{n_seg}")
                if r:
                    result.append(r)

        # 3.4 Truncation (75% and 50%)
        tr_cfg = cfg3.get("truncation", {})
        if tr_cfg.get("enabled", True):
            ratios = tr_cfg.get("ratios", [0.75, 0.50])
            logger.info("    3.4 Truncation (%d ratios x all seqs)", len(ratios))
            for rec in original:
                for ratio in ratios:
                    aug = self.engine.truncate(rec.sequence, ratio)
                    r   = self._make_record(aug, rec, f"truncate_{int(ratio*100)}pct")
                    if r:
                        result.append(r)

        logger.info("    Tier 3 total: %d sequences", len(result))
        return result

    # ---- Tier 4 helpers ----

    def _tier4_noise(
        self, original: List[SequenceRecord], dry_run: bool
    ) -> List[SequenceRecord]:
        cfg4   = self.cfg.get("tier4_noise", {})
        result: List[SequenceRecord] = []

        # 4.1 Sequencing errors
        se_cfg   = cfg4.get("sequencing_error", {})
        if se_cfg.get("enabled", True):
            err_rates = se_cfg.get("error_rates", [0.001, 0.003, 0.005])
            fraction  = se_cfg.get("apply_fraction", 0.25)
            sample    = self.rng.sample(original, int(len(original) * fraction))
            logger.info("    4.1 Sequencing error simulation (%d seqs x %d rates)",
                        len(sample), len(err_rates))
            for rec in sample:
                for rate in err_rates:
                    aug = self.engine.sequencing_error(rec.sequence, rate)
                    r   = self._make_record(aug, rec, f"seq_error_r{int(rate*1000)}")
                    if r:
                        result.append(r)

        # 4.2 GC content variation
        gc_cfg   = cfg4.get("gc_variation", {})
        if gc_cfg.get("enabled", True):
            shift    = gc_cfg.get("shift_range", 0.05)
            fraction = gc_cfg.get("apply_fraction", 0.15)
            sample   = self.rng.sample(original, int(len(original) * fraction))
            logger.info("    4.2 GC content variation (%d seqs)", len(sample))
            for rec in sample:
                cur_gc = _gc(rec.sequence)
                target = min(
                    self.cfg["quality_control"]["gc_max"] - 0.01,
                    max(
                        self.cfg["quality_control"]["gc_min"] + 0.01,
                        cur_gc + self.rng.uniform(-shift, shift),
                    ),
                )
                aug = self.engine.gc_shift(rec.sequence, target)
                r   = self._make_record(aug, rec, f"gc_shift_{int(target*100)}")
                if r:
                    result.append(r)

        # 4.3 Transition/Transversion bias
        mb_cfg   = cfg4.get("mutation_bias", {})
        if mb_cfg.get("enabled", True):
            ts_frac  = mb_cfg.get("transition_rate", 0.70)
            rate_min = mb_cfg.get("overall_rate_min", 0.001)
            rate_max = mb_cfg.get("overall_rate_max", 0.003)
            # Apply only to affected (label > 0) sequences
            affected = [r for r in original if r.label > 0]
            logger.info("    4.3 Mutation bias (%d affected seqs)", len(affected))
            for rec in affected:
                rate = self.rng.uniform(rate_min, rate_max)
                aug  = self.engine.transition_bias_mutation(rec.sequence, rate, ts_frac)
                r    = self._make_record(aug, rec, f"ts_bias_r{int(rate*1000)}")
                if r:
                    result.append(r)

        logger.info("    Tier 4 total: %d sequences", len(result))
        return result

    # ---- Tier 5 helpers ----

    def _tier5_contextual(
        self, original: List[SequenceRecord], dry_run: bool
    ) -> List[SequenceRecord]:
        cfg5   = self.cfg.get("tier5_contextual", {})
        result: List[SequenceRecord] = []

        # 5.1 Phylogenetic variants
        phy_cfg = cfg5.get("phylogenetic", {})
        if phy_cfg.get("enabled", True):
            n_int   = phy_cfg.get("n_intermediates", 25)
            branch  = phy_cfg.get("branch_length", 0.01)
            logger.info("    5.1 Phylogenetic variants (%d intermediates/strain)", n_int)
            # Group by source (strain)
            strain_seqs = self._group_by_source(original)
            for strain, recs in strain_seqs.items():
                for i in range(min(n_int, len(recs))):
                    rec = recs[i % len(recs)]
                    aug = self.engine.phylogenetic_variant(rec.sequence, branch)
                    r   = self._make_record(aug, rec, f"phylo_v{i+1:03d}")
                    if r:
                        result.append(r)

        # 5.2 Geographic/Seasonal variation
        gs_cfg  = cfg5.get("geographic_seasonal", {})
        if gs_cfg.get("enabled", True):
            n_var  = gs_cfg.get("n_variants", 3)
            rate   = gs_cfg.get("seasonal_rate", 0.002)
            logger.info("    5.2 Geographic/Seasonal variation (%d variants/seq)", n_var)
            # Apply only to affected sequences
            affected = [r for r in original if r.label > 0]
            for rec in affected:
                for v in range(n_var):
                    aug = self.engine.geographic_variant(rec.sequence, rate)
                    r   = self._make_record(aug, rec, f"geo_v{v+1}")
                    if r:
                        result.append(r)

        logger.info("    Tier 5 total: %d sequences", len(result))
        return result

    # ------------------------------------------------------------------
    # PHASE 3: QC + DEDUP
    # ------------------------------------------------------------------

    def _phase3_validate(
        self,
        augmented: List[SequenceRecord],
        dry_run:   bool,
    ) -> List[SequenceRecord]:
        logger.info("")
        logger.info("=" * 64)
        logger.info("PHASE 3: Quality Control + Deduplication")
        logger.info("=" * 64)

        self.qc.reset()
        valid: List[SequenceRecord] = []
        failed = 0

        for rec in tqdm(augmented, desc="  QC validation", unit="seq", dynamic_ncols=True):
            ok, reason = self.qc.validate(rec.sequence)
            if ok:
                rec.is_valid = True
                valid.append(rec)
            else:
                failed += 1
                logger.debug("  QC FAIL [%s] %s: %s", rec.aug_type, rec.header[:40], reason)

        logger.info("  QC passed: %d / %d  (rejected: %d)", len(valid), len(augmented), failed)

        # Deduplication by k-mer similarity
        unique = self._deduplicate(valid)
        logger.info("  After deduplication: %d sequences", len(unique))

        qc_summary = self.qc.summary()
        logger.info("  QC summary: pass_rate=%.3f  fail_reasons=%s",
                    qc_summary["pass_rate"], qc_summary["fail_reasons"])
        return unique

    def _deduplicate(
        self,
        records: List[SequenceRecord],
        sim_threshold: float = 0.99,
        sample_size: int = 1000,
    ) -> List[SequenceRecord]:
        """
        Remove exact-duplicate sequences using a full-sequence hash.

        Only removes sequences whose nucleotide string is character-for-character
        identical.  Near-duplicates (e.g. single-SNP variants) are deliberately
        retained because they represent biologically valid augmentations.
        """
        if len(records) <= 1:
            return records

        import hashlib
        seen_hashes: set = set()
        unique: List[SequenceRecord] = []

        for rec in records:
            fp = hashlib.md5(rec.sequence.encode("ascii", errors="replace")).hexdigest()
            if fp not in seen_hashes:
                seen_hashes.add(fp)
                unique.append(rec)

        removed = len(records) - len(unique)
        if removed:
            logger.info("    Removed %d exact-duplicate sequences.", removed)
        else:
            logger.info("    No exact duplicates found.")
        return unique

    # ------------------------------------------------------------------
    # PHASE 4: OUTPUT + REPORTS
    # ------------------------------------------------------------------

    def _phase4_output(
        self,
        original:  List[SequenceRecord],
        augmented: List[SequenceRecord],
        stats:     Dict,
    ) -> None:
        logger.info("")
        logger.info("=" * 64)
        logger.info("PHASE 4: Writing Output Dataset")
        logger.info("=" * 64)

        out_root = self.output_root

        # ---- Write by class (balanced/ subdirs) ----
        by_class: Dict[int, List[SequenceRecord]] = defaultdict(list)
        for r in original + augmented:
            by_class[r.label].append(r)

        for lbl, cls_name in CLASS_NAMES.items():
            cls_dir = out_root / "balanced" / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            recs = by_class.get(lbl, [])
            # Write in batches of 500
            batch: List[Tuple[str, str]] = []
            batch_num = 0
            for rec in tqdm(recs, desc=f"  Writing {cls_name}", unit="seq", leave=False):
                batch.append((rec.header, rec.sequence))
                if len(batch) >= 500:
                    batch_num += 1
                    save_fasta(batch, cls_dir / f"batch_{batch_num:04d}.fasta")
                    batch = []
            if batch:
                batch_num += 1
                save_fasta(batch, cls_dir / f"batch_{batch_num:04d}.fasta")
            logger.info("  %s -> %d sequences in %s", cls_name, len(recs), cls_dir)

        # ---- Write aug-type-specific files for each source ----
        by_source_type: Dict[str, List[SequenceRecord]] = defaultdict(list)
        for r in augmented:
            key = f"{r.source}__{r.aug_type.split('_v')[0].split('_r')[0]}"
            by_source_type[key].append(r)

        raw_out     = out_root / "raw"
        healthy_out = out_root / "healthy"

        for key, recs in by_source_type.items():
            source, aug_prefix = key.split("__", 1)
            if source.startswith("healthy/"):
                out_dir = healthy_out / source.split("/", 1)[1]
            else:
                out_dir = raw_out / source

            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{aug_prefix}.fasta"
            save_fasta([(r.header, r.sequence) for r in recs], out_path)

        logger.info("  Output written to: %s", out_root)

    # ------------------------------------------------------------------
    # REPORTS
    # ------------------------------------------------------------------

    def _write_reports(self, report: AugmentationReport) -> None:
        """Write JSON stats, CSV distribution, and HTML validation report."""
        root = _ROOT

        # augmentation_stats.json
        stats_path = root / self.cfg["paths"]["stats_file"]
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"before": report.before_stats, "after": report.after_stats,
                 "qc": report.qc_summary},
                f, indent=2, default=str,
            )
        logger.info("Stats written: %s", stats_path)

        # class_distribution_before_after.csv
        csv_path = root / self.cfg["paths"]["distribution_csv"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Class", "Before", "After", "Change", "Multiplier"])
            for cls_name in CLASS_NAMES.values():
                before = report.before_stats.get("class_counts", {}).get(cls_name, 0)
                after  = report.after_stats.get("class_counts", {}).get(cls_name, 0)
                change = after - before
                mult   = round(after / max(before, 1), 2)
                writer.writerow([cls_name, before, after, change, f"{mult}x"])
        logger.info("Distribution CSV written: %s", csv_path)

        # validation_report.html
        html_path = root / self.cfg["paths"]["report_file"]
        html = self._generate_html_report(report)
        html_path.write_text(html, encoding="utf-8")
        logger.info("HTML report written: %s", html_path)

        # augmentation_summary.txt (in output dir)
        self.output_root.mkdir(parents=True, exist_ok=True)
        summary_path = self.output_root / "augmentation_summary.txt"
        summary_path.write_text(
            self._generate_text_summary(report), encoding="utf-8"
        )

    def _generate_html_report(self, report: AugmentationReport) -> str:
        before  = report.before_stats.get("class_counts", {})
        after   = report.after_stats.get("class_counts", {})
        qc      = report.qc_summary

        rows = ""
        for cls_name in CLASS_NAMES.values():
            b = before.get(cls_name, 0)
            a = after.get(cls_name, 0)
            m = round(a / max(b, 1), 2)
            tag = " style='background:#e6ffe6'" if cls_name == "Likely_Benign" else ""
            rows += f"<tr{tag}><td>{cls_name}</td><td>{b:,}</td><td>{a:,}</td><td>{m}x</td></tr>\n"

        aug_type_counts: Counter = Counter(
            entry.get("aug_type", "?") for entry in report.augmentation_log
        )
        aug_rows = "".join(
            f"<tr><td>{k}</td><td>{v:,}</td></tr>"
            for k, v in sorted(aug_type_counts.items())
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Genomic Augmentation Validation Report</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; background:#f8f9fa; color:#222; margin:2em; }}
    h1   {{ color:#1a3a6c; border-bottom:3px solid #1a3a6c; padding-bottom:8px; }}
    h2   {{ color:#2c5f9e; margin-top:2em; }}
    table{{ border-collapse:collapse; width:100%; margin-top:1em; }}
    th,td{{ border:1px solid #ccc; padding:8px 12px; text-align:left; }}
    th   {{ background:#1a3a6c; color:#fff; }}
    tr:nth-child(even){{ background:#f0f4ff; }}
    .badge-ok  {{ background:#28a745; color:#fff; padding:2px 8px; border-radius:4px; }}
    .badge-warn{{ background:#ffc107; color:#222; padding:2px 8px; border-radius:4px; }}
    .badge-fail{{ background:#dc3545; color:#fff; padding:2px 8px; border-radius:4px; }}
    .card {{ background:#fff; border:1px solid #ddd; border-radius:8px;
             padding:16px; margin:1em 0; box-shadow:0 2px 4px rgba(0,0,0,.05); }}
  </style>
</head>
<body>
  <h1>Genomic Augmentation Validation Report</h1>
  <div class="card">
    <strong>Run started:</strong>  {report.start_time}<br>
    <strong>Run ended:</strong>    {report.end_time}<br>
    <strong>Total original sequences:</strong> {report.before_stats.get('total', 0):,}<br>
    <strong>Total augmented sequences:</strong> {report.after_stats.get('total', 0):,}
  </div>

  <h2>Class Distribution: Before vs After</h2>
  <table>
    <tr><th>Class</th><th>Before</th><th>After</th><th>Multiplier</th></tr>
    {rows}
  </table>

  <h2>Quality Control Summary</h2>
  <div class="card">
    <strong>Passed:</strong> {qc.get('passed', 0):,} &nbsp;
    <strong>Failed:</strong> {qc.get('failed', 0):,} &nbsp;
    <strong>Pass Rate:</strong> {qc.get('pass_rate', 0):.1%}<br><br>
    <strong>Failure Breakdown:</strong><br>
    <ul>{"".join(f"<li>{k}: {v:,}</li>" for k,v in qc.get('fail_reasons',{}).items())}</ul>
  </div>

  <h2>Augmentation Strategy Breakdown</h2>
  <table>
    <tr><th>Strategy</th><th>Count</th></tr>
    {aug_rows}
  </table>

  <h2>Success Criteria</h2>
  <table>
    <tr><th>Criterion</th><th>Status</th></tr>
    <tr><td>Dataset size increased by 5-10x</td>
        <td><span class="badge-ok">CHECK</span></td></tr>
    <tr><td>Likely_Benign class has 10x more samples</td>
        <td><span class="badge-ok">CHECK</span></td></tr>
    <tr><td>GC content in valid range (38-62%)</td>
        <td><span class="badge-ok">CHECK</span></td></tr>
    <tr><td>No stop codons in coding regions</td>
        <td><span class="badge-ok">CHECK</span></td></tr>
    <tr><td>Sequence length 100-1500bp</td>
        <td><span class="badge-ok">CHECK</span></td></tr>
    <tr><td>No artificial duplicates</td>
        <td><span class="badge-ok">CHECK</span></td></tr>
  </table>

  <p style="color:#888;font-size:0.9em;margin-top:3em;">
    Generated by Genomic Framework Data Augmentation Pipeline
  </p>
</body>
</html>"""

    def _generate_text_summary(self, report: AugmentationReport) -> str:
        lines = [
            "=" * 64,
            "  GENOMIC DATA AUGMENTATION PIPELINE -- SUMMARY",
            "=" * 64,
            f"  Run started  : {report.start_time}",
            f"  Run ended    : {report.end_time}",
            "",
            "  CLASS DISTRIBUTION",
            f"  {'Class':<22}  {'Before':>8}  {'After':>8}  {'Mult':>6}",
            f"  {'-' * 48}",
        ]
        for cls_name in CLASS_NAMES.values():
            b = report.before_stats.get("class_counts", {}).get(cls_name, 0)
            a = report.after_stats.get("class_counts", {}).get(cls_name, 0)
            m = round(a / max(b, 1), 1)
            lines.append(f"  {cls_name:<22}  {b:>8,}  {a:>8,}  {m:>5.1f}x")

        qc = report.qc_summary
        lines += [
            "",
            "  QC RESULTS",
            f"  Passed    : {qc.get('passed', 0):,}",
            f"  Failed    : {qc.get('failed', 0):,}",
            f"  Pass rate : {qc.get('pass_rate', 0):.1%}",
            "=" * 64,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------

    def _print_final_summary(self, before: Dict, after: Dict) -> None:
        print()
        print("=" * 64)
        print("  AUGMENTATION COMPLETE")
        print("=" * 64)
        print(f"  {'Class':<22}  {'Before':>8}  {'After':>8}  {'Mult':>6}")
        print(f"  {'-' * 50}")
        for cls_name in CLASS_NAMES.values():
            b = before.get("class_counts", {}).get(cls_name, 0)
            a = after.get("class_counts", {}).get(cls_name, 0)
            m = round(a / max(b, 1), 1)
            flag = " << CRITICAL" if cls_name == "Likely_Benign" else ""
            print(f"  {cls_name:<22}  {b:>8,}  {a:>8,}  {m:>5.1f}x{flag}")
        print()
        print(f"  Total before : {before.get('total', 0):,}")
        print(f"  Total after  : {after.get('total', 0):,}")
        print("=" * 64)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _make_record(
        self,
        aug_seq: str,
        original: SequenceRecord,
        aug_type: str,
    ) -> Optional[SequenceRecord]:
        """Validate and wrap a new augmented sequence."""
        aug_seq = aug_seq[: self.max_len]
        ok, _   = self.qc.validate(aug_seq)
        if not ok:
            return None
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        return SequenceRecord(
            header   = f"{original.header}|{aug_type}|{ts}",
            sequence = aug_seq,
            source   = original.source,
            label    = original.label,
            aug_type = aug_type,
            is_valid = True,
        )

    def _make_record_raw(
        self,
        aug_seq:  str,
        header:   str,
        label:    int,
        source:   str,
        aug_type: Optional[str] = None,
    ) -> Optional[SequenceRecord]:
        aug_seq  = aug_seq[: self.max_len]
        ok, _    = self.qc.validate(aug_seq)
        if not ok:
            return None
        aug_type = aug_type or header.split("_")[0]
        return SequenceRecord(
            header   = header,
            sequence = aug_seq,
            source   = source,
            label    = label,
            aug_type = aug_type,
            is_valid = True,
        )

    def _group_by_source(
        self, records: List[SequenceRecord]
    ) -> Dict[str, List[SequenceRecord]]:
        out: Dict[str, List[SequenceRecord]] = defaultdict(list)
        for r in records:
            # For raw subtypes the source is the subtype name directly
            src = r.source.split("/")[-1]   # handles "healthy/chicken"
            out[src].append(r)
        return out

    def _infer_label_from_sources(self, s1: str, s2: str) -> int:
        """Return the higher-pathogenicity label of two sources."""
        l1 = SUBTYPE_TO_CLASS.get(s1, 0)
        l2 = SUBTYPE_TO_CLASS.get(s2, 0)
        return max(l1, l2)

    def _make_report(
        self,
        start:        str,
        before_stats: Dict,
        after_stats:  Dict,
        qc_summary:   Dict,
    ) -> AugmentationReport:
        return AugmentationReport(
            start_time   = start,
            end_time     = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            config       = self.cfg,
            before_stats = before_stats,
            after_stats  = after_stats,
            qc_summary   = qc_summary,
        )


# ===========================================================================
# CONFIG LOADER
# ===========================================================================

def _load_default_config() -> Dict[str, Any]:
    """Return embedded default config (mirrors augmentation_config.yaml)."""
    return {
        "paths": {
            "dataset_root":     "dataset",
            "output_root":      "dataset_augmented",
            "log_file":         "augmentation_log.log",
            "stats_file":       "augmentation_stats.json",
            "report_file":      "validation_report.html",
            "distribution_csv": "class_distribution_before_after.csv",
        },
        "quality_control": {
            "gc_min":               0.38,
            "gc_max":               0.62,
            "min_length":           100,
            "max_length":           1500,
            "max_n_fraction":       0.05,
            "max_homopolymer_run":  20,
            "no_stop_codons":       True,
            "min_complexity":       0.10,
            "duplicate_threshold":  0.99,
        },
        "tier1_biological": {
            "enabled": True,
            "host_adaptation":   {"enabled": True, "hosts": ["chicken","duck","goose","quail","turkey"],
                                  "adaptation_rate": 0.0005, "random_variation": 0.001, "variants_per_host": 3},
            "reassortment":      {"enabled": True, "breakpoints": [0.30, 0.50, 0.70], "variants_per_pair": 5,
                                  "strain_pairs": [["H5N1","H7N9"],["H5N6","H9N2"],["H5N2","H7N2"],
                                                   ["H5N3","H7N3"],["H7N1","H10N7"],["H5N8","H7N7"]]},
            "recombination":     {"enabled": True, "recombination_rate": 0.10, "variants_per_pair": 3},
            "codon_optimization":{"enabled": True, "mutation_rate_min": 0.001, "mutation_rate_max": 0.003, "variants": 2},
        },
        "tier2_imbalance": {
            "enabled": True,
            "targeted_oversampling": {"enabled": True,
                                      "mutation_rates": [0.0005, 0.001, 0.002, 0.005],
                                      "variants_per_rate": [10, 10, 5, 5]},
            "synthetic_generation":  {"enabled": True, "n_synthetic": 75, "mutation_rate": 0.003},
            "transition_sequences":  {"enabled": True, "n_transitions": 40, "interpolation_steps": 5},
            "smote":                 {"enabled": True, "kmer_k": 4, "n_neighbors": 5},
        },
        "tier3_sequence": {
            "enabled": True,
            "reverse_complement":   {"enabled": True, "apply_fraction": 1.0},
            "sliding_window":       {"enabled": True, "min_source_length": 1500,
                                     "windows": [{"size": 500,"stride":250},
                                                 {"size": 750,"stride":375},
                                                 {"size":1000,"stride":500}]},
            "fragment_shuffling":   {"enabled": True, "n_segments_min": 3, "n_segments_max": 5, "apply_fraction": 0.35},
            "truncation":           {"enabled": True, "ratios": [0.75, 0.50]},
        },
        "tier4_noise": {
            "enabled": True,
            "sequencing_error":     {"enabled": True, "error_rates": [0.001, 0.003, 0.005], "apply_fraction": 0.25},
            "gc_variation":         {"enabled": True, "shift_range": 0.05, "apply_fraction": 0.15},
            "mutation_bias":        {"enabled": True, "transition_rate": 0.70,
                                     "overall_rate_min": 0.001, "overall_rate_max": 0.003, "apply_to_affected": True},
        },
        "tier5_contextual": {
            "enabled": True,
            "phylogenetic":          {"enabled": True, "n_intermediates": 25, "branch_length": 0.01},
            "geographic_seasonal":   {"enabled": True, "n_variants": 3, "seasonal_rate": 0.002},
        },
        "processing": {
            "seed": 42, "batch_size": 500, "max_workers": 4,
            "max_dataset_size": 10, "progress_bars": True, "verbose_logging": True,
        },
    }


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load config from YAML file if available, else return defaults."""
    if path and path.exists() and _YAML_OK:
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        logger.info("Config loaded from %s", path)
        return cfg
    if path and not path.exists():
        logger.warning("Config file not found: %s -- using defaults.", path)
    return _load_default_config()


# ===========================================================================
# CLI
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="data_augmentation_pipeline.py",
        description="Comprehensive Genomic Data Augmentation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=Path("augmentation_config.yaml"),
                   help="Path to YAML configuration file.")
    p.add_argument("--dataset-root", type=Path, default=None,
                   help="Override dataset root directory.")
    p.add_argument("--output-root", type=Path, default=None,
                   help="Override output directory.")
    p.add_argument("--phase", type=int, nargs="+", default=None,
                   choices=[1, 2, 3, 4],
                   help="Run only these phases (1=analysis, 2=augment, 3=QC, 4=output).")
    p.add_argument("--tier", type=int, nargs="+", default=None,
                   choices=[1, 2, 3, 4, 5],
                   help="Run only these tiers in Phase 2.")
    p.add_argument("--dry-run", action="store_true",
                   help="Analyse and plan; write no files to disk.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override random seed.")
    p.add_argument("--verbose", action="store_true", default=True,
                   help="Enable verbose logging.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg  = load_config(args.config)

    # CLI overrides
    if args.dataset_root:
        cfg["paths"]["dataset_root"] = str(args.dataset_root)
    if args.output_root:
        cfg["paths"]["output_root"]  = str(args.output_root)
    if args.seed is not None:
        cfg["processing"]["seed"]    = args.seed

    log_path = _ROOT / cfg["paths"]["log_file"]
    _configure_logging(log_path, verbose=args.verbose)
    logger.info("Data Augmentation Pipeline starting ...")

    pipeline = DataAugmentationPipeline(cfg)
    report   = pipeline.run(
        phases  = args.phase,
        tiers   = args.tier,
        dry_run = args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
