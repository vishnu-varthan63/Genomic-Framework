"""
Genomic_Framework/pipeline/augmentor.py
========================================
Data Augmentation Pipeline for Poultry Genomic Sequences.

Project : Genomic Framework — Poultry Disease Classification
Dataset : Healthy (chicken / duck / goose / quail / turkey)
          vs Affected (H5N1, H5N2, H5N3, H5N6, H5N8, H5N9,
                       H7N1, H7N2, H7N3, H7N7, H7N9, H9N2, H10N7, H10N8)

Augmentation Categories
------------------------
1. Sequence-level   : reverse_complement, random_mutation, random_insertion,
                      random_deletion, sliding_window, sequence_shuffling
2. Biological       : simulate_reassortment, simulate_recombination,
                      host_adaptation_simulation, codon_optimization
3. Noise-based      : gc_content_shift, transition_bias, transversion_bias
4. Data-balancing   : mixup_augmentation, cutmix_augmentation
                      (SMOTE operates on embedding space; see create_balanced_dataset)

Output
------
* Augmented FASTA files written **alongside** original files with prefix
  ``aug_<strategy>_<original_filename>`` (e.g. ``aug_rc_batch_001.fasta``).
* A single ``augmentation_metadata.json`` placed at the dataset root.

Usage
-----
    from Genomic_Framework.pipeline.augmentor import GenomicAugmenter, AUGMENTATION_CONFIG
    aug = GenomicAugmenter(AUGMENTATION_CONFIG)
    metadata = aug.augment_all(dataset_root=Path("dataset"), seed=42)
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ===========================================================================
# CONSTANTS
# ===========================================================================

NUCLEOTIDES: List[str] = ["A", "C", "G", "T"]

#: Transition pairs (purine↔purine, pyrimidine↔pyrimidine)
TRANSITIONS: Dict[str, str] = {"A": "G", "G": "A", "C": "T", "T": "C"}

#: Transversion pairs (purine↔pyrimidine)
TRANSVERSIONS: Dict[str, List[str]] = {
    "A": ["C", "T"], "G": ["C", "T"],
    "C": ["A", "G"], "T": ["A", "G"],
}

#: Avian-optimised codon table (most frequent codon per amino acid in birds)
#: Each 3-nt codon → synonymous avian-preferred codon
AVIAN_CODON_TABLE: Dict[str, str] = {
    # Ala
    "GCT": "GCC", "GCA": "GCC", "GCG": "GCC", "GCC": "GCC",
    # Arg
    "CGT": "AGG", "CGC": "CGG", "CGA": "AGG", "CGG": "CGG", "AGA": "AGG", "AGG": "AGG",
    # Asn
    "AAT": "AAC", "AAC": "AAC",
    # Asp
    "GAT": "GAC", "GAC": "GAC",
    # Cys
    "TGT": "TGC", "TGC": "TGC",
    # Gln
    "CAA": "CAG", "CAG": "CAG",
    # Glu
    "GAA": "GAG", "GAG": "GAG",
    # Gly
    "GGT": "GGC", "GGA": "GGC", "GGG": "GGC", "GGC": "GGC",
    # His
    "CAT": "CAC", "CAC": "CAC",
    # Ile
    "ATT": "ATC", "ATA": "ATC", "ATC": "ATC",
    # Leu
    "TTA": "CTG", "TTG": "CTG", "CTT": "CTG", "CTC": "CTG", "CTA": "CTG", "CTG": "CTG",
    # Lys
    "AAA": "AAG", "AAG": "AAG",
    # Met
    "ATG": "ATG",
    # Phe
    "TTT": "TTC", "TTC": "TTC",
    # Pro
    "CCT": "CCC", "CCA": "CCC", "CCG": "CCC", "CCC": "CCC",
    # Ser
    "TCT": "AGC", "TCC": "AGC", "TCA": "AGC", "TCG": "AGC", "AGT": "AGC", "AGC": "AGC",
    # Stop
    "TAA": "TGA", "TAG": "TGA", "TGA": "TGA",
    # Thr
    "ACT": "ACC", "ACA": "ACC", "ACG": "ACC", "ACC": "ACC",
    # Trp
    "TGG": "TGG",
    # Tyr
    "TAT": "TAC", "TAC": "TAC",
    # Val
    "GTT": "GTC", "GTA": "GTC", "GTG": "GTC", "GTC": "GTC",
}

#: Host-specific mutation bias profiles (simulated)
#: Maps (host, nucleotide) → list of replacement nucleotides with weights
HOST_ADAPTATION_BIAS: Dict[str, Dict[str, List[Tuple[str, float]]]] = {
    "chicken": {
        "A": [("G", 0.55), ("C", 0.25), ("T", 0.20)],
        "G": [("A", 0.50), ("C", 0.30), ("T", 0.20)],
        "C": [("T", 0.55), ("A", 0.25), ("G", 0.20)],
        "T": [("C", 0.50), ("A", 0.30), ("G", 0.20)],
    },
    "duck": {
        "A": [("G", 0.40), ("C", 0.35), ("T", 0.25)],
        "G": [("A", 0.45), ("T", 0.30), ("C", 0.25)],
        "C": [("T", 0.40), ("G", 0.35), ("A", 0.25)],
        "T": [("C", 0.45), ("G", 0.30), ("A", 0.25)],
    },
    "goose": {
        "A": [("G", 0.45), ("T", 0.30), ("C", 0.25)],
        "G": [("A", 0.45), ("C", 0.35), ("T", 0.20)],
        "C": [("T", 0.50), ("A", 0.30), ("G", 0.20)],
        "T": [("C", 0.50), ("G", 0.25), ("A", 0.25)],
    },
    "quail": {
        "A": [("G", 0.50), ("C", 0.30), ("T", 0.20)],
        "G": [("A", 0.50), ("T", 0.25), ("C", 0.25)],
        "C": [("T", 0.55), ("A", 0.20), ("G", 0.25)],
        "T": [("C", 0.55), ("A", 0.25), ("G", 0.20)],
    },
    "turkey": {
        "A": [("G", 0.48), ("C", 0.28), ("T", 0.24)],
        "G": [("A", 0.48), ("C", 0.28), ("T", 0.24)],
        "C": [("T", 0.48), ("A", 0.28), ("G", 0.24)],
        "T": [("C", 0.48), ("G", 0.28), ("A", 0.24)],
    },
}

# ===========================================================================
# DEFAULT CONFIGURATION
# ===========================================================================

AUGMENTATION_CONFIG: Dict[str, Any] = {
    "healthy": {
        "strategies": [
            "reverse_complement",
            "random_mutation",
            "sliding_window",
            "sequence_shuffling",
            "gc_content_shift",
        ],
        "mutation_rate": 0.0005,
        "window_size": 1000,
        "stride": 500,
        "kmer_size": 6,
        "samples_per_original": 5,
    },
    "affected": {
        "strategies": [
            "random_mutation",
            "simulate_reassortment",
            "simulate_recombination",
            "random_insertion",
            "random_deletion",
            "transition_bias",
            "transversion_bias",
        ],
        "mutation_rate": 0.001,
        "recombination_rate": 0.1,
        "samples_per_original": 3,
    },
    "common": {
        "gc_range": (0.38, 0.62),
        "kmer_size": 6,
        "max_sequence_length": 1500,
        "min_sequence_length": 100,
        "transition_rate": 0.70,
        "transversion_rate": 0.30,
    },
}

# ===========================================================================
# DATA CLASSES
# ===========================================================================


@dataclass
class AugmentationRecord:
    """Single augmented-sequence metadata record."""

    original_file:    str
    original_header:  str
    strategy:         str
    output_file:      str
    original_length:  int
    augmented_length: int
    params:           Dict[str, Any] = field(default_factory=dict)
    timestamp:        str = ""


@dataclass
class AugmentationSummary:
    """Top-level augmentation run summary."""

    run_timestamp:    str
    dataset_root:     str
    total_original:   int
    total_augmented:  int
    records:          List[Dict] = field(default_factory=list)
    strategy_counts:  Dict[str, int] = field(default_factory=dict)
    class_counts:     Dict[str, int] = field(default_factory=dict)


# ===========================================================================
# GENOMIC AUGMENTER
# ===========================================================================


class GenomicAugmenter:
    """
    Production-grade data augmentation pipeline for poultry genomic sequences.

    Implements sequence-level, biological, noise-based, and data-balancing
    augmentation strategies for FASTA-format genomic datasets.

    Parameters
    ----------
    config : dict
        Augmentation configuration (see ``AUGMENTATION_CONFIG`` for schema).
    seed   : int
        Master random seed for reproducibility.

    Examples
    --------
    >>> aug = GenomicAugmenter(AUGMENTATION_CONFIG, seed=42)
    >>> meta = aug.augment_all(dataset_root=Path("dataset"))
    """

    def __init__(
        self,
        config: Dict[str, Any],
        seed: int = 42,
    ) -> None:
        self.config  = config
        self.seed    = seed
        self._rng    = random.Random(seed)
        self._common = config.get("common", {})

        self._max_len = self._common.get("max_sequence_length", 1500)
        self._min_len = self._common.get("min_sequence_length", 100)

        logger.info(
            "GenomicAugmenter initialised — seed=%d  max_len=%d",
            seed, self._max_len,
        )

    # ===================================================================
    # PUBLIC API — top-level orchestration
    # ===================================================================

    def augment_all(
        self,
        dataset_root: Path,
        dry_run: bool = False,
    ) -> AugmentationSummary:
        """
        Run the full augmentation pipeline over the entire dataset.

        Scans ``dataset/raw/{SUBTYPE}/`` and ``dataset/healthy/{species}/``
        and writes augmented FASTA files alongside the originals.

        Parameters
        ----------
        dataset_root : Path — Root directory containing ``raw/`` and ``healthy/``.
        dry_run      : bool — If True, log what would be done but write nothing.

        Returns
        -------
        :class:`AugmentationSummary`
        """
        dataset_root = Path(dataset_root)
        raw_dir     = dataset_root / "raw"
        healthy_dir = dataset_root / "healthy"

        if not raw_dir.exists():
            raise FileNotFoundError(f"raw/ directory not found: {raw_dir}")
        if not healthy_dir.exists():
            raise FileNotFoundError(f"healthy/ directory not found: {healthy_dir}")

        run_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        summary = AugmentationSummary(
            run_timestamp=run_ts,
            dataset_root=str(dataset_root.resolve()),
            total_original=0,
            total_augmented=0,
        )

        logger.info("=" * 64)
        logger.info("GenomicAugmenter — starting augmentation run")
        logger.info("  Dataset root : %s", dataset_root.resolve())
        logger.info("  Dry-run      : %s", dry_run)
        logger.info("=" * 64)

        # ---- Healthy species ----
        h_cfg = self.config.get("healthy", {})
        for species_dir in sorted(healthy_dir.iterdir()):
            if not species_dir.is_dir():
                continue
            records = self._augment_directory(
                src_dir=species_dir,
                cfg=h_cfg,
                category="healthy",
                dry_run=dry_run,
            )
            summary.records.extend(records)
            summary.total_augmented += len(records)
            logger.info(
                "  [healthy/%s]  +%d augmented sequences",
                species_dir.name, len(records),
            )

        # ---- Viral subtypes ----
        a_cfg = self.config.get("affected", {})
        for subtype_dir in sorted(raw_dir.iterdir()):
            if not subtype_dir.is_dir():
                continue
            records = self._augment_directory(
                src_dir=subtype_dir,
                cfg=a_cfg,
                category="affected",
                dry_run=dry_run,
            )
            summary.records.extend(records)
            summary.total_augmented += len(records)
            logger.info(
                "  [raw/%s]  +%d augmented sequences",
                subtype_dir.name, len(records),
            )

        # ---- Compute statistics ----
        summary.strategy_counts = Counter(
            r["strategy"] for r in summary.records
        )
        summary.total_original = sum(
            1 for r in summary.records if r.get("strategy") == "original"
        )

        # ---- Write metadata ----
        if not dry_run:
            meta_path = dataset_root / "augmentation_metadata.json"
            self._write_metadata(summary, meta_path)
            logger.info("Metadata written: %s", meta_path)

        logger.info(
            "Augmentation complete — %d sequences generated across %d strategies.",
            summary.total_augmented,
            len(summary.strategy_counts),
        )
        return summary

    def augment_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        augmentation_strategies: List[str],
        samples_per_original: int = 3,
        dry_run: bool = False,
    ) -> List[AugmentationRecord]:
        """
        Augment a single directory of FASTA files and write results to *output_dir*.

        Parameters
        ----------
        input_dir              : Path — Directory containing source ``.fasta`` files.
        output_dir             : Path — Output directory for augmented files.
        augmentation_strategies: list[str] — Strategy names to apply.
        samples_per_original   : int  — Augmented copies per original sequence.
        dry_run                : bool — Log only, no file writes.

        Returns
        -------
        list[:class:`AugmentationRecord`]
        """
        input_dir  = Path(input_dir)
        output_dir = Path(output_dir)

        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

        records: List[AugmentationRecord] = []
        fasta_files = sorted(
            list(input_dir.glob("*.fasta"))
            + list(input_dir.glob("*.fa"))
            + list(input_dir.glob("*.fna"))
        )

        if not fasta_files:
            logger.warning("No FASTA files found in %s", input_dir)
            return records

        for fpath in tqdm(
            fasta_files,
            desc=f"  Augmenting {input_dir.name}",
            unit="file",
            dynamic_ncols=True,
            leave=False,
        ):
            seqs = self.load_fasta(fpath)
            if not seqs:
                logger.debug("Empty file skipped: %s", fpath)
                continue

            aug_seqs_for_file: List[Tuple[str, str]] = []

            for header, seq in seqs:
                if len(seq) < self._min_len:
                    continue
                aug_seq_list = self._apply_all_strategies(
                    seq, augmentation_strategies, samples_per_original
                )
                for strategy, aug_seq in aug_seq_list:
                    aug_header = f"{header}|aug={strategy}"
                    aug_seqs_for_file.append((aug_header, aug_seq))
                    records.append(AugmentationRecord(
                        original_file=str(fpath),
                        original_header=header,
                        strategy=strategy,
                        output_file=str(output_dir / f"aug_{fpath.name}"),
                        original_length=len(seq),
                        augmented_length=len(aug_seq),
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    ))

            if aug_seqs_for_file and not dry_run:
                out_path = output_dir / f"aug_{fpath.name}"
                self.save_fasta(aug_seqs_for_file, out_path)

        return records

    def create_balanced_dataset(
        self,
        healthy_dir: Path,
        affected_dirs: List[Path],
        target_size: int,
        output_dir: Path,
        dry_run: bool = False,
    ) -> Dict[str, int]:
        """
        Create a class-balanced FASTA dataset by augmenting minority classes
        until each class reaches *target_size* sequences.

        Parameters
        ----------
        healthy_dir   : Path       — Directory with healthy FASTA files.
        affected_dirs : list[Path] — Directories for each viral subtype.
        target_size   : int        — Target number of sequences per class.
        output_dir    : Path       — Root output directory.
        dry_run       : bool       — Log only.

        Returns
        -------
        dict[str, int] — Achieved sequence count per directory.
        """
        healthy_dir = Path(healthy_dir)
        output_dir  = Path(output_dir)

        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

        achieved: Dict[str, int] = {}

        # ---- Healthy ----
        healthy_count = self._count_sequences_in_dir(healthy_dir)
        needed        = max(0, target_size - healthy_count)
        logger.info(
            "  healthy: %d existing → need %d more to reach %d",
            healthy_count, needed, target_size,
        )
        if needed > 0:
            h_out = output_dir / "healthy"
            self._balance_directory(
                src_dir=healthy_dir,
                out_dir=h_out,
                n_needed=needed,
                strategies=self.config.get("healthy", {}).get(
                    "strategies", ["random_mutation", "reverse_complement"]
                ),
                dry_run=dry_run,
            )
        achieved["healthy"] = min(healthy_count + needed, target_size)

        # ---- Viral subtypes ----
        for aff_dir in affected_dirs:
            aff_dir = Path(aff_dir)
            count   = self._count_sequences_in_dir(aff_dir)
            needed  = max(0, target_size - count)
            logger.info(
                "  %s: %d existing → need %d more to reach %d",
                aff_dir.name, count, needed, target_size,
            )
            if needed > 0:
                a_out = output_dir / aff_dir.name
                self._balance_directory(
                    src_dir=aff_dir,
                    out_dir=a_out,
                    n_needed=needed,
                    strategies=self.config.get("affected", {}).get(
                        "strategies", ["random_mutation"]
                    ),
                    dry_run=dry_run,
                )
            achieved[aff_dir.name] = min(count + needed, target_size)

        logger.info(
            "Balancing complete. Achieved counts: %s", achieved
        )
        return achieved

    # ===================================================================
    # FASTA I/O
    # ===================================================================

    @staticmethod
    def load_fasta(filepath: Path) -> List[Tuple[str, str]]:
        """
        Parse a FASTA / FNA file into (header, sequence) pairs.

        Parameters
        ----------
        filepath : Path — Path to the ``.fasta`` or ``.fna`` file.

        Returns
        -------
        list[tuple[str, str]] — ``[(header_without_gt, sequence), …]``
        """
        filepath = Path(filepath)
        records:  List[Tuple[str, str]] = []
        header:   Optional[str]         = None
        buf:      List[str]             = []

        try:
            with filepath.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.rstrip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        if header is not None and buf:
                            records.append((header, "".join(buf).upper()))
                        header = line[1:].strip()
                        buf    = []
                    else:
                        # Keep only valid nucleotide characters
                        cleaned = re.sub(r"[^ACGTNacgtn]", "N", line)
                        buf.append(cleaned.upper())

                if header is not None and buf:
                    records.append((header, "".join(buf).upper()))

        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Failed to read %s: %s", filepath, exc)

        return records

    @staticmethod
    def save_fasta(
        sequences: List[Tuple[str, str]],
        output_path: Path,
        line_width: int = 60,
    ) -> None:
        """
        Write (header, sequence) pairs to a FASTA file.

        Parameters
        ----------
        sequences   : list[tuple[str, str]] — (header, sequence) pairs.
        output_path : Path                  — Output file path (created if absent).
        line_width  : int                   — Characters per sequence line (60 is standard).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as fh:
            for header, seq in sequences:
                fh.write(f">{header}\n")
                for i in range(0, len(seq), line_width):
                    fh.write(seq[i: i + line_width] + "\n")

    # ===================================================================
    # 1. SEQUENCE-LEVEL AUGMENTATIONS
    # ===================================================================

    def reverse_complement(self, sequence: str) -> str:
        """
        Generate the reverse complement of *sequence*.

        Parameters
        ----------
        sequence : str — Nucleotide string (A/C/G/T/N).

        Returns
        -------
        str — Reverse complement sequence.
        """
        comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
        return "".join(comp.get(b, "N") for b in reversed(sequence.upper()))

    def random_mutation(
        self, sequence: str, mutation_rate: float = 0.001
    ) -> str:
        """
        Introduce random point mutations (SNPs) at *mutation_rate* per site.

        Parameters
        ----------
        sequence      : str   — Input nucleotide sequence.
        mutation_rate : float — Probability of mutation per nucleotide (default 0.001).

        Returns
        -------
        str — Mutated sequence (same length as input).
        """
        seq_list = list(sequence)
        for i, base in enumerate(seq_list):
            if base not in ("A", "C", "G", "T"):
                continue
            if self._rng.random() < mutation_rate:
                alts = [n for n in NUCLEOTIDES if n != base]
                seq_list[i] = self._rng.choice(alts)
        return "".join(seq_list)

    def random_insertion(
        self, sequence: str, insert_rate: float = 0.001
    ) -> str:
        """
        Insert random nucleotides at random positions.

        Parameters
        ----------
        sequence    : str   — Input nucleotide sequence.
        insert_rate : float — Expected fraction of positions that receive an insertion.

        Returns
        -------
        str — Sequence with insertions (length ≥ original length).
        """
        n_insert   = max(1, int(len(sequence) * insert_rate))
        seq_list   = list(sequence)
        for _ in range(n_insert):
            pos = self._rng.randint(0, len(seq_list))
            seq_list.insert(pos, self._rng.choice(NUCLEOTIDES))
        result = "".join(seq_list)
        return result[: self._max_len]

    def random_deletion(
        self, sequence: str, delete_rate: float = 0.001
    ) -> str:
        """
        Delete random nucleotides from *sequence*.

        Parameters
        ----------
        sequence    : str   — Input nucleotide sequence.
        delete_rate : float — Expected fraction of positions deleted.

        Returns
        -------
        str — Sequence with deletions (length ≤ original length).
        """
        n_delete = max(1, int(len(sequence) * delete_rate))
        seq_list = list(sequence)
        for _ in range(n_delete):
            if len(seq_list) <= self._min_len:
                break
            pos = self._rng.randint(0, len(seq_list) - 1)
            del seq_list[pos]
        return "".join(seq_list)

    def sliding_window(
        self,
        sequence: str,
        window_size: int = 1000,
        stride: int = 500,
    ) -> List[str]:
        """
        Extract overlapping subsequence windows.

        Parameters
        ----------
        sequence    : str — Input nucleotide sequence.
        window_size : int — Length of each window (characters).
        stride      : int — Step size between windows.

        Returns
        -------
        list[str] — All windows; may be empty if sequence is shorter than *window_size*.
        """
        windows: List[str] = []
        n = len(sequence)
        if n < window_size:
            return [sequence]       # return the whole sequence as one window
        for start in range(0, n - window_size + 1, stride):
            windows.append(sequence[start: start + window_size])
        return windows

    def sequence_shuffling(
        self, sequence: str, kmer_size: int = 6
    ) -> str:
        """
        Shuffle *sequence* at the k-mer level to preserve local composition
        while disrupting long-range order.

        Parameters
        ----------
        sequence  : str — Input nucleotide sequence.
        kmer_size : int — k-mer word size (default 6).

        Returns
        -------
        str — k-mer-shuffled sequence of the same length.
        """
        kmers: List[str] = []
        n = len(sequence)
        for i in range(0, n - kmer_size + 1, kmer_size):
            kmers.append(sequence[i: i + kmer_size])
        # Capture tail
        tail = sequence[(n // kmer_size) * kmer_size:]
        self._rng.shuffle(kmers)
        result = "".join(kmers) + tail
        return result[:n]

    # ===================================================================
    # 2. BIOLOGICAL AUGMENTATIONS
    # ===================================================================

    def simulate_reassortment(
        self,
        seq1: str,
        seq2: str,
        breakpoint: float = 0.5,
    ) -> str:
        """
        Simulate viral segment reassortment between two sequences.

        Splices the 5' portion of *seq1* with the 3' portion of *seq2* at
        *breakpoint* (expressed as fraction of *seq1*'s length).

        Parameters
        ----------
        seq1       : str   — Donor sequence (5' end used).
        seq2       : str   — Donor sequence (3' end used).
        breakpoint : float — Fraction of *seq1* used (0 < breakpoint < 1).

        Returns
        -------
        str — Reassortant chimeric sequence.
        """
        bp  = max(1, int(len(seq1) * breakpoint))
        bp2 = max(0, len(seq2) - (len(seq1) - bp))
        return (seq1[:bp] + seq2[bp2:])[:self._max_len]

    def simulate_recombination(
        self,
        seq1: str,
        seq2: str,
        recombination_rate: float = 0.1,
    ) -> str:
        """
        Simulate homologous recombination by exchanging segments between
        *seq1* and *seq2* at multiple crossover points.

        Parameters
        ----------
        seq1               : str   — First parent sequence.
        seq2               : str   — Second parent sequence.
        recombination_rate : float — Expected fraction of crossover events per base.

        Returns
        -------
        str — Recombinant sequence derived primarily from *seq1*.
        """
        min_len = min(len(seq1), len(seq2))
        # Determine crossover positions
        n_crossovers = max(1, int(min_len * recombination_rate))
        positions    = sorted(self._rng.sample(range(1, min_len), k=min(n_crossovers, min_len - 1)))
        positions    = [0] + positions + [min_len]

        result  = []
        current = seq1    # start with seq1
        for i in range(len(positions) - 1):
            start, end = positions[i], positions[i + 1]
            result.append(current[start:end])
            current = seq2 if current is seq1 else seq1  # swap

        return ("".join(result) + seq1[min_len:])[:self._max_len]

    def host_adaptation_simulation(
        self,
        sequence: str,
        host: str = "chicken",
        adaptation_rate: float = 0.005,
    ) -> str:
        """
        Simulate viral adaptation to a specific avian host by biasing
        mutation probabilities according to host-specific synonymous
        substitution profiles.

        Parameters
        ----------
        sequence        : str   — Input viral sequence.
        host            : str   — Target host species: chicken | duck | goose | quail | turkey.
        adaptation_rate : float — Fraction of sites that receive a host-biased substitution.

        Returns
        -------
        str — Host-adapted sequence.
        """
        host = host.lower()
        if host not in HOST_ADAPTATION_BIAS:
            logger.warning("Unknown host '%s'; using 'chicken'.", host)
            host = "chicken"

        bias     = HOST_ADAPTATION_BIAS[host]
        seq_list = list(sequence)

        for i, base in enumerate(seq_list):
            if base not in bias:
                continue
            if self._rng.random() < adaptation_rate:
                options, weights = zip(*bias[base])
                # Weighted random choice
                r = self._rng.random()
                cumulative = 0.0
                for nuc, w in zip(options, weights):
                    cumulative += w
                    if r <= cumulative:
                        seq_list[i] = nuc
                        break

        return "".join(seq_list)

    def codon_optimization(
        self, sequence: str, target_organism: str = "avian"
    ) -> str:
        """
        Replace each codon with the most-frequent synonymous codon for the
        target organism (avian by default).

        Only applies to positions that are multiples of 3 (in-frame).
        The sequence length is preserved.

        Parameters
        ----------
        sequence         : str — Input nucleotide sequence (CDS recommended).
        target_organism  : str — Target organism key; currently only ``"avian"`` is supported.

        Returns
        -------
        str — Codon-optimised sequence.
        """
        result = []
        n      = len(sequence)
        for i in range(0, n - 2, 3):
            codon = sequence[i: i + 3].upper()
            optimised = AVIAN_CODON_TABLE.get(codon, codon)
            result.append(optimised)
        # Append any trailing nucleotides that don't form a complete codon
        tail_start = (n // 3) * 3
        result.append(sequence[tail_start:])
        return "".join(result)

    # ===================================================================
    # 3. NOISE-BASED AUGMENTATIONS
    # ===================================================================

    def gc_content_shift(
        self,
        sequence: str,
        target_gc: Optional[float] = None,
    ) -> str:
        """
        Shift the GC content of *sequence* towards *target_gc* by introducing
        targeted AT→GC or GC→AT synonymous-like substitutions.

        Parameters
        ----------
        sequence  : str            — Input nucleotide sequence.
        target_gc : float | None   — Desired GC fraction (0–1).
                                     If None, picks a random value from gc_range.

        Returns
        -------
        str — Sequence with adjusted GC content (same length).
        """
        gc_range = self._common.get("gc_range", (0.38, 0.62))
        if target_gc is None:
            target_gc = self._rng.uniform(*gc_range)

        seq_list = list(sequence)
        n        = len(seq_list)
        if n == 0:
            return sequence

        # Current GC count
        gc_count = sum(1 for b in seq_list if b in ("G", "C"))
        current_gc = gc_count / n
        target_count = int(target_gc * n)
        delta        = target_count - gc_count

        if delta == 0:
            return sequence

        if delta > 0:
            # Need more GC → replace AT with GC
            candidates = [i for i, b in enumerate(seq_list) if b in ("A", "T")]
            to_swap    = min(abs(delta), len(candidates))
            for idx in self._rng.sample(candidates, to_swap):
                seq_list[idx] = self._rng.choice(["G", "C"])
        else:
            # Need more AT → replace GC with AT
            candidates = [i for i, b in enumerate(seq_list) if b in ("G", "C")]
            to_swap    = min(abs(delta), len(candidates))
            for idx in self._rng.sample(candidates, to_swap):
                seq_list[idx] = self._rng.choice(["A", "T"])

        return "".join(seq_list)

    def transition_bias(
        self,
        sequence: str,
        trans_rate: float = 0.70,
        overall_rate: float = 0.005,
    ) -> str:
        """
        Introduce substitutions with a transition bias (purine↔purine,
        pyrimidine↔pyrimidine).

        Parameters
        ----------
        sequence     : str   — Input nucleotide sequence.
        trans_rate   : float — Fraction of mutations that are transitions (default 0.70).
        overall_rate : float — Total fraction of sites mutated.

        Returns
        -------
        str — Sequence with transition-biased mutations.
        """
        seq_list = list(sequence)
        for i, base in enumerate(seq_list):
            if base not in TRANSITIONS:
                continue
            if self._rng.random() < overall_rate:
                if self._rng.random() < trans_rate:
                    seq_list[i] = TRANSITIONS[base]
                else:
                    seq_list[i] = self._rng.choice(TRANSVERSIONS[base])
        return "".join(seq_list)

    def transversion_bias(
        self,
        sequence: str,
        transv_rate: float = 0.70,
        overall_rate: float = 0.005,
    ) -> str:
        """
        Introduce substitutions with a transversion bias (purine↔pyrimidine).

        Parameters
        ----------
        sequence     : str   — Input nucleotide sequence.
        transv_rate  : float — Fraction of mutations that are transversions (default 0.70).
        overall_rate : float — Total fraction of sites mutated.

        Returns
        -------
        str — Sequence with transversion-biased mutations.
        """
        seq_list = list(sequence)
        for i, base in enumerate(seq_list):
            if base not in TRANSVERSIONS:
                continue
            if self._rng.random() < overall_rate:
                if self._rng.random() < transv_rate:
                    seq_list[i] = self._rng.choice(TRANSVERSIONS[base])
                else:
                    seq_list[i] = TRANSITIONS[base]
        return "".join(seq_list)

    # ===================================================================
    # 4. DATA-BALANCING AUGMENTATIONS
    # ===================================================================

    def mixup_augmentation(
        self,
        seq1: str,
        seq2: str,
        alpha: float = 0.2,
    ) -> str:
        """
        Sequence-space MixUp: produce a chimeric sequence by element-wise
        probabilistic selection of nucleotides from *seq1* and *seq2*.

        Each position is drawn from *seq1* with probability ``lam`` and from
        *seq2* with probability ``1 - lam``, where ``lam ~ Beta(alpha, alpha)``.

        Parameters
        ----------
        seq1  : str   — First sequence.
        seq2  : str   — Second sequence.
        alpha : float — Beta distribution concentration parameter.

        Returns
        -------
        str — MixUp sequence (same length as the shorter input).
        """
        # Sample lambda from Beta distribution approximation
        lam = self._beta_sample(alpha, alpha)
        n   = min(len(seq1), len(seq2))
        result = []
        for i in range(n):
            result.append(seq1[i] if self._rng.random() < lam else seq2[i])
        return "".join(result)

    def cutmix_augmentation(
        self,
        seq1: str,
        seq2: str,
        cut_ratio: float = 0.3,
    ) -> str:
        """
        Sequence-space CutMix: replace a contiguous segment of *seq1* with
        the corresponding segment from *seq2*.

        Parameters
        ----------
        seq1      : str   — Base sequence (most of this is retained).
        seq2      : str   — Donor sequence (one segment is extracted).
        cut_ratio : float — Fraction of sequence length to replace.

        Returns
        -------
        str — CutMix sequence (same length as *seq1*).
        """
        n       = len(seq1)
        cut_len = max(1, int(n * cut_ratio))
        start   = self._rng.randint(0, n - cut_len)
        end     = start + cut_len

        # Take the segment from seq2 at the same position (clamp if needed)
        s2_end   = min(end, len(seq2))
        s2_start = max(0, s2_end - cut_len)
        patch    = seq2[s2_start:s2_end]

        # Pad patch to cut_len if seq2 was shorter
        if len(patch) < cut_len:
            patch = patch + seq1[start + len(patch): end]

        result = seq1[:start] + patch + seq1[end:]
        return result[:n]

    # ===================================================================
    # APPLY AUGMENTATION (generic dispatch)
    # ===================================================================

    def apply_augmentation(
        self,
        sequence: str,
        augmentation_type: str,
        params: Optional[Dict[str, Any]] = None,
        partner_seq: Optional[str] = None,
    ) -> str:
        """
        Apply a single named augmentation to *sequence*.

        Parameters
        ----------
        sequence           : str            — Input nucleotide sequence.
        augmentation_type  : str            — Name of the augmentation (see table below).
        params             : dict | None    — Override default parameters.
        partner_seq        : str | None     — Second sequence (required for reassortment,
                                              recombination, mixup, cutmix).

        Supported augmentation_type values
        -----------------------------------
        ``reverse_complement``, ``random_mutation``, ``random_insertion``,
        ``random_deletion``, ``sliding_window``, ``sequence_shuffling``,
        ``simulate_reassortment``, ``simulate_recombination``,
        ``host_adaptation_simulation``, ``codon_optimization``,
        ``gc_content_shift``, ``transition_bias``, ``transversion_bias``,
        ``mixup``, ``cutmix``.

        Returns
        -------
        str — Augmented sequence string.
              For ``sliding_window``, returns the **first** window only; use
              :meth:`sliding_window` directly to get all windows.
        """
        params = params or {}
        seq    = sequence.upper()

        if augmentation_type == "reverse_complement":
            return self.reverse_complement(seq)

        if augmentation_type == "random_mutation":
            rate = params.get("mutation_rate", 0.001)
            return self.random_mutation(seq, rate)

        if augmentation_type == "random_insertion":
            rate = params.get("insert_rate", 0.001)
            return self.random_insertion(seq, rate)

        if augmentation_type == "random_deletion":
            rate = params.get("delete_rate", 0.001)
            return self.random_deletion(seq, rate)

        if augmentation_type == "sliding_window":
            ws      = params.get("window_size", 1000)
            stride  = params.get("stride", 500)
            windows = self.sliding_window(seq, ws, stride)
            return windows[0] if windows else seq

        if augmentation_type == "sequence_shuffling":
            k = params.get("kmer_size", self._common.get("kmer_size", 6))
            return self.sequence_shuffling(seq, k)

        if augmentation_type == "simulate_reassortment":
            if partner_seq is None:
                logger.warning("simulate_reassortment: no partner_seq; returning original.")
                return seq
            bp = params.get("breakpoint", 0.5)
            return self.simulate_reassortment(seq, partner_seq.upper(), bp)

        if augmentation_type == "simulate_recombination" or augmentation_type == "recombination":
            if partner_seq is None:
                logger.warning("simulate_recombination: no partner_seq; returning original.")
                return seq
            rate = params.get("recombination_rate", 0.1)
            return self.simulate_recombination(seq, partner_seq.upper(), rate)

        if augmentation_type == "host_adaptation_simulation":
            host = params.get("host", "chicken")
            rate = params.get("adaptation_rate", 0.005)
            return self.host_adaptation_simulation(seq, host, rate)

        if augmentation_type == "codon_optimization":
            org = params.get("target_organism", "avian")
            return self.codon_optimization(seq, org)

        if augmentation_type == "gc_content_shift":
            gc = params.get("target_gc", None)
            return self.gc_content_shift(seq, gc)

        if augmentation_type == "transition_bias":
            tr   = params.get("trans_rate", self._common.get("transition_rate", 0.70))
            rate = params.get("overall_rate", 0.005)
            return self.transition_bias(seq, tr, rate)

        if augmentation_type == "transversion_bias":
            tv   = params.get("transv_rate", self._common.get("transversion_rate", 0.30))
            rate = params.get("overall_rate", 0.005)
            return self.transversion_bias(seq, tv, rate)

        if augmentation_type == "mixup":
            if partner_seq is None:
                logger.warning("mixup: no partner_seq; returning original.")
                return seq
            a = params.get("alpha", 0.2)
            return self.mixup_augmentation(seq, partner_seq.upper(), a)

        if augmentation_type == "cutmix":
            if partner_seq is None:
                logger.warning("cutmix: no partner_seq; returning original.")
                return seq
            cr = params.get("cut_ratio", 0.3)
            return self.cutmix_augmentation(seq, partner_seq.upper(), cr)

        logger.warning("Unknown augmentation_type '%s'; returning original.", augmentation_type)
        return seq

    # ===================================================================
    # INTERNAL HELPERS
    # ===================================================================

    def _augment_directory(
        self,
        src_dir: Path,
        cfg: Dict[str, Any],
        category: str,
        dry_run: bool,
    ) -> List[Dict]:
        """
        Augment all FASTA files in *src_dir* and write files alongside originals.

        Returns list of record dicts suitable for JSON serialisation.
        """
        strategies           = cfg.get("strategies", ["random_mutation"])
        samples_per_original = cfg.get("samples_per_original", 3)
        mutation_rate        = cfg.get("mutation_rate", 0.001)
        window_size          = cfg.get("window_size", 1000)
        stride               = cfg.get("stride", 500)
        recomb_rate          = cfg.get("recombination_rate", 0.1)

        fasta_files = sorted(
            list(src_dir.glob("*.fasta"))
            + list(src_dir.glob("*.fa"))
            + list(src_dir.glob("*.fna"))
        )

        # ---- Dry-run: skip all file I/O; just report what would be done ----
        if dry_run:
            n_files    = len(fasta_files)
            # Estimate: assume ~100 seqs/file (conservative; actual may vary)
            est_seqs   = n_files * 100
            est_output = est_seqs * len(strategies) * samples_per_original
            logger.info(
                "  [DRY-RUN] %s: %d file(s), ~%d seqs → would generate ~%d "
                "augmented seqs across %d strategies",
                src_dir.name, n_files, est_seqs, est_output, len(strategies),
            )
            # Return lightweight placeholder records
            records: List[Dict] = []
            for fpath in fasta_files:
                for strategy in strategies:
                    for _ in range(samples_per_original):
                        records.append({
                            "original_file":   str(fpath),
                            "original_header": "<dry-run>",
                            "strategy":        strategy,
                            "output_file": str(
                                src_dir / f"aug_{strategy}_{fpath.name}"
                            ),
                            "original_length":  0,
                            "augmented_length": 0,
                            "category":         category,
                            "dry_run":          True,
                            "timestamp": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            ),
                        })
            return records

        all_records: List[Dict] = []

        for fpath in fasta_files:
            seqs = self.load_fasta(fpath)
            if not seqs:
                continue

            # Build per-strategy output buckets
            strategy_buckets: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

            for header, seq in seqs:
                if len(seq) < self._min_len:
                    continue

                for strategy in strategies:
                    for sample_idx in range(samples_per_original):
                        # For paired strategies, pick a random partner from the same file
                        partner = None
                        if strategy in (
                            "simulate_reassortment", "simulate_recombination",
                            "recombination", "mixup", "cutmix"
                        ):
                            _, partner = self._rng.choice(seqs)

                        params = {
                            "mutation_rate":      mutation_rate,
                            "window_size":        window_size,
                            "stride":             stride,
                            "recombination_rate": recomb_rate,
                        }

                        aug_seq = self.apply_augmentation(
                            seq, strategy, params=params, partner_seq=partner
                        )

                        # Enforce length constraints
                        aug_seq = aug_seq[: self._max_len]
                        if len(aug_seq) < self._min_len:
                            continue

                        aug_header = (
                            f"{header}|aug={strategy}|sample={sample_idx + 1}"
                        )
                        strategy_buckets[strategy].append((aug_header, aug_seq))

                        all_records.append({
                            "original_file":   str(fpath),
                            "original_header": header,
                            "strategy":        strategy,
                            "output_file": str(
                                src_dir / f"aug_{strategy}_{fpath.name}"
                            ),
                            "original_length":  len(seq),
                            "augmented_length": len(aug_seq),
                            "category":         category,
                            "timestamp": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            ),
                        })

            # Write one FASTA file per strategy per source file
            if not dry_run:
                for strategy, aug_seqs in strategy_buckets.items():
                    out_name = f"aug_{strategy}_{fpath.name}"
                    out_path = src_dir / out_name
                    self.save_fasta(aug_seqs, out_path)
                    logger.debug(
                        "  Wrote %d seqs → %s", len(aug_seqs), out_path
                    )

        return all_records

    def _apply_all_strategies(
        self,
        sequence: str,
        strategies: List[str],
        samples_per_original: int,
    ) -> List[Tuple[str, str]]:
        """
        Apply each strategy *samples_per_original* times and return
        ``[(strategy_name, augmented_seq), …]``.
        """
        results: List[Tuple[str, str]] = []
        for strategy in strategies:
            for _ in range(samples_per_original):
                aug = self.apply_augmentation(sequence, strategy)
                aug = aug[: self._max_len]
                if len(aug) >= self._min_len:
                    results.append((strategy, aug))
        return results

    def _count_sequences_in_dir(self, directory: Path) -> int:
        """Count total sequences across all FASTA files in *directory*."""
        total = 0
        for fpath in directory.glob("*.fasta"):
            total += len(self.load_fasta(fpath))
        for fpath in directory.glob("*.fna"):
            total += len(self.load_fasta(fpath))
        return total

    def _balance_directory(
        self,
        src_dir: Path,
        out_dir: Path,
        n_needed: int,
        strategies: List[str],
        dry_run: bool,
    ) -> None:
        """
        Generate *n_needed* augmented sequences from *src_dir* and write to *out_dir*.
        """
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        all_seqs: List[Tuple[str, str]] = []
        for fpath in sorted(src_dir.glob("*.fasta")) + sorted(src_dir.glob("*.fna")):
            all_seqs.extend(self.load_fasta(fpath))

        if not all_seqs:
            logger.warning("No sequences found in %s for balancing.", src_dir)
            return

        generated: List[Tuple[str, str]] = []
        i = 0
        while len(generated) < n_needed:
            header, seq = all_seqs[i % len(all_seqs)]
            strategy    = strategies[i % len(strategies)]
            aug_seq     = self.apply_augmentation(seq, strategy)
            aug_seq     = aug_seq[: self._max_len]
            if len(aug_seq) >= self._min_len:
                aug_header = f"{header}|aug={strategy}|balance={i}"
                generated.append((aug_header, aug_seq))
            i += 1

        if not dry_run:
            out_path = out_dir / f"balanced_{src_dir.name}.fasta"
            self.save_fasta(generated[:n_needed], out_path)
            logger.info(
                "  Wrote %d balancing sequences → %s",
                n_needed, out_path,
            )

    @staticmethod
    def _beta_sample(alpha: float, beta: float) -> float:
        """
        Sample from Beta(alpha, beta) using two Gamma deviates.
        Falls back to 0.5 for edge cases.
        """
        try:
            import math
            x = random.gammavariate(alpha, 1.0)
            y = random.gammavariate(beta,  1.0)
            return x / (x + y + 1e-8)
        except Exception:
            return 0.5

    @staticmethod
    def _write_metadata(
        summary: AugmentationSummary, path: Path
    ) -> None:
        """Serialise *summary* to a JSON file at *path*."""
        data = {
            "run_timestamp":   summary.run_timestamp,
            "dataset_root":    summary.dataset_root,
            "total_original":  summary.total_original,
            "total_augmented": summary.total_augmented,
            "strategy_counts": dict(summary.strategy_counts),
            "records":         summary.records,
        }
        path.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )

    def __repr__(self) -> str:  # noqa: D401
        strategies = list(self.config.get("healthy", {}).get("strategies", []))
        return (
            f"GenomicAugmenter(seed={self.seed}, "
            f"strategies={strategies})"
        )
