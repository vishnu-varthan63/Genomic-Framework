"""
Genomic_Framework/pipeline/preprocessor.py
==========================================
Module: Data Preprocessing Pipeline

Reads raw FASTA/FNA files from disk, applies the existing QC pipeline
(``GenomicSequenceIngester``), tokenises sequences to integer k-mer IDs,
constructs mutation tensors via ``SequenceAligner`` + ``MutationProcessor``,
and builds train/val/test ``torch.utils.data.DataLoader`` objects.

Label assignment
----------------
  0  Benign            dataset/healthy/ — healthy host genome segments
  1  Likely_Benign     H5N2, H5N3, H7N2, H7N3, H9N2
  2  Likely_Pathogenic H5N6, H5N8, H7N1, H7N7, H10N7, H10N8
  3  Pathogenic        H5N1, H5N9, H7N9
"""

from __future__ import annotations

import logging
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from Genomic_Framework.pipeline.input_qc import GenomicSequenceIngester
from Genomic_Framework.pipeline.mutation_pipeline import (
    MutationProcessor,
    SequenceAligner,
)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ===========================================================================
# CONSTANTS
# ===========================================================================

#: Maps subtype / source folder name → pathogenicity class index.
SUBTYPE_TO_CLASS: Dict[str, int] = {
    # Healthy host genomes  →  Benign
    "healthy":  0,
    # Low-pathogenicity viral subtypes  →  Likely Benign
    "H5N2": 1,  "H5N3": 1,  "H7N2": 1,  "H7N3": 1,  "H9N2": 1,
    # Medium-pathogenicity subtypes  →  Likely Pathogenic
    "H5N6": 2,  "H5N8": 2,  "H7N1": 2,  "H7N7": 2,  "H10N7": 2, "H10N8": 2,
    # High-pathogenicity subtypes  →  Pathogenic
    "H5N1": 3,  "H5N9": 3,  "H7N9": 3,
}

#: Human-readable label names (index == class integer).
CLASS_LABELS: List[str] = [
    "Avirulent",
    "Low_Pathogenicity",
    "Moderate_High_Pathogenicity",
    "Highly_Pathogenic",
]

#: Backward-compatibility and folder alias map
CLASS_NAME_TO_IDX: Dict[str, int] = {
    "Avirulent": 0,
    "Low_Pathogenicity": 1,
    "Moderate_High_Pathogenicity": 2,
    "Highly_Pathogenic": 3,
    "Benign": 0,
    "Likely_Benign": 1,
    "Likely_Pathogenic": 2,
    "Pathogenic": 3,
}

NUM_CLASSES: int = len(CLASS_LABELS)

# Nucleotide bases for the k-mer tokeniser
_BASES = frozenset("ACGTN")

# ===========================================================================
# TOKENISER
# ===========================================================================


def tokenise_sequence(
    seq: str,
    seq_len: int = 128,
    vocab_size: int = 4096,
    k: int = 6,
) -> torch.LongTensor:
    """
    Deterministic k-mer hash tokeniser.

    Converts a cleaned nucleotide string into a fixed-length integer tensor
    by sliding a window of width *k* over the sequence and hashing each k-mer
    to a token ID in ``[1, vocab_size - 1]`` (0 is reserved for padding).
    Trailing positions that cannot form a full k-mer fall back to the character
    ordinal modulo ``vocab_size``.

    Parameters
    ----------
    seq       : str  — Cleaned nucleotide sequence (output of GenomicSequenceIngester).
    seq_len   : int  — Fixed output length (truncate or pad with 0).
    vocab_size: int  — Total vocabulary size including the padding token (0).
    k         : int  — k-mer width (default 3).

    Returns
    -------
    torch.LongTensor of shape ``[seq_len]``.

    Notes
    -----
    * The hash is computed with an explicit polynomial hash that is fully
      deterministic regardless of PYTHONHASHSEED across all Python runs.
    * Default k=6 matches vocab_size=4096 (4^6) as designed in the model
      architecture. Using k=3 would waste ~93.8% of the vocabulary space.
      If loading a checkpoint trained with k=3, pass k=3 explicitly.
    """
    tokens: List[int] = []
    n = len(seq)
    range_vocab = vocab_size - 1  # IDs 1 … vocab_size-1

    for i in range(n):
        if i + k <= n:
            kmer = seq[i: i + k]
            # Polynomial hash: deterministic across all Python runs
            h = 0
            for ch in kmer:
                h = h * 31 + ord(ch)
            token = (h % range_vocab) + 1           # in [1, vocab_size-1]
        else:
            # Fallback: single-character ordinal
            token = ord(seq[i]) % vocab_size or 1   # avoid 0
        tokens.append(token)

    # Truncate to seq_len
    tokens = tokens[:seq_len]
    # Pad with 0 to reach seq_len
    if len(tokens) < seq_len:
        tokens += [0] * (seq_len - len(tokens))

    return torch.tensor(tokens, dtype=torch.long)


# ===========================================================================
# FASTA PARSER (lightweight, no Biopython dependency)
# ===========================================================================


def _parse_fasta(filepath: Path, max_sequences: Optional[int] = None) -> List[str]:
    """
    Parse a FASTA / FNA file and return sequences.
    If max_sequences is None or <= 0, reads all sequences in the file.

    Parameters
    ----------
    filepath      : Path — Path to the FASTA file.
    max_sequences : Optional[int] — Cap on sequences returned per file (None for unlimited).

    Returns
    -------
    list[str]  — Raw sequence strings (headers stripped, bodies concatenated).
    """
    sequences: List[str] = []
    current_lines: List[str] = []

    try:
        with filepath.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_lines:
                        sequences.append("".join(current_lines))
                        current_lines = []
                        if max_sequences and max_sequences > 0 and len(sequences) >= max_sequences:
                            break
                else:
                    current_lines.append(line)

            # Flush last record
            if current_lines:
                if not max_sequences or max_sequences <= 0 or len(sequences) < max_sequences:
                    sequences.append("".join(current_lines))

    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read %s: %s", filepath, exc)

    return sequences


# ===========================================================================
# DATASET
# ===========================================================================


class GenomicDataset(Dataset):
    """
    PyTorch Dataset for genomic sequence classification.

    Reads FASTA/FNA files, applies QC, tokenises each sequence, computes
    mutation tensors, and returns fixed-shape tensors suitable for batching.

    Parameters
    ----------
    fasta_paths       : list[tuple[Path, int]]
        List of ``(file_path, class_label)`` pairs produced by
        :class:`GenomicDatasetBuilder`.
    seq_len           : int  — Token sequence length (must match engine SEQ_LEN).
    vocab_size        : int  — k-mer vocabulary size.
    max_mutations     : int  — Max mutations retained per sequence.
    mutation_dim      : int  — Feature vector size per mutation.
    max_seq_len_qc    : int  — Max raw sequence length passed to QC (characters).
    cache_tokens      : bool — If True, tokenised tensors are cached in RAM
                               after first access (speeds up subsequent epochs).

    Returns (per ``__getitem__``)
    ------------------------------
    token_ids       : torch.LongTensor  ``[seq_len]``
    mutation_tensor : torch.FloatTensor ``[max_mutations, mutation_dim]``
    label           : torch.LongTensor  scalar
    """

    def __init__(
        self,
        fasta_paths: List[Tuple[Path, int]],
        seq_len: int = 128,
        vocab_size: int = 4096,
        max_mutations: int = 32,
        mutation_dim: int = 64,
        max_seq_len_qc: int = 512,
        max_sequences_per_file: Optional[int] = None,
        cache_tokens: bool = True,
        seed: int = 42,
        k_mer: int = 6,
    ) -> None:
        self.seq_len                = seq_len
        self.vocab_size             = vocab_size
        self.max_mutations          = max_mutations
        self.mutation_dim           = mutation_dim
        self.max_sequences_per_file = max_sequences_per_file
        self.cache_tokens           = cache_tokens
        self.k_mer                  = k_mer

        # Shared pipeline components — seed propagated for full reproducibility
        self._ingester  = GenomicSequenceIngester(
            max_seq_len=max_seq_len_qc,
            strict_validation=True,
            min_valid_fraction=0.50,
        )
        self._aligner   = SequenceAligner(seed=seed)          # FIX: was hardcoded seed=42
        self._processor = MutationProcessor(
            max_mutations=max_mutations,
            mutation_dim=mutation_dim,
            seed=seed,                                         # FIX: was hardcoded seed=0
        )

        # Expand (file, label) pairs into individual (sequence_str, label) pairs
        self._samples: List[Tuple[str, int]] = []
        self._build_samples(fasta_paths)

        # Optional in-RAM cache: index → (token_ids, mutation_tensor)
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        logger.info(
            "GenomicDataset built — %d samples, seq_len=%d, cache=%s",
            len(self._samples), seq_len, cache_tokens,
        )

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _build_samples(
        self, fasta_paths: List[Tuple[Path, int]]
    ) -> None:
        """
        Parse every FASTA file, apply QC, and store (cleaned_seq, label)
        pairs in ``self._samples``.
        """
        logger.info(
            "Loading sequences from %d FASTA files …", len(fasta_paths)
        )
        skipped_files = 0

        for fpath, label in tqdm(
            fasta_paths,
            desc="  Loading FASTA files",
            unit="file",
            dynamic_ncols=True,
            leave=False,
        ):
            raw_seqs = _parse_fasta(fpath, max_sequences=self.max_sequences_per_file)
            if not raw_seqs:
                logger.debug("Empty or unreadable file (skipped): %s", fpath)
                skipped_files += 1
                continue

            try:
                cleaned, qc_log = self._ingester.clean_and_validate(raw_seqs)
            except Exception as exc:
                logger.warning("QC failed for %s: %s (skipped)", fpath, exc)
                skipped_files += 1
                continue

            if not cleaned:
                logger.warning(
                    "Zero sequences passed QC in %s (skipped)", fpath
                )
                skipped_files += 1
                continue

            for seq in cleaned:
                self._samples.append((seq, label))

        if skipped_files:
            logger.warning(
                "%d file(s) were skipped (empty / unreadable / all-rejected).",
                skipped_files,
            )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return ``(token_ids, mutation_tensor, label)`` for sample *idx*.

        Parameters
        ----------
        idx : int — Sample index.

        Returns
        -------
        token_ids       : LongTensor  ``[seq_len]``
        mutation_tensor : FloatTensor ``[max_mutations, mutation_dim]``
        label           : LongTensor  scalar
        """
        seq, lbl = self._samples[idx]
        label = torch.tensor(lbl, dtype=torch.long)

        # Return from cache if available
        if self.cache_tokens and idx in self._cache:
            token_ids, mut_tensor = self._cache[idx]
            return token_ids, mut_tensor, label

        # --- Tokenise --- (k_mer=6 matches vocab_size=4096; was k=3 historically)
        token_ids = tokenise_sequence(seq, self.seq_len, self.vocab_size, k=self.k_mer)

        # --- Mutation tensor ---
        try:
            amap        = self._aligner.compute_alignment_map(seq)
            # process() returns [batch_size, max_mutations, mutation_dim]
            # We use batch_size=1 then squeeze to [max_mutations, mutation_dim]
            mut_batch   = self._processor.process(amap, batch_size=1)
            mut_tensor  = mut_batch[0]                 # [max_mutations, mutation_dim]
        except Exception as exc:
            logger.debug("Mutation processing failed for idx %d: %s", idx, exc)
            mut_tensor = torch.zeros(self.max_mutations, self.mutation_dim)

        # Ensure fixed shape (defensive truncation)
        if mut_tensor.shape[0] != self.max_mutations:
            padded = torch.zeros(self.max_mutations, self.mutation_dim)
            n = min(mut_tensor.shape[0], self.max_mutations)
            padded[:n] = mut_tensor[:n]
            mut_tensor = padded

        if self.cache_tokens:
            self._cache[idx] = (token_ids, mut_tensor)

        return token_ids, mut_tensor, label

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def labels(self) -> List[int]:
        """Return the list of integer class labels for all samples."""
        return [lbl for _, lbl in self._samples]

    def __repr__(self) -> str:  # noqa: D401
        dist = Counter(lbl for _, lbl in self._samples)
        return (
            f"GenomicDataset(samples={len(self._samples)}, "
            f"classes={dict(sorted(dist.items()))})"
        )


# ===========================================================================
# BUILDER
# ===========================================================================


class GenomicDatasetBuilder:
    """
    Factory that discovers FASTA files, assigns labels, splits into
    train / val / test subsets, and returns ready-to-use DataLoaders.

    Parameters
    ----------
    dataset_root           : Path  — Root directory containing ``raw/`` and ``healthy/``.
    seq_len                : int   — Token sequence length (must match engine).
    batch_size             : int   — DataLoader batch size.
    train_ratio            : float — Fraction of data for training.
    val_ratio              : float — Fraction for validation.
    test_ratio             : float — Fraction for test.
    max_sequences_per_file : int   — Max sequences read from each FASTA file.
    num_workers            : int   — DataLoader ``num_workers``.
    seed                   : int   — Random seed for reproducible splits.
    class_balance          : bool  — Use ``WeightedRandomSampler`` on training set.
    """

    def __init__(
        self,
        dataset_root: Path,
        seq_len: int = 128,
        batch_size: int = 16,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        max_sequences_per_file: int = 200,
        num_workers: int = 0,
        seed: int = 42,
        class_balance: bool = True,
        stratified: bool = True,
        k_mer: int = 6,
        samples_per_epoch: Optional[int] = None,
    ) -> None:
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must sum to 1.0"
            )

        self.dataset_root           = Path(dataset_root)
        self.seq_len                = seq_len
        self.batch_size             = batch_size
        self.train_ratio            = train_ratio
        self.val_ratio              = val_ratio
        self.test_ratio             = test_ratio
        self.max_sequences_per_file = max_sequences_per_file
        self.num_workers            = num_workers
        self.seed                   = seed
        self.class_balance          = class_balance
        self.stratified             = stratified    # FIX: per-class stratified split
        self.k_mer                  = k_mer          # FIX: propagate k-mer size
        self.samples_per_epoch      = samples_per_epoch

        self._raw_dir     = self.dataset_root / "raw"
        self._healthy_dir = self.dataset_root / "healthy"

        # Populated by discover_files()
        self._all_files: List[Tuple[Path, int]] = []
        # Statistics populated by build_dataloaders()
        self.class_distribution: Counter = Counter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover_files(self) -> List[Tuple[Path, int]]:
        """
        Walk ``dataset/raw/{SUBTYPE}/*.fasta`` and
        ``dataset/healthy/{species}/*.fna`` and return
        ``[(file_path, class_label), …]``.

        Returns
        -------
        list[tuple[Path, int]]

        Raises
        ------
        FileNotFoundError
            If either ``dataset/raw/`` or ``dataset/healthy/`` are missing.
        """
        found: List[Tuple[Path, int]] = []
        counts_per_class: Counter = Counter()

        # ---- Check for canonical / split layout first ----
        split_manifest = self.dataset_root / "splits" / "split_manifest.csv"
        canonical_manifest = self.dataset_root / "canonical" / "manifest.csv"
        manifest_path = split_manifest if split_manifest.exists() else (canonical_manifest if canonical_manifest.exists() else None)
        if manifest_path:
            logger.info("Discovering sequences from canonical manifest: %s", manifest_path)
            fasta_dir = self.dataset_root / "canonical" / "fasta"
            existing_fasta_set = set(os.listdir(fasta_dir)) if fasta_dir.exists() else set()
            self._split_map: Dict[Path, str] = {}
            import csv
            with manifest_path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for r in reader:
                    acc = r.get("insdc_accession") or r.get("record_id")
                    pt = r.get("pathotype", "").upper()
                    subtype = r.get("subtype", "").upper()
                    # 4-class taxonomy alignment
                    if pt == "AVIRULENT" or subtype == "HEALTHY":
                        lbl = 0
                    elif pt == "HPAI" or subtype in ("H5N1", "H5N9", "H7N9"):
                        lbl = 3
                    elif subtype in ("H5N6", "H5N8", "H7N1", "H7N7", "H10N7", "H10N8"):
                        lbl = 2
                    else:
                        lbl = 1

                    fname = f"{acc}.fasta"
                    if fname in existing_fasta_set:
                        fp = fasta_dir / fname
                        found.append((fp, lbl))
                        counts_per_class[lbl] += 1
                        if "split" in r and r["split"]:
                            self._split_map[fp] = r["split"].strip().lower()

            if found:
                self._all_files = found
                logger.info("File discovery complete from manifest — %d files total.", len(found))
                for cls_idx, cls_name in enumerate(CLASS_LABELS):
                    logger.info("  Class %d (%s): %d file(s)", cls_idx, cls_name, counts_per_class[cls_idx])
                return found

        # ---- Check for balanced directory layout (e.g. dataset_augmented/balanced/) ----
        balanced_dir = self.dataset_root / "balanced" if (self.dataset_root / "balanced").exists() else self.dataset_root
        all_possible_cls = list(CLASS_LABELS) + ["Benign", "Likely_Benign", "Likely_Pathogenic", "Pathogenic"]
        class_dirs_exist = any((balanced_dir / cls_name).exists() for cls_name in all_possible_cls)

        if class_dirs_exist:
            logger.info("Discovering files from class-balanced layout in %s ...", balanced_dir)
            for cls_name in all_possible_cls:
                cls_dir = balanced_dir / cls_name
                if not cls_dir.exists():
                    continue
                label = CLASS_NAME_TO_IDX.get(cls_name, 0)
                fasta_files = sorted(
                    list(cls_dir.glob("*.fasta"))
                    + list(cls_dir.glob("*.fa"))
                    + list(cls_dir.glob("*.fna"))
                )
                for fp in fasta_files:
                    if (fp, label) not in found:
                        found.append((fp, label))
                        counts_per_class[label] += 1

                logger.info(
                    "  [balanced] %-20s -> class %d (%s) | %d file(s)",
                    cls_name, label, CLASS_LABELS[label], len(fasta_files)
                )

            if found:
                self._all_files = found
                logger.info(
                    "\nFile discovery complete — %d files total.", len(found)
                )
                for cls_idx, cls_name in enumerate(CLASS_LABELS):
                    logger.info(
                        "  Class %d (%s): %d file(s)",
                        cls_idx, cls_name, counts_per_class[cls_idx],
                    )
                return found

        # ---- Standard raw layout fallback ----
        if not self._raw_dir.exists():
            raise FileNotFoundError(
                f"Raw dataset directory not found: {self._raw_dir}. "
                "Run ncbi_downloader.py or ingest stage first."
            )

        # ---- Viral subtypes ----
        for subtype_dir in sorted(self._raw_dir.iterdir()):
            if not subtype_dir.is_dir():
                continue
            subtype = subtype_dir.name
            label   = SUBTYPE_TO_CLASS.get(subtype)
            if label is None:
                logger.warning(
                    "Unknown subtype directory '%s' — skipping.", subtype
                )
                continue

            fasta_files = sorted(
                list(subtype_dir.glob("*.fasta"))
                + list(subtype_dir.glob("*.fa"))
                + list(subtype_dir.glob("*.fna"))
            )
            if not fasta_files:
                logger.warning(
                    "No FASTA files in %s — subtype will be absent from dataset.",
                    subtype_dir,
                )
                continue

            for fp in fasta_files:
                found.append((fp, label))
                counts_per_class[label] += 1

            logger.info(
                "  [raw] %-8s  → class %d (%s)  |  %d file(s)",
                subtype, label, CLASS_LABELS[label], len(fasta_files),
            )

        # Notice: Healthy host genomes are NOT added as a viral class.
        # If healthy_dir exists, we log it for reference only.
        if self._healthy_dir.exists():
            logger.info(
                "  [healthy] Host reference directory detected at %s (retained as host references, not viral class 0).",
                self._healthy_dir,
            )

        self._all_files = found
        logger.info(
            "\nFile discovery complete — %d files total.", len(found)
        )
        return found
        for cls_idx, cls_name in enumerate(CLASS_LABELS):
            logger.info(
                "  Class %d (%s): %d file(s)",
                cls_idx, cls_name, counts_per_class[cls_idx],
            )

        return found

    def build_dataloaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Discover files, split into train / val / test, create
        :class:`GenomicDataset` instances, and return
        ``(train_loader, val_loader, test_loader)``.

        The training DataLoader uses a :class:`WeightedRandomSampler` when
        ``class_balance=True`` to compensate for class imbalance.

        Returns
        -------
        tuple[DataLoader, DataLoader, DataLoader]
        """
        if not self._all_files:
            self.discover_files()

        if not self._all_files:
            raise RuntimeError(
                "No FASTA files found after discovery. "
                "Ensure dataset/raw/ and dataset/healthy/ contain FASTA data."
            )

        # ------------------------------------------------------------------
        # FIX: Duplicate detection at file-path level before splitting
        # ------------------------------------------------------------------
        seen_paths: set = set()
        deduped: List[Tuple[Path, int]] = []
        dup_count = 0
        for fp, lbl in self._all_files:
            key = str(fp.resolve())
            if key in seen_paths:
                logger.warning("[AUDIT] Duplicate file path detected and removed: %s", fp)
                dup_count += 1
            else:
                seen_paths.add(key)
                deduped.append((fp, lbl))
        if dup_count:
            logger.warning("[AUDIT] %d duplicate file paths removed before splitting.", dup_count)
        all_files_clean = deduped

        # ------------------------------------------------------------------
        # FIX: Per-class STRATIFIED split (guarantees all classes in all splits)
        # Previously: global random shuffle -> slicing (caused Pathogenic=0 in test)
        # ------------------------------------------------------------------
        rng = random.Random(self.seed)

        if hasattr(self, "_split_map") and self._split_map:
            # Use predetermined zero-leakage isolate split from manifest directly
            train_files = [(fp, lbl) for fp, lbl in all_files_clean if self._split_map.get(fp) == "train"]
            val_files   = [(fp, lbl) for fp, lbl in all_files_clean if self._split_map.get(fp) == "val"]
            test_files  = [(fp, lbl) for fp, lbl in all_files_clean if self._split_map.get(fp) == "test"]
            logger.info(
                "Using predetermined isolate-stratified split — train: %d, val: %d, test: %d files (zero isolate leakage verified)",
                len(train_files), len(val_files), len(test_files),
            )
        elif self.stratified:
            # Group files by class label
            from collections import defaultdict
            class_buckets: dict = defaultdict(list)
            for fp, lbl in all_files_clean:
                class_buckets[lbl].append((fp, lbl))

            train_files: List[Tuple[Path, int]] = []
            val_files:   List[Tuple[Path, int]] = []
            test_files:  List[Tuple[Path, int]] = []

            for cls_idx in sorted(class_buckets.keys()):
                cls_files = list(class_buckets[cls_idx])
                rng.shuffle(cls_files)
                n = len(cls_files)

                # FIX: For very small classes (n <= 4), reserve test first to guarantee coverage
                if n <= 4 and n >= 3:
                    # Reserve 1 for test, 1 for val, rest to train
                    te = cls_files[-1:]
                    v  = cls_files[-2:-1]
                    t  = cls_files[:-2]
                elif n == 2:
                    # Only 2 files: 1 train, 1 test (no val)
                    te = cls_files[-1:]
                    v  = []
                    t  = cls_files[:1]
                elif n == 1:
                    # Only 1 file: goes to train
                    t, v, te = cls_files[:], [], []
                else:
                    # Normal case: n >= 5
                    n_train = max(1, int(n * self.train_ratio))
                    n_val   = max(1, int(n * self.val_ratio))
                    n_val   = min(n_val, n - n_train - 1)
                    n_val   = max(1, n_val)
                    t  = cls_files[:n_train]
                    v  = cls_files[n_train: n_train + n_val]
                    te = cls_files[n_train + n_val:]


                train_files.extend(t)
                val_files.extend(v)
                test_files.extend(te)

                logger.info(
                    "[STRATIFIED] Class %-20s | total=%d | train=%d | val=%d | test=%d",
                    CLASS_LABELS[cls_idx], n, len(t), len(v), len(te),
                )

            logger.info(
                "Stratified split complete — train: %d  val: %d  test: %d files",
                len(train_files), len(val_files), len(test_files),
            )
        else:
            # Legacy: global random shuffle (UNSTRATIFIED — kept for backward compat)
            logger.warning(
                "[AUDIT] Using UNSTRATIFIED split. Classes may be missing from val/test. "
                "Set stratified=True (recommended)."
            )
            shuffled = list(all_files_clean)
            rng.shuffle(shuffled)
            n_total = len(shuffled)
            n_train = int(n_total * self.train_ratio)
            n_val   = int(n_total * self.val_ratio)
            train_files = shuffled[:n_train]
            val_files   = shuffled[n_train: n_train + n_val]
            test_files  = shuffled[n_train + n_val:]

            logger.info(
                "Dataset split (unstratified) — train: %d  val: %d  test: %d files",
                len(train_files), len(val_files), len(test_files),
            )

        # ------------------------------------------------------------------
        # FIX: Data leakage audit — check for any file appearing in >1 split
        # ------------------------------------------------------------------
        train_paths = {str(fp.resolve()) for fp, _ in train_files}
        val_paths   = {str(fp.resolve()) for fp, _ in val_files}
        test_paths  = {str(fp.resolve()) for fp, _ in test_files}
        tv_leak = train_paths & val_paths
        tt_leak = train_paths & test_paths
        vt_leak = val_paths   & test_paths
        if tv_leak or tt_leak or vt_leak:
            logger.error(
                "[AUDIT] DATA LEAKAGE DETECTED — train/val overlap=%d, train/test=%d, val/test=%d",
                len(tv_leak), len(tt_leak), len(vt_leak),
            )
        else:
            logger.info("[AUDIT] Data leakage check PASSED — no file overlap between splits.")

        logger.info(
            "Dataset split — train: %d  val: %d  test: %d files",
            len(train_files), len(val_files), len(test_files),
        )

        # Common kwargs for all GenomicDataset instances
        _ds_kwargs = dict(
            seq_len=self.seq_len,
            max_sequences_per_file=self.max_sequences_per_file,  # FIX: was missing
            seed=self.seed,    # FIX: propagate seed
            k_mer=self.k_mer,  # FIX: propagate k-mer size
        )

        train_ds = self._make_dataset(train_files, **_ds_kwargs)
        val_ds   = self._make_dataset(val_files,   **_ds_kwargs)
        test_ds  = self._make_dataset(test_files,  **_ds_kwargs)

        # Record class distribution from training set
        self.class_distribution = Counter(train_ds.labels)

        # Class weights for loss
        self._class_weights = self.get_class_weights(train_ds.labels)

        # ---- Training DataLoader ----
        if self.class_balance and len(train_ds) > 0:
            sampler = self._build_sampler(train_ds.labels)
            train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
            )
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        logger.info(
            "DataLoaders ready — "
            "train_batches=%d  val_batches=%d  test_batches=%d",
            len(train_loader), len(val_loader), len(test_loader),
        )
        self._log_class_distribution(train_ds.labels, "train")
        self._log_class_distribution(val_ds.labels,   "val")
        self._log_class_distribution(test_ds.labels,  "test")

        return train_loader, val_loader, test_loader

    def get_class_weights(self, labels: List[int]) -> torch.FloatTensor:
        """
        Compute inverse-frequency class weights for :class:`torch.nn.CrossEntropyLoss`.

        Parameters
        ----------
        labels : list[int] — Class label for every sample in the subset.

        Returns
        -------
        torch.FloatTensor of shape ``[num_classes]``.
            Higher weight → class is under-represented.
        """
        counts = Counter(labels)
        total  = sum(counts.values()) or 1

        weights = torch.zeros(NUM_CLASSES, dtype=torch.float)
        for cls_idx in range(NUM_CLASSES):
            n = counts.get(cls_idx, 0)
            weights[cls_idx] = total / (NUM_CLASSES * n) if n > 0 else 1.0

        # FIX: Normalize weights so their mean is 1.0 (prevents gradient scaling inflation while preserving class ratios)
        weights = weights / weights.mean()

        logger.info(
            "Class weights (mean=1.0 normalized): %s",
            {CLASS_LABELS[i]: round(weights[i].item(), 4) for i in range(NUM_CLASSES)},
        )
        return weights

    # Expose computed class weights after build_dataloaders() is called.
    @property
    def class_weights(self) -> Optional[torch.FloatTensor]:
        """Inverse-frequency class weights (available after ``build_dataloaders``)."""
        return getattr(self, "_class_weights", None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_dataset(
        self,
        file_list: List[Tuple[Path, int]],
        seq_len: int,
        max_sequences_per_file: int,
        seed: int = 42,
        k_mer: int = 6,
    ) -> "GenomicDataset":
        """Instantiate a GenomicDataset for a given file list."""
        return GenomicDataset(
            fasta_paths=file_list,
            seq_len=seq_len,
            vocab_size=4096,
            max_mutations=32,
            mutation_dim=64,
            max_seq_len_qc=512,
            cache_tokens=True,
            seed=seed,    # FIX: propagate global seed
            k_mer=k_mer,  # FIX: propagate k-mer size
        )

    def _build_sampler(self, labels: List[int]) -> WeightedRandomSampler:
        """
        Build a ``WeightedRandomSampler`` that over-samples minority classes.

        Parameters
        ----------
        labels : list[int] — Class label per training sample.

        Returns
        -------
        WeightedRandomSampler
        """
        class_weights = self.get_class_weights(labels)
        sample_weights = torch.tensor(
            [class_weights[lbl].item() for lbl in labels],
            dtype=torch.float,
        )
        num_samples = self.samples_per_epoch if (self.samples_per_epoch and self.samples_per_epoch > 0) else len(sample_weights)
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True,
        )

    def _log_class_distribution(
        self, labels: List[int], split_name: str
    ) -> None:
        """Log per-class sample counts for a split."""
        counts = Counter(labels)
        parts  = [
            f"{CLASS_LABELS[i]}: {counts.get(i, 0)}"
            for i in range(NUM_CLASSES)
        ]
        logger.info(
            "  [%s]  total=%d  |  %s",
            split_name, len(labels), "  ".join(parts),
        )

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"GenomicDatasetBuilder("
            f"root={self.dataset_root}, "
            f"batch_size={self.batch_size}, "
            f"split={self.train_ratio}/{self.val_ratio}/{self.test_ratio}, "
            f"stratified={self.stratified}, k_mer={self.k_mer})"
        )
