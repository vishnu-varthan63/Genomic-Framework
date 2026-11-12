"""
pipeline/mutation_pipeline.py
==============================
Module 3  — Reference Alignment / BLAST Simulator  (:class:`SequenceAligner`)
Module 6  — Mutation Identification                 (:class:`MutationProcessor`)
Module 7  — Mutation Annotation
Module 8  — Mutation Prioritisation
Module 9  — Mutation Feature Encoding

These four sub-responsibilities are unified inside the single orchestration
class :class:`MutationProcessor` as per the specification, which delegates
alignment work to :class:`SequenceAligner`.

All outputs that feed downstream PyTorch modules are returned as
``torch.Tensor`` objects so the entire pipeline is end-to-end differentiable
at the feature-encoding boundary.

Output Tensor Shape
-------------------
:class:`MutationProcessor` produces a tensor of shape
``[batch_size, max_mutations, mutation_dim]``.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Region / annotation catalogue (simulated)
# ---------------------------------------------------------------------------
_REGION_TYPES: List[str] = [
    "coding",
    "non_coding",
    "promoter",
    "enhancer",
    "structural_loop",
    "splice_site",
    "UTR_5prime",
    "UTR_3prime",
]

# Positional importance weights keyed by region type (higher = more important)
_REGION_WEIGHTS: Dict[str, float] = {
    "coding": 1.00,
    "splice_site": 0.95,
    "promoter": 0.85,
    "enhancer": 0.75,
    "UTR_5prime": 0.65,
    "UTR_3prime": 0.60,
    "structural_loop": 0.50,
    "non_coding": 0.30,
}

# One-hot width for region encoding inside the feature vector
_REGION_ONE_HOT_DIM: int = len(_REGION_TYPES)  # 8

# Mutation type catalogue
_MUTATION_TYPES: List[str] = ["SNP", "insertion", "deletion", "MNP", "none"]
_MUTATION_TYPE_DIM: int = len(_MUTATION_TYPES)  # 5


# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------

@dataclass
class MutationRecord:
    """
    Represents a single annotated, scored mutation.

    Attributes
    ----------
    position : int
        Zero-based index of the mutation in the query sequence.
    query_char : str
        Nucleotide character in the query at *position*.
    ref_char : str
        Nucleotide character in the reference at *position*.
    mutation_type : str
        One of ``_MUTATION_TYPES`` above.
    region : str
        Simulated genomic region where the mutation resides.
    importance_score : float
        Composite importance score (0 – 1) used for prioritisation.
    """

    position: int
    query_char: str
    ref_char: str
    mutation_type: str = "SNP"
    region: str = "non_coding"
    importance_score: float = 0.0


@dataclass
class AlignmentMap:
    """
    Output of :meth:`SequenceAligner.compute_alignment_map`.

    Attributes
    ----------
    query_seq : str
        The query sequence (after QC).
    ref_seq : str
        The reference sequence (possibly truncated / padded to match query).
    mismatch_positions : list[int]
        Zero-based positions where query and reference differ.
    edit_distance : int
        Total character-level edit distance (Levenshtein simulation).
    alignment_string : str
        Compact visual representation ('|' = match, 'X' = mismatch, '-' = gap).
    structural_diffs : list[dict]
        Detailed per-position difference metadata.
    """

    query_seq: str
    ref_seq: str
    mismatch_positions: List[int] = field(default_factory=list)
    edit_distance: int = 0
    alignment_string: str = ""
    structural_diffs: List[Dict] = field(default_factory=list)


# ===========================================================================
# Module 3 — SequenceAligner
# ===========================================================================

class SequenceAligner:
    """
    Reference alignment simulator (BLAST-like pairwise aligner).

    Performs a character-by-character comparison between a processed query
    sequence and an internal wild-type reference, reporting mismatch
    coordinates and an edit-distance approximation.

    Parameters
    ----------
    reference_seq : str | None
        The wild-type reference genome fragment.  If ``None``, a synthetic
        256-character poly-ACGT reference is generated automatically.
    seed : int
        Random seed for reproducible synthetic reference generation.

    Examples
    --------
    >>> aligner = SequenceAligner()
    >>> amap = aligner.compute_alignment_map("ACGTNNACGT")
    >>> print(amap.mismatch_positions)
    """

    # Canonical Avian Influenza Hemagglutinin (HA) wild-type reference sequence (A/Goose/Guangdong/1/1996)
    _CANONICAL_HA_REF: str = (
        "ATGGAGAAAATAGTGCTTCTTCTTGCAATAGTCAGTCTTGTTAAAAGTGATCAGATTTGCATTGGTTACCATGCAAACAACTCGACA"
        "GAGCAGGTTGACACAATAATGGAAAAGAACGTTACTGTTACACATGCCCAAGACATACTGGAAAAGACACACAACGGGAAGCTCTGC"
        "GATCTAGATGGAGTGAAGCCTCTAATTTTGAGAGATTGTAGTGTAGCTGGATGGCTCCTCGGAAACCCAATGTGTGACGAATTCATC"
        "AATGTGCCGGAATGGTCTTACATAGTGGAGAAGGCCAATCCAGTCAATGACCTCTGTTACCCAGGGGATTTCAATGACTATGAAGAA"
        "TTGAAACACCTATTGAGCAGAATAAACCATTTTGAGAAAATTCAGATCATCCCCAAAAGTTCTTGGTCCAGTCATGAAGCCTCATTA"
        "GGGGTGAGCTCAGCATGTCCATACCAGGGAAAGTCCTCCTTTTTCAGAAATGTGGTATGGCTTATCAAAAAGAACAGTACATACCCA"
    )
    _DEFAULT_REF_LEN: int = 512

    def __init__(
        self,
        reference_seq: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        if reference_seq is None:
            # Use canonical A/Goose/Guangdong/1/1996 wild-type HA reference
            self.reference_seq: str = (self._CANONICAL_HA_REF * 2)[: self._DEFAULT_REF_LEN]
            logger.info(
                "SequenceAligner — initialized canonical AIV HA reference sequence of length %d.",
                len(self.reference_seq),
            )
        else:
            self.reference_seq = reference_seq.upper()
            logger.info(
                "SequenceAligner — loaded user reference of length %d.",
                len(self.reference_seq),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_alignment_map(self, query_seq: str) -> AlignmentMap:
        """
        Align *query_seq* against the stored reference and return an
        :class:`AlignmentMap`.

        Strategy
        --------
        The alignment is performed character-by-character (global, no-gap
        alignment).  When the sequences differ in length the shorter one is
        right-padded with ``'N'`` to ensure aligned comparison.

        Parameters
        ----------
        query_seq : str
            The cleaned, QC-passed query sequence.

        Returns
        -------
        AlignmentMap
            Full alignment result including mismatch positions and edit
            distance.
        """
        query = query_seq.upper()
        ref = self.reference_seq

        # Normalise lengths to the shorter of the two for valid comparison
        cmp_len = min(len(query), len(ref))
        q_cmp = query[:cmp_len]
        r_cmp = ref[:cmp_len]

        mismatch_positions: List[int] = []
        alignment_chars: List[str] = []
        structural_diffs: List[Dict] = []

        for i, (qc, rc) in enumerate(zip(q_cmp, r_cmp)):
            if qc == rc or qc == "N" or rc == "N":
                alignment_chars.append("|")
            else:
                alignment_chars.append("X")
                mismatch_positions.append(i)
                structural_diffs.append({
                    "position": i,
                    "query_char": qc,
                    "ref_char": rc,
                    "offset_from_start": i,
                    "offset_from_end": cmp_len - 1 - i,
                })

        alignment_string = "".join(alignment_chars)
        edit_distance = self._levenshtein_distance_approx(q_cmp, r_cmp, mismatch_positions)

        amap = AlignmentMap(
            query_seq=query,
            ref_seq=ref,
            mismatch_positions=mismatch_positions,
            edit_distance=edit_distance,
            alignment_string=alignment_string,
            structural_diffs=structural_diffs,
        )

        logger.info(
            "Alignment complete — cmp_len=%d, mismatches=%d, edit_dist=%d",
            cmp_len,
            len(mismatch_positions),
            edit_distance,
        )
        return amap

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _levenshtein_distance_approx(
        s1: str,
        s2: str,
        precomputed_mismatches: Optional[List[int]] = None,
    ) -> int:
        """
        Fast Levenshtein-distance approximation for equal-length strings.

        For sequences of the same length (as produced by the global alignment
        above) the edit distance reduces to the Hamming distance — the number
        of character-level mismatches.  We use the precomputed mismatch list
        when available to avoid re-scanning.
        """
        if precomputed_mismatches is not None:
            return len(precomputed_mismatches)
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"SequenceAligner(ref_len={len(self.reference_seq)})"
        )


# ===========================================================================
# Modules 6-9 — MutationProcessor
# ===========================================================================

class MutationProcessor:
    """
    Unified mutation-identification, annotation, prioritisation, and feature-
    encoding orchestrator.

    Sub-responsibilities
    --------------------
    * **Module 6 — Identification**: extract raw mismatch coordinates from
      an :class:`AlignmentMap`.
    * **Module 7 — Annotation**: assign genomic-region labels and mutation
      types to each raw coordinate.
    * **Module 8 — Prioritisation**: score and rank mutations; retain only the
      top ``max_mutations`` by composite importance.
    * **Module 9 — Encoding**: convert prioritised :class:`MutationRecord`
      objects into a fixed-size numerical feature tensor.

    Parameters
    ----------
    max_mutations : int
        Maximum number of mutations to retain after prioritisation.  The
        output tensor second dimension equals this value.
    mutation_dim : int
        Size of the feature vector encoding each mutation.  Output tensor
        last dimension equals this value.
    seed : int
        Random seed for reproducible stochastic annotation (region assignment
        uses a seeded RNG to simulate database lookup).

    Output Tensor Shape
    -------------------
    ``[batch_size, max_mutations, mutation_dim]``
    """

    def __init__(
        self,
        max_mutations: int = 32,
        mutation_dim: int = 64,
        seed: int = 0,
    ) -> None:
        self.max_mutations: int = max_mutations
        self.mutation_dim: int = mutation_dim
        self._rng: random.Random = random.Random(seed)

        # Sanity check: feature vector must be large enough to hold all fields
        # Position (1) + ref one-hot (4) + query one-hot (4) + type one-hot (5)
        # + region one-hot (8) + score (1) + rank normalised (1) = 24 minimum
        _min_dim = 1 + 4 + 4 + _MUTATION_TYPE_DIM + _REGION_ONE_HOT_DIM + 2
        if mutation_dim < _min_dim:
            raise ValueError(
                f"mutation_dim ({mutation_dim}) must be ≥ {_min_dim} to encode "
                "all mutation metadata fields."
            )

        logger.info(
            "MutationProcessor initialised — max_mutations=%d, mutation_dim=%d",
            max_mutations,
            mutation_dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        alignment_map: AlignmentMap,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Full Mod 6-9 pipeline: identify → annotate → prioritise → encode.

        Parameters
        ----------
        alignment_map : AlignmentMap
            Output of :meth:`SequenceAligner.compute_alignment_map`.
        batch_size : int
            Number of identical copies to stack along dim-0 (simulates a
            multi-sample batch when a single alignment map is available).

        Returns
        -------
        torch.Tensor
            Shape ``[batch_size, max_mutations, mutation_dim]``.
        """
        # --- Mod 6: Identification ---
        raw_mutations = self._identify_mutations(alignment_map)

        # --- Mod 7: Annotation ---
        annotated = self._annotate_mutations(raw_mutations)

        # --- Mod 8: Prioritisation ---
        prioritised = self._prioritise_mutations(annotated)

        # --- Mod 9: Encoding ---
        single_sample_tensor = self._encode_mutations(
            prioritised, alignment_map
        )  # [max_mutations, mutation_dim]

        # Expand to batch dimension
        batch_tensor = single_sample_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
        # Shape: [batch_size, max_mutations, mutation_dim]

        logger.info(
            "MutationProcessor output tensor shape: %s",
            list(batch_tensor.shape),
        )
        return batch_tensor

    # ------------------------------------------------------------------
    # Module 6 — Mutation Identification
    # ------------------------------------------------------------------

    def _identify_mutations(self, amap: AlignmentMap) -> List[MutationRecord]:
        """
        Extract a raw list of :class:`MutationRecord` objects from the
        alignment map.  Each mismatch position produces one record.
        """
        records: List[MutationRecord] = []
        for diff in amap.structural_diffs:
            pos: int = diff["position"]
            qc: str = diff["query_char"]
            rc: str = diff["ref_char"]
            records.append(
                MutationRecord(
                    position=pos,
                    query_char=qc,
                    ref_char=rc,
                )
            )
        logger.debug("Module 6 — identified %d raw mutations.", len(records))
        return records

    # ------------------------------------------------------------------
    # Module 7 — Mutation Annotation
    # ------------------------------------------------------------------

    def _annotate_mutations(
        self, records: List[MutationRecord]
    ) -> List[MutationRecord]:
        """
        Assign genomic-region labels and mutation-type classifications to each
        raw :class:`MutationRecord`.

        Region assignment is simulated by hashing the position modulo the
        region catalogue length — in production this would query a GTF/GFF
        annotation database.
        """
        for rec in records:
            # Simulated region lookup (deterministic by position)
            region_idx = rec.position % len(_REGION_TYPES)
            rec.region = _REGION_TYPES[region_idx]

            # Mutation-type classification (SNP is the default for single
            # character mismatches; insertion/deletion would require gap tokens)
            rec.mutation_type = "SNP"

        logger.debug("Module 7 — annotated %d mutations.", len(records))
        return records

    # ------------------------------------------------------------------
    # Module 8 — Mutation Prioritisation
    # ------------------------------------------------------------------

    def _prioritise_mutations(
        self, records: List[MutationRecord]
    ) -> List[MutationRecord]:
        """
        Score each mutation and retain the top ``max_mutations`` by composite
        importance.

        Scoring formula
        ---------------
        ``score = region_weight * positional_decay``

        where ``positional_decay`` is a Gaussian decay penalising mutations
        very close to the sequence start (untranslated) or far from coding
        regions.
        """
        if not records:
            return records

        # Compute importance scores
        max_pos = max(r.position for r in records) or 1
        for rec in records:
            region_weight = _REGION_WEIGHTS.get(rec.region, 0.30)
            # Gaussian decay centred at 30 % into the sequence
            centre = 0.3 * max_pos
            sigma = 0.4 * max_pos
            positional_decay = math.exp(
                -0.5 * ((rec.position - centre) / sigma) ** 2
            )
            rec.importance_score = round(region_weight * positional_decay, 6)

        # Sort descending by importance
        records.sort(key=lambda r: r.importance_score, reverse=True)

        # Retain top-k
        prioritised = records[: self.max_mutations]
        logger.debug(
            "Module 8 — prioritised %d / %d mutations.",
            len(prioritised),
            len(records),
        )
        return prioritised

    # ------------------------------------------------------------------
    # Module 9 — Mutation Feature Encoding
    # ------------------------------------------------------------------

    def _encode_mutations(
        self,
        records: List[MutationRecord],
        amap: AlignmentMap,
    ) -> torch.Tensor:
        """
        Convert a list of annotated + prioritised :class:`MutationRecord`
        objects into a dense numerical tensor.

        Feature vector layout (per mutation)
        -------------------------------------
        [0]     Normalised position  (1 dim)
        [1:5]   One-hot encoded reference nucleotide  (4 dims: A C G T)
        [5:9]   One-hot encoded query nucleotide  (4 dims: A C G T)
        [9:14]  One-hot encoded mutation type  (5 dims)
        [14:22] One-hot encoded region  (8 dims)
        [22]    Importance score  (1 dim)
        [23]    Rank normalised (0 = top rank → 1 = bottom)  (1 dim)
        [24:]   Zero-padded to reach ``mutation_dim``

        Total deterministic dims = 24; rest padded to ``mutation_dim``.
        """
        seq_len = len(amap.query_seq) or 1
        base_order = ["A", "C", "G", "T"]
        n_records = len(records)

        # Initialise tensor with zeros — handles padding implicitly
        tensor = torch.zeros(self.max_mutations, self.mutation_dim)

        for rank, rec in enumerate(records):
            vec = torch.zeros(self.mutation_dim)
            offset = 0

            # -- Normalised position --
            vec[offset] = rec.position / seq_len
            offset += 1

            # -- Reference nucleotide one-hot --
            if rec.ref_char in base_order:
                vec[offset + base_order.index(rec.ref_char)] = 1.0
            offset += 4

            # -- Query nucleotide one-hot --
            if rec.query_char in base_order:
                vec[offset + base_order.index(rec.query_char)] = 1.0
            offset += 4

            # -- Mutation type one-hot --
            if rec.mutation_type in _MUTATION_TYPES:
                vec[offset + _MUTATION_TYPES.index(rec.mutation_type)] = 1.0
            offset += _MUTATION_TYPE_DIM

            # -- Region one-hot --
            if rec.region in _REGION_TYPES:
                vec[offset + _REGION_TYPES.index(rec.region)] = 1.0
            offset += _REGION_ONE_HOT_DIM

            # -- Importance score --
            vec[offset] = rec.importance_score
            offset += 1

            # -- Rank normalised --
            vec[offset] = rank / max(n_records - 1, 1)
            offset += 1

            # Remaining dims are left as 0.0 (padding)
            tensor[rank] = vec

        logger.debug(
            "Module 9 — encoded tensor shape: %s", list(tensor.shape)
        )
        return tensor  # [max_mutations, mutation_dim]

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"MutationProcessor("
            f"max_mutations={self.max_mutations}, "
            f"mutation_dim={self.mutation_dim})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test (module-level)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Build a synthetic query sequence
    import random as _rnd
    _rnd.seed(7)
    _bases = ["A", "C", "G", "T"]
    query = "".join(_rnd.choice(_bases) for _ in range(64))

    aligner = SequenceAligner(seed=42)
    amap = aligner.compute_alignment_map(query)

    print(f"\n=== SequenceAligner Smoke Test ===")
    print(f"  Query  : {amap.query_seq[:40]}…")
    print(f"  Ref    : {amap.ref_seq[:40]}…")
    print(f"  Align  : {amap.alignment_string[:40]}…")
    print(f"  Mismatches: {len(amap.mismatch_positions)} positions")
    print(f"  Edit distance: {amap.edit_distance}")

    processor = MutationProcessor(max_mutations=8, mutation_dim=64)
    mut_tensor = processor.process(amap, batch_size=2)

    print(f"\n=== MutationProcessor Smoke Test ===")
    print(f"  Output tensor shape: {list(mut_tensor.shape)}")
    print(f"  First mutation feature (first 10 dims): {mut_tensor[0, 0, :10]}")
