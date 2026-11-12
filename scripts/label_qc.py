"""
scripts/label_qc.py
===================
Stage 2 — QC, contamination filtering, and HA-cleavage pathotype labeling.

Reads  : dataset/canonical/manifest.csv  +  dataset/canonical/fasta/*.fasta
Writes : dataset/canonical/manifest.csv  (pathotype + label_confidence filled in)
         dataset/excluded/excluded_report.csv

Pathotype Rules (WOAH definition)
----------------------------------
  HPAI  : HA segment with polybasic cleavage site motif (≥2 basic residues at
           the HA1/HA2 junction: R/K-R/K-X-R/K or similar) in H5 or H7 subtypes.
  LPAI  : HA segment, monobasic cleavage site, OR H9/H10/other.
  INDETERMINATE : non-HA segment; pathotype inherited from an isolate-linked HA
           when available, otherwise labelled INDETERMINATE.

This module translates nucleotide sequence → amino acids using the standard
genetic code and scans the HA1/HA2 cleavage region (last ~10 aa of HA1) for
the polybasic motif.  No Biopython required.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("label_qc")

# ---------------------------------------------------------------------------
# Standard genetic code (codon → aa, single-letter)
# ---------------------------------------------------------------------------
_CODON_TABLE: Dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# HPAI cleavage site motif: polybasic insert of form [RK]-[RK]-X-[RK]↓G
# or [RK]-[RK]-[RK]-[RK]↓G found in H5/H7 HPAI strains preceding the HA2 fusion peptide (GLF / GIF).
_HPAI_MOTIF = re.compile(r"([RK][KR][RKTR][RK]|RERRRKKR|REKRRKR|RG[KR]{3,}|[RK]{2,}[A-Z]?[RK])(?=G[ILF])", re.ASCII)
_HPAI_FALLBACK = re.compile(r"[RK]{3,}[A-Z]?[RK]", re.ASCII)
_BASIC_AA = frozenset("RK")


def _translate(nt_seq: str) -> str:
    """Translate nucleotide sequence (all 3 frames, return longest ORF)."""
    nt_seq = nt_seq.upper().replace("U", "T")
    best_prot = ""
    for frame in range(3):
        prot_parts: List[str] = []
        for i in range(frame, len(nt_seq) - 2, 3):
            codon = nt_seq[i:i+3]
            aa = _CODON_TABLE.get(codon, "X")
            if aa == "*":
                candidate = "".join(prot_parts)
                if len(candidate) > len(best_prot):
                    best_prot = candidate
                prot_parts = []
            else:
                prot_parts.append(aa)
        candidate = "".join(prot_parts)
        if len(candidate) > len(best_prot):
            best_prot = candidate
    return best_prot


def _classify_ha_pathotype(seq: str, subtype: str) -> Tuple[str, str]:
    """
    Translate HA nucleotide sequence and classify pathotype.

    Returns (pathotype, confidence).
      pathotype   : 'HPAI' | 'LPAI' | 'INDETERMINATE'
      confidence  : 'high' | 'medium' | 'low'
    """
    # Only H5 and H7 can be HPAI by WOAH definition based on cleavage site alone
    is_h5_or_h7 = subtype.upper().startswith(("H5", "H7"))

    prot = _translate(seq)
    if len(prot) < 30:
        return "INDETERMINATE", "low"   # too short to assess

    # Scan for polybasic motif immediately preceding the HA2 fusion peptide (G[ILF])
    if _HPAI_MOTIF.search(prot):
        if is_h5_or_h7:
            return "HPAI", "high"
        return "HPAI", "medium"

    # Fallback: scan cleavage loop region (residues 290..380 in standard HA numbering)
    if len(prot) >= 300:
        window = prot[290: min(len(prot), 380)]
        if _HPAI_FALLBACK.search(window):
            if is_h5_or_h7:
                return "HPAI", "high"
            return "HPAI", "medium"

    if is_h5_or_h7:
        return "LPAI", "high"
    return "LPAI", "high"


def _load_fasta(fasta_path: Path) -> Optional[str]:
    """Read a single-record FASTA and return the sequence."""
    try:
        lines = fasta_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "".join(ln.strip() for ln in lines if not ln.startswith(">"))
    except OSError:
        return None


def _is_contaminant(seq: str, min_viral_len: int = 500) -> bool:
    """Basic heuristic: too short → likely fragment/contaminant."""
    return len(seq) < min_viral_len


def _get_isolate_key(acc: str, subtype: str) -> str:
    """Derive isolate key for segment linkage and pathotype inheritance."""
    if acc.startswith("BVBRC_"):
        return "_".join(acc.split("_")[:2])
    # Nextstrain references
    if acc.startswith("AF1443"):
        return "nextstrain_gsgd96_h5n1"
    if acc.startswith("NC_0264") or acc == "KJ411975":
        return "nextstrain_shanghai13_h7n9"
    if acc.startswith("CY0878"):
        return "nextstrain_h9n2"
    # General INSDC accession: strip version and take 7-char prefix + subtype
    acc_clean = acc.split(".")[0]
    return f"{acc_clean[:7]}_{subtype.upper()}"


def run_label_qc(cfg: Dict) -> int:
    canonical_dir = Path(cfg["paths"]["canonical_dir"])
    excluded_dir  = Path(cfg["paths"].get("excluded_dir", "dataset/excluded"))
    manifest_path = canonical_dir / "manifest.csv"
    fasta_dir     = canonical_dir / "fasta"

    if not manifest_path.exists():
        logger.error("Manifest not found at %s. Run ingest stage first.", manifest_path)
        return 1

    excluded_dir.mkdir(parents=True, exist_ok=True)
    excluded_report = excluded_dir / "excluded_report.csv"

    # Read manifest (filter for real biological isolates)
    with manifest_path.open("r", encoding="utf-8") as fh:
        raw_rows = list(csv.DictReader(fh))

    rows = [
        r for r in raw_rows
        if not r.get("source", "").startswith("augmented_") and "_aug" not in r.get("insdc_accession", "")
    ]
    logger.info("Loaded %d manifest rows (%d real biological isolates for QC)", len(raw_rows), len(rows))

    ha_pathotype_map: Dict[str, Tuple[str, str]] = {}
    valid_records: List[Tuple[Dict, str]] = []  # (row, seq)
    excluded: List[Dict] = []

    # Pass 1: Validate sequences, filter contaminants, and classify all HA segments
    for row in rows:
        acc = row["insdc_accession"]
        segment = row.get("segment", "")
        subtype = row.get("subtype", "")
        fasta_path = fasta_dir / f"{acc}.fasta"

        if not fasta_path.exists():
            row["_exclude_reason"] = "fasta_missing"
            excluded.append(row)
            continue

        seq = _load_fasta(fasta_path)
        if not seq:
            row["_exclude_reason"] = "empty_fasta"
            excluded.append(row)
            continue

        if _is_contaminant(seq):
            row["_exclude_reason"] = "too_short_contaminant"
            excluded.append(row)
            continue

        valid_records.append((row, seq))

        if segment == "HA":
            pathotype, confidence = _classify_ha_pathotype(seq, subtype)
            row["pathotype"] = pathotype
            row["label_confidence"] = confidence
            isolate_key = _get_isolate_key(acc, subtype)
            ha_pathotype_map[isolate_key] = (pathotype, confidence)

    # Pass 2: Inherit pathotype to non-HA segments from isolate-linked HA
    inherit_enabled = cfg.get("labeling", {}).get("inherit_non_ha_from_isolate", True)
    indeterminate_pt = cfg.get("labeling", {}).get("indeterminate_pathotype", "INDETERMINATE")

    accepted: List[Dict] = []
    for row, seq in valid_records:
        segment = row.get("segment", "")
        subtype = row.get("subtype", "")
        acc = row["insdc_accession"]

        if segment != "HA":
            isolate_key = _get_isolate_key(acc, subtype)
            if inherit_enabled and isolate_key in ha_pathotype_map:
                pt, conf = ha_pathotype_map[isolate_key]
                row["pathotype"] = pt
                row["label_confidence"] = f"inherited_{conf}"
            else:
                row["pathotype"] = indeterminate_pt
                row["label_confidence"] = "low"

        accepted.append(row)

    logger.info(
        "Label QC complete: %d accepted, %d excluded",
        len(accepted), len(excluded),
    )

    # Pathotype distribution
    pt_counts: Dict[str, int] = defaultdict(int)
    for row in accepted:
        pt_counts[row.get("pathotype", "UNKNOWN")] += 1
    logger.info("Pathotype distribution: %s", dict(pt_counts))

    # Write updated manifest (accepted only)
    fieldnames = list(rows[0].keys()) if rows else []
    # Remove internal fields
    fieldnames = [f for f in fieldnames if not f.startswith("_")]

    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(accepted)
    logger.info("Updated manifest: %s", manifest_path)

    # Write exclusion report
    with excluded_report.open("w", newline="", encoding="utf-8") as fh:
        exclude_fields = fieldnames + ["_exclude_reason"]
        writer = csv.DictWriter(fh, fieldnames=exclude_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(excluded)
    logger.info("Exclusion report: %s  (%d rows)", excluded_report, len(excluded))

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Label QC for avian influenza sequences")
    p.add_argument("--config", type=Path, default=Path("configs/pipeline_config.yaml"))
    args = p.parse_args(argv)

    with args.config.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    return run_label_qc(cfg)


if __name__ == "__main__":
    sys.exit(main())

