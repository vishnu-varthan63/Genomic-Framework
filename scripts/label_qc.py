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
# or [RK]-[RK]-[RK]-[RK]↓G  found in H5/H7 HPAI strains.
# We search the translated HA for this pattern in the last 25 aa before Gly
_HPAI_MOTIF = re.compile(r"[RK]{2,}[A-Z]?[RK]G", re.ASCII)
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

    # Scan last 30 aa of HA1 region (approximate: last 30 aa before length/3)
    ha1_end = min(len(prot), len(prot) // 3 + 60)
    scan_region = prot[max(0, ha1_end - 30): ha1_end + 10]

    if _HPAI_MOTIF.search(scan_region):
        if is_h5_or_h7:
            return "HPAI", "high"
        else:
            return "HPAI", "medium"   # unusual but possible

    # Count basic residues in cleavage window
    basic_count = sum(1 for aa in scan_region if aa in _BASIC_AA)
    if is_h5_or_h7:
        if basic_count >= 2:
            return "LPAI", "medium"
        return "LPAI", "high"
    else:
        return "LPAI", "medium"


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

    # Read manifest
    with manifest_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    logger.info("Loaded %d manifest rows", len(rows))

    # Build isolate → HA pathotype map (for inheriting to non-HA segments)
    # Key: (accession prefix without last char) is too crude; use subtype+segment grouping
    # We'll group by: first 10 chars of accession (isolate proxy) + subtype
    ha_pathotype_map: Dict[str, Tuple[str, str]] = {}

    # First pass: label HA segments
    accepted: List[Dict] = []
    excluded: List[Dict] = []

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

        if segment == "HA":
            pathotype, confidence = _classify_ha_pathotype(seq, subtype)
            row["pathotype"] = pathotype
            row["label_confidence"] = confidence
            # Store for inheritance
            isolate_key = acc[:8] + subtype  # crude isolate proxy
            ha_pathotype_map[isolate_key] = (pathotype, confidence)
        else:
            # Non-HA: inherit from isolate-linked HA if possible
            isolate_key = acc[:8] + subtype
            if isolate_key in ha_pathotype_map:
                pt, conf = ha_pathotype_map[isolate_key]
                row["pathotype"] = pt
                row["label_confidence"] = "inherited_" + conf
            else:
                row["pathotype"] = cfg["labeling"].get("indeterminate_pathotype", "INDETERMINATE")
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

