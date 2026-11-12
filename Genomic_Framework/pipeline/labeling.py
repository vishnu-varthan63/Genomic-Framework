"""QC, INSDC dedupe, HA cleavage pathotyping, isolate-linked labels."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Genomic_Framework.pipeline.ingest import MANIFEST_FIELDS
from Genomic_Framework.pipeline.taxonomy import (
    INDETERMINATE_PATHOTYPE,
    classify_ha_pathotype,
    host_ecology,
    looks_contaminated,
    normalize_subtype,
    parse_segment,
)

logger = logging.getLogger(__name__)

CANONICAL_FIELDS = MANIFEST_FIELDS + [
    "pathotype", "label_confidence", "host_ecology", "exclude_reason", "split",
]


class SequenceCurator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        paths = config.get("paths", {})
        self.canonical_dir = Path(paths.get("canonical_dir", "dataset/canonical"))
        self.excluded_dir = Path(paths.get("excluded_dir", "dataset/excluded"))
        qc = config.get("qc", {})
        self.max_n_fraction = float(qc.get("max_n_fraction", 0.05))
        self.min_len = int(config.get("ingest", {}).get("min_len", 700))
        self.max_len = int(config.get("ingest", {}).get("max_len", 2500))

    def curate(self, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        kept: List[Dict[str, Any]] = []
        excluded: List[Dict[str, Any]] = []
        seen_acc: set[str] = set()
        seen_hash: set[str] = set()

        for rec in records:
            reason = self._exclude_reason(rec, seen_acc, seen_hash)
            rec = dict(rec)
            rec["host_ecology"] = host_ecology(rec.get("host", ""))
            rec["subtype"] = rec.get("subtype") or normalize_subtype(rec.get("header", "")) or ""
            rec["segment"] = rec.get("segment") or parse_segment(rec.get("header", "")) or ""
            if reason:
                rec["exclude_reason"] = reason
                rec["pathotype"] = INDETERMINATE_PATHOTYPE
                rec["label_confidence"] = "excluded"
                rec["split"] = ""
                excluded.append(rec)
                continue
            seen_acc.add(rec["insdc_accession"])
            seen_hash.add(rec["sha256"])
            kept.append(rec)

        kept = self._assign_pathotypes(kept)
        self._write_outputs(kept, excluded)
        logger.info("Curation kept %d / excluded %d records.", len(kept), len(excluded))
        return kept, excluded

    def _exclude_reason(self, rec: Dict[str, Any], seen_acc: set, seen_hash: set) -> Optional[str]:
        acc = rec.get("insdc_accession") or rec.get("record_id") or ""
        seq = rec.get("sequence") or ""
        header = rec.get("header") or ""
        if not acc:
            return "missing_accession"
        if acc in seen_acc:
            return "duplicate_accession"
        if rec.get("sha256") in seen_hash:
            return "duplicate_sequence"
        if rec.get("organism_taxid") not in ("11320", 11320, "", None):
            return "wrong_taxid"
        marker = looks_contaminated(header, seq)
        if marker:
            return f"contaminant:{marker}"
        if not seq or not all(ch in "ACGTN" for ch in seq):
            return "invalid_alphabet"
        n_frac = seq.count("N") / max(len(seq), 1)
        if n_frac > self.max_n_fraction:
            return "high_n_fraction"
        if not (self.min_len <= len(seq) <= self.max_len):
            return "length_out_of_range"
        if not rec.get("subtype") and not normalize_subtype(header):
            return "unparseable_subtype"
        if not rec.get("segment") and not parse_segment(header):
            return "unparseable_segment"
        return None

    def _assign_pathotypes(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        isolate_pathotype: Dict[str, str] = {}
        isolate_conf: Dict[str, str] = {}
        for rec in records:
            if rec.get("segment") != "HA":
                continue
            patho, motif = classify_ha_pathotype(rec.get("sequence", ""))
            rec["pathotype"] = patho
            rec["label_confidence"] = "ha_cleavage" if patho != INDETERMINATE_PATHOTYPE else "indeterminate"
            rec["cleavage_motif"] = motif
            iso = rec.get("isolate_id") or rec["insdc_accession"]
            if patho != INDETERMINATE_PATHOTYPE:
                isolate_pathotype[iso] = patho
                isolate_conf[iso] = "isolate_ha_link"

        inherit = bool(self.config.get("labeling", {}).get("inherit_non_ha_from_isolate", True))
        for rec in records:
            if rec.get("segment") == "HA":
                rec.setdefault("pathotype", INDETERMINATE_PATHOTYPE)
                rec.setdefault("label_confidence", "indeterminate")
                continue
            iso = rec.get("isolate_id") or rec["insdc_accession"]
            if inherit and iso in isolate_pathotype:
                rec["pathotype"] = isolate_pathotype[iso]
                rec["label_confidence"] = isolate_conf[iso]
            else:
                rec["pathotype"] = INDETERMINATE_PATHOTYPE
                rec["label_confidence"] = "indeterminate"
        return records

    def _write_outputs(self, kept: List[Dict[str, Any]], excluded: List[Dict[str, Any]]) -> None:
        self.canonical_dir.mkdir(parents=True, exist_ok=True)
        self.excluded_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv(self.canonical_dir / "manifest.csv", kept)
        self._write_csv(self.excluded_dir / "excluded_manifest.csv", excluded)
        report = {
            "n_kept": len(kept),
            "n_excluded": len(excluded),
            "exclude_reasons": _count(excluded, "exclude_reason"),
            "subtypes": _count(kept, "subtype"),
            "segments": _count(kept, "segment"),
            "pathotypes": _count(kept, "pathotype"),
        }
        (self.canonical_dir / "qc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (self.excluded_dir / "exclusion_report.json").write_text(
            json.dumps({"n_excluded": len(excluded), "reasons": report["exclude_reasons"]}, indent=2),
            encoding="utf-8",
        )

    def _write_csv(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CANONICAL_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def attach_sequences(records: List[Dict[str, Any]], fasta_dir: Path) -> List[Dict[str, Any]]:
    out = []
    for rec in records:
        rec = dict(rec)
        if rec.get("sequence"):
            out.append(rec)
            continue
        fasta_path = rec.get("fasta_path") or str(fasta_dir / f"{rec['insdc_accession']}.fasta")
        p = Path(fasta_path)
        if p.exists():
            rec["sequence"] = _read_fasta_seq(p)
        out.append(rec)
    return out


def _read_fasta_seq(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "".join(ln.strip() for ln in lines if ln and not ln.startswith(">")).upper()


def _count(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    c: Dict[str, int] = defaultdict(int)
    for row in rows:
        c[str(row.get(key) or "NA")] += 1
    return dict(c)
