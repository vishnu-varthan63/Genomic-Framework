"""
scripts/ingest_external_sources.py
==================================
Stage 1 (External Sources) — Download avian influenza nucleotide sequences
from Option A (BV-BRC / Bacterial and Viral Bioinformatics Resource Center)
and Option B (Nextstrain Avian Flu curated references).

Sources & Characteristics:
--------------------------
1. BV-BRC (formerly IRD / FluDB):
   - Over 350,000 avian influenza records covering all subtypes (H1-H16, N1-N9).
   - High-performance REST/RQL POST interface:
     - Genome metadata: https://www.bv-brc.org/api/genome/
     - Segment sequence: https://www.bv-brc.org/api/genome_sequence/
   - Rich epidemiological and ecological metadata: host species, geographic
     origin, collection year, and isolate identifiers.

2. Nextstrain Avian Flu:
   - Curated reference genomes across major avian influenza lineages:
     h5n1 (Goose/Guangdong/1/96 clade 2.3.4.4b progenitor), h7n9 (zoonotic),
     h9n2 (poultry endemic), h10nx, h5nx, and h7nx.
   - GenBank flat files for all 8 viral segments (PB2, PB1, PA, HA, NP, NA, M, NS).
   - Pure Python GenBank parser (no Biopython required).

Quality Control & Deduplication:
--------------------------------
- Standard length filtering: 700 to 2,500 nt.
- ACGTN normalization and ambiguous residue threshold (max 5% N).
- Synthetic construct and patent rejection.
- Zero-leakage accession and sequence SHA-256 deduplication against canonical manifest.
- Pure Python and requests (compliant with system Application Control policy).

Usage:
------
  # Download from both BV-BRC and Nextstrain
  python scripts/ingest_external_sources.py --source all --limit-per-subtype 50

  # Download from Nextstrain curated references only
  python scripts/ingest_external_sources.py --source nextstrain

  # Download from BV-BRC only
  python scripts/ingest_external_sources.py --source bvbrc --limit-per-subtype 50 --max-total 500

  # Dry run to inspect external data availability
  python scripts/ingest_external_sources.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest_external")

_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Catalogues & Endpoints
# ---------------------------------------------------------------------------

BVBRC_API_GENOME = "https://www.bv-brc.org/api/genome/"
BVBRC_API_SEQUENCE = "https://www.bv-brc.org/api/genome_sequence/"

NEXTSTRAIN_CONFIG_URL = "https://raw.githubusercontent.com/nextstrain/avian-flu/master/config"
NEXTSTRAIN_LINEAGES = ["h5n1", "h7n9", "h9n2", "h10nx", "h5nx", "h7nx"]
NEXTSTRAIN_SEGMENTS = ["pb2", "pb1", "pa", "ha", "np", "na", "mp", "ns"]

ALL_AVIAN_SUBTYPES: List[str] = [
    # High-consequence / high-pathogenicity H5 clade
    "H5N1", "H5N2", "H5N3", "H5N4", "H5N5", "H5N6", "H5N8", "H5N9",
    # Zoonotic & avian pathotype H7 clade
    "H7N1", "H7N2", "H7N3", "H7N4", "H7N7", "H7N8", "H7N9",
    # Endemic poultry & zoonotic H9 clade
    "H9N2", "H9N1",
    # Emerging poultry & spillover H10 clade
    "H10N3", "H10N7", "H10N8",
    # Wild waterfowl & poultry reservoir clades
    "H1N1", "H2N2", "H3N2", "H3N8", "H4N6", "H6N1", "H6N2",
    "H11N9", "H12N5", "H13N6", "H14N5", "H15N9", "H16N3",
]

SEGMENT_NUM_TO_NAME = {
    "1": "PB2", "2": "PB1", "3": "PA", "4": "HA",
    "5": "NP",  "6": "NA",  "7": "M",  "8": "NS",
    1: "PB2", 2: "PB1", 3: "PA", 4: "HA",
    5: "NP", 6: "NA", 7: "M", 8: "NS",
    "mp": "M", "MP": "M",
}

MANIFEST_FIELDS = [
    "record_id", "insdc_accession", "source", "organism_taxid",
    "subtype", "segment", "host", "country", "collection_date",
    "seq_len", "sha256", "pathotype", "label_confidence", "split",
]

_SYNTHETIC_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bpatent\b", r"\bvector\b", r"\bclone\b", r"\brecombinant\b",
        r"\bsynthetic\b", r"\bartificial\b", r"\bconstruct\b",
        r"\bexpression\s+vector\b", r"\bplasmid\b",
    ]
]


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def _normalise_seq(seq: str) -> str:
    """Normalize raw nucleotide string to uppercase ACGTN."""
    seq = seq.upper().replace("\r", "").replace("\n", "").replace(" ", "").replace("-", "")
    seq = seq.replace("U", "T")
    return re.sub(r"[^ACGTN]", "N", seq)


def _sha256(text: str) -> str:
    """Compute deterministic SHA-256 checksum."""
    return hashlib.sha256(text.encode("ascii", errors="replace")).hexdigest()


def _looks_synthetic(text: str) -> bool:
    """Return True if description contains synthetic or patent indicators."""
    return any(p.search(text) for p in _SYNTHETIC_PATTERNS)


def _parse_segment_from_text(text: str) -> str:
    """Extract segment name (PB2..NS) from text description."""
    seg_m = re.search(r"segment\s+([1-8])", text, re.IGNORECASE)
    if seg_m:
        return SEGMENT_NUM_TO_NAME.get(seg_m.group(1), "")
    named = re.search(
        r"\b(polymerase basic 2|pb2|polymerase basic 1|pb1|polymerase acidic|pa|"
        r"hemagglutinin|ha|nucleoprotein|np|neuraminidase|na|matrix|m1|m2|"
        r"nonstructural|ns1|nep|ns)\b",
        text,
        re.IGNORECASE,
    )
    if named:
        n = named.group(1).lower()
        if "pb2" in n or "polymerase basic 2" in n:
            return "PB2"
        elif "pb1" in n or "polymerase basic 1" in n:
            return "PB1"
        elif "pa" in n or "polymerase acidic" in n:
            return "PA"
        elif "hemagglutinin" in n or n == "ha":
            return "HA"
        elif "nucleoprotein" in n or n == "np":
            return "NP"
        elif "neuraminidase" in n or n == "na":
            return "NA"
        elif "matrix" in n or n in ("m", "m1", "m2"):
            return "M"
        elif "nonstructural" in n or n in ("ns", "ns1", "nep"):
            return "NS"
    return ""


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------

class ExternalIngestCheckpoint:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = {"completed_keys": [], "stats": {}}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Could not read external checkpoint %s: %s", path, e)

    def is_completed(self, key: str) -> bool:
        return key in self.data.get("completed_keys", [])

    def record_completed(self, key: str, count: int) -> None:
        if "completed_keys" not in self.data:
            self.data["completed_keys"] = []
        if key not in self.data["completed_keys"]:
            self.data["completed_keys"].append(key)
        self.data.setdefault("stats", {})[key] = count
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Nextstrain Ingest (Option B)
# ---------------------------------------------------------------------------

def _parse_genbank_record(gb_text: str, default_sub: str = "", default_seg: str = "") -> Optional[Dict[str, str]]:
    """Parse pure-Python GenBank flat file record."""
    acc_m = re.search(r"^ACCESSION\s+([A-Za-z0-9_\.]+)", gb_text, re.MULTILINE)
    acc = acc_m.group(1).strip() if acc_m else None
    if not acc or acc == "UNKNOWN":
        loc_m = re.search(r"^LOCUS\s+([A-Za-z0-9_\.]+)", gb_text, re.MULTILINE)
        acc = loc_m.group(1).strip() if loc_m else None

    if not acc:
        return None

    def_m = re.search(r"^DEFINITION\s+(.+?)(?=\n[A-Z]|\Z)", gb_text, re.MULTILINE | re.DOTALL)
    defn = " ".join(def_m.group(1).split()) if def_m else ""

    # Segment
    seg = default_seg.upper()
    if seg in ("MP", "M1", "M2"):
        seg = "M"
    parsed_seg = _parse_segment_from_text(defn)
    if parsed_seg:
        seg = parsed_seg

    # Subtype
    sub = default_sub.upper()
    sub_m = re.search(r"\(?(H\d{1,2}N\d{1,2})\)?", defn, re.IGNORECASE)
    if sub_m:
        sub = sub_m.group(1).upper()
    elif "H5" in sub:
        sub = "H5N1"  # canonical default for nextstrain h5 lineages
    elif "H7" in sub:
        sub = "H7N9"

    # Host & Country & Date from strain or definition
    host = ""
    country = ""
    col_date = ""
    strain_m = re.search(r"\((A/[^()]+(?:\([^()]+\))?[^()]*)\)", defn) or re.search(r"\b(A/[A-Za-z0-9_\-\.\/]+)", defn)
    if strain_m:
        isolate = strain_m.group(1)
        parts = [p.strip() for p in isolate.split("/")]
        if len(parts) >= 2:
            host = parts[1]
        if len(parts) >= 3:
            country = parts[2]
        ym = re.search(r"(19\d\d|20\d\d)", isolate)
        if ym:
            col_date = ym.group(1)

    # Origin sequence
    origin_m = re.search(r"ORIGIN\s*\n(.*?)(?=\n//|\Z)", gb_text, re.DOTALL)
    if not origin_m:
        return None

    seq_raw = origin_m.group(1)
    seq = re.sub(r"[^a-zA-Z]", "", seq_raw).upper()

    return {
        "accession": acc,
        "definition": defn,
        "segment": seg,
        "subtype": sub,
        "host": host or "Avian",
        "country": country,
        "collection_date": col_date,
        "sequence": seq,
    }


def ingest_nextstrain(
    canonical_fasta_dir: Path,
    raw_staging_dir: Path,
    seen_sha256: Set[str],
    seen_accessions: Set[str],
    min_len: int = 700,
    max_len: int = 2500,
    max_n_fraction: float = 0.05,
    checkpoint: Optional[ExternalIngestCheckpoint] = None,
) -> List[Dict]:
    """Download and process verified reference segment GenBank files from Nextstrain Avian Flu."""
    logger.info("==================================================================")
    logger.info("STARTING NEXTSTRAIN INGESTION (Option B): Curated Reference Lineages")
    logger.info("==================================================================")

    raw_dir = raw_staging_dir / "nextstrain"
    raw_dir.mkdir(parents=True, exist_ok=True)
    canonical_fasta_dir.mkdir(parents=True, exist_ok=True)

    accepted_rows: List[Dict] = []
    session = requests.Session()

    for lineage in NEXTSTRAIN_LINEAGES:
        for seg in NEXTSTRAIN_SEGMENTS:
            key = f"nextstrain_{lineage}_{seg}"
            if checkpoint and checkpoint.is_completed(key):
                logger.info("[Nextstrain] %s already completed in checkpoint; skipping.", key)
                continue

            url = f"{NEXTSTRAIN_CONFIG_URL}/{lineage}/reference_{lineage}_{seg}.gb"
            logger.info("[Nextstrain] Fetching reference %s ...", url)

            try:
                resp = session.get(url, timeout=15)
                if resp.status_code != 200:
                    logger.warning("[Nextstrain] HTTP %d for %s; skipping.", resp.status_code, url)
                    continue

                gb_text = resp.text
                raw_file = raw_dir / f"reference_{lineage}_{seg}.gb"
                raw_file.write_text(gb_text, encoding="utf-8")

                parsed = _parse_genbank_record(gb_text, default_sub=lineage, default_seg=seg)
                if not parsed:
                    logger.warning("[Nextstrain] Failed to parse GenBank record from %s", url)
                    continue

                acc = parsed["accession"]
                raw_seq = parsed["sequence"]

                if _looks_synthetic(parsed["definition"]):
                    logger.info("[Nextstrain] Skipping synthetic record: %s", acc)
                    continue

                norm_seq = _normalise_seq(raw_seq)
                seq_len = len(norm_seq)

                if seq_len < min_len or seq_len > max_len:
                    logger.info("[Nextstrain] Skipping %s: length %d outside [%d, %d]", acc, seq_len, min_len, max_len)
                    continue

                n_count = norm_seq.count("N")
                if n_count / seq_len > max_n_fraction:
                    logger.info("[Nextstrain] Skipping %s: N fraction %.3f exceeds %.3f", acc, n_count / seq_len, max_n_fraction)
                    continue

                sha = _sha256(norm_seq)
                if sha in seen_sha256 or acc in seen_accessions:
                    logger.info("[Nextstrain] Skipping duplicate %s (SHA or accession seen)", acc)
                    continue

                # Write canonical FASTA
                fasta_path = canonical_fasta_dir / f"{acc}.fasta"
                header = f">{acc} Nextstrain|{lineage}|{acc} {parsed['definition']}"
                fasta_path.write_text(f"{header}\n{norm_seq}\n", encoding="utf-8")

                row = {
                    "record_id": f"nextstrain_{acc}",
                    "insdc_accession": acc,
                    "source": "nextstrain",
                    "organism_taxid": "11320",
                    "subtype": parsed["subtype"],
                    "segment": parsed["segment"],
                    "host": parsed["host"],
                    "country": parsed["country"],
                    "collection_date": parsed["collection_date"],
                    "seq_len": seq_len,
                    "sha256": sha,
                    "pathotype": "",
                    "label_confidence": "",
                    "split": "",
                }
                accepted_rows.append(row)
                seen_sha256.add(sha)
                seen_accessions.add(acc)

                if checkpoint:
                    checkpoint.record_completed(key, 1)

            except Exception as e:
                logger.error("[Nextstrain] Error downloading %s: %s", url, e)

    # Check for cattle spillover root genome if present
    extra_url = f"{NEXTSTRAIN_CONFIG_URL}/h5n1-cattle-outbreak/h5_cattle_genome_root.gb"
    try:
        r = session.get(extra_url, timeout=10)
        if r.status_code == 200:
            (raw_dir / "h5_cattle_genome_root.gb").write_text(r.text, encoding="utf-8")
            parsed = _parse_genbank_record(r.text, default_sub="H5N1", default_seg="HA")
            if parsed and parsed["accession"] not in seen_accessions:
                norm_seq = _normalise_seq(parsed["sequence"])
                if min_len <= len(norm_seq) <= max_len:
                    sha = _sha256(norm_seq)
                    if sha not in seen_sha256:
                        acc = parsed["accession"]
                        fasta_path = canonical_fasta_dir / f"{acc}.fasta"
                        fasta_path.write_text(f">{acc} Nextstrain|cattle-root|{acc} {parsed['definition']}\n{norm_seq}\n", encoding="utf-8")
                        accepted_rows.append({
                            "record_id": f"nextstrain_{acc}",
                            "insdc_accession": acc,
                            "source": "nextstrain",
                            "organism_taxid": "11320",
                            "subtype": parsed["subtype"],
                            "segment": parsed["segment"],
                            "host": "Bovine",
                            "country": parsed["country"],
                            "collection_date": parsed["collection_date"],
                            "seq_len": len(norm_seq),
                            "sha256": sha,
                            "pathotype": "",
                            "label_confidence": "",
                            "split": "",
                        })
                        seen_sha256.add(sha)
                        seen_accessions.add(acc)
    except Exception as e:
        logger.debug("[Nextstrain] Cattle root genome fetch optional note: %s", e)

    logger.info("[Nextstrain] Completed. Ingested %d unique canonical reference sequences.", len(accepted_rows))
    return accepted_rows


# ---------------------------------------------------------------------------
# BV-BRC Ingest (Option A)
# ---------------------------------------------------------------------------

def _fetch_bvbrc_genomes_for_subtype(
    session: requests.Session,
    subtype: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Query BV-BRC genome endpoint for avian influenza isolates of a specific subtype."""
    headers = {
        "Content-Type": "application/rqlquery+x-www-form-urlencoded",
        "Accept": "application/json",
    }
    # Query avian host group, taxon 11320, specific subtype
    query = (
        f"and(eq(taxon_id,11320),eq(host_group,Avian),eq(subtype,{subtype}))"
        f"&select(genome_id,genome_name,subtype,host_common_name,host_name,collection_year,country)"
        f"&limit({limit},0)"
    )
    for attempt in range(3):
        try:
            resp = session.post(BVBRC_API_GENOME, data=query, headers=headers, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
            logger.warning("[BV-BRC] Genomes query returned HTTP %d on attempt %d", resp.status_code, attempt + 1)
            time.sleep(1.0 * (attempt + 1))
        except Exception as e:
            logger.warning("[BV-BRC] Genomes query failed on attempt %d: %s", attempt + 1, e)
            time.sleep(1.5 * (attempt + 1))
    return []


def _fetch_bvbrc_sequences_batch(
    session: requests.Session,
    genome_ids: List[str],
    min_len: int = 700,
    max_len: int = 2500,
) -> List[Dict[str, Any]]:
    """Fetch sequences for a batch of BV-BRC genome IDs."""
    if not genome_ids:
        return []
    headers = {
        "Content-Type": "application/rqlquery+x-www-form-urlencoded",
        "Accept": "application/json",
    }
    gids_str = ",".join(genome_ids)
    query = (
        f"in(genome_id,({gids_str}))"
        f"&select(genome_id,accession,segment,description,length,sequence)"
        f"&limit({len(genome_ids) * 10})"
    )
    for attempt in range(3):
        try:
            resp = session.post(BVBRC_API_SEQUENCE, data=query, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
            logger.warning("[BV-BRC] Sequence batch query returned HTTP %d on attempt %d", resp.status_code, attempt + 1)
            time.sleep(1.0 * (attempt + 1))
        except Exception as e:
            logger.warning("[BV-BRC] Sequence batch query failed on attempt %d: %s", attempt + 1, e)
            time.sleep(1.5 * (attempt + 1))
    return []


def ingest_bvbrc(
    canonical_fasta_dir: Path,
    raw_staging_dir: Path,
    seen_sha256: Set[str],
    seen_accessions: Set[str],
    subtypes_list: List[str],
    limit_per_subtype: int = 50,
    max_total: Optional[int] = 500,
    min_len: int = 700,
    max_len: int = 2500,
    max_n_fraction: float = 0.05,
    checkpoint: Optional[ExternalIngestCheckpoint] = None,
) -> List[Dict]:
    """Download avian influenza sequences across catalogued subtypes from BV-BRC (FluDB)."""
    logger.info("==================================================================")
    logger.info("STARTING BV-BRC INGESTION (Option A): %d Subtypes (target limit: %d/subtype)",
                len(subtypes_list), limit_per_subtype)
    logger.info("==================================================================")

    raw_dir = raw_staging_dir / "bvbrc"
    raw_dir.mkdir(parents=True, exist_ok=True)
    canonical_fasta_dir.mkdir(parents=True, exist_ok=True)

    accepted_rows: List[Dict] = []
    session = requests.Session()
    batch_dump_idx = 1

    for s_idx, subtype in enumerate(subtypes_list, 1):
        if max_total and len(accepted_rows) >= max_total:
            logger.info("[BV-BRC] Reached max_total target (%d sequences); concluding ingestion.", max_total)
            break

        cp_key = f"bvbrc_subtype_{subtype}"
        if checkpoint and checkpoint.is_completed(cp_key):
            logger.info("[%d/%d] [BV-BRC] %s already completed in checkpoint; skipping.", s_idx, len(subtypes_list), subtype)
            continue

        logger.info("[%d/%d] [BV-BRC] Querying genomes for subtype %s ...", s_idx, len(subtypes_list), subtype)
        genomes = _fetch_bvbrc_genomes_for_subtype(session, subtype, limit=limit_per_subtype)
        if not genomes:
            logger.info("[BV-BRC] No genomes found for subtype %s.", subtype)
            if checkpoint:
                checkpoint.record_completed(cp_key, 0)
            continue

        logger.info("[BV-BRC] Retrieved %d genomes for subtype %s; fetching sequence segments ...", len(genomes), subtype)
        genome_meta_by_id = {g["genome_id"]: g for g in genomes}

        # Batch fetch sequence records (chunks of 30 genome_ids)
        chunk_size = 30
        gids = list(genome_meta_by_id.keys())
        subtype_accepted = 0

        for i in range(0, len(gids), chunk_size):
            chunk = gids[i : i + chunk_size]
            seq_records = _fetch_bvbrc_sequences_batch(session, chunk, min_len=min_len, max_len=max_len)
            if not seq_records:
                continue

            # Save raw batch dump
            raw_dump_file = raw_dir / f"bvbrc_{subtype}_batch_{batch_dump_idx:04d}.json"
            raw_dump_file.write_text(json.dumps(seq_records, indent=2), encoding="utf-8")
            batch_dump_idx += 1

            for s_rec in seq_records:
                raw_seq = s_rec.get("sequence", "")
                if not raw_seq:
                    continue

                gid = s_rec.get("genome_id")
                g_meta = genome_meta_by_id.get(gid, {})
                desc = s_rec.get("description") or g_meta.get("genome_name", "")

                if _looks_synthetic(desc):
                    continue

                norm_seq = _normalise_seq(raw_seq)
                seq_len = len(norm_seq)
                if seq_len < min_len or seq_len > max_len:
                    continue

                n_count = norm_seq.count("N")
                if n_count / seq_len > max_n_fraction:
                    continue

                sha = _sha256(norm_seq)
                if sha in seen_sha256:
                    continue

                # Accession identification
                acc = (s_rec.get("accession") or "").strip()
                if not acc or acc == "UNKNOWN":
                    # Fallback to unique BV-BRC identifier
                    seg_num = s_rec.get("segment") or "seg"
                    acc = f"BVBRC_{gid}_{seg_num}"

                if acc in seen_accessions:
                    continue

                # Determine segment name
                seg_raw = s_rec.get("segment")
                seg_name = SEGMENT_NUM_TO_NAME.get(str(seg_raw), "")
                if not seg_name:
                    seg_name = _parse_segment_from_text(desc)

                host = g_meta.get("host_common_name") or g_meta.get("host_name") or "Avian"
                country = g_meta.get("country", "")
                col_year = str(g_meta.get("collection_year", ""))

                # Write canonical FASTA
                fasta_path = canonical_fasta_dir / f"{acc}.fasta"
                header = f">{acc} BVBRC|{gid}|{acc} {desc}"
                fasta_path.write_text(f"{header}\n{norm_seq}\n", encoding="utf-8")

                row = {
                    "record_id": f"bvbrc_{acc}",
                    "insdc_accession": acc,
                    "source": "bvbrc",
                    "organism_taxid": "11320",
                    "subtype": g_meta.get("subtype") or subtype,
                    "segment": seg_name,
                    "host": host,
                    "country": country,
                    "collection_date": col_year,
                    "seq_len": seq_len,
                    "sha256": sha,
                    "pathotype": "",
                    "label_confidence": "",
                    "split": "",
                }
                accepted_rows.append(row)
                seen_sha256.add(sha)
                seen_accessions.add(acc)
                subtype_accepted += 1

                if max_total and len(accepted_rows) >= max_total:
                    break

        logger.info("[BV-BRC] Subtype %s: accepted %d new unique sequences.", subtype, subtype_accepted)
        if checkpoint:
            checkpoint.record_completed(cp_key, subtype_accepted)

    logger.info("[BV-BRC] Ingestion completed. Accepted %d new unique sequences.", len(accepted_rows))
    return accepted_rows


# ---------------------------------------------------------------------------
# Manifest Update
# ---------------------------------------------------------------------------

def append_to_manifest(new_rows: List[Dict], manifest_path: Path) -> int:
    """Safely append new rows to canonical manifest.csv."""
    if not new_rows:
        logger.info("No new rows to append to manifest.")
        return 0

    existing_rows: List[Dict] = []
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            existing_rows = list(csv.DictReader(fh))

    # Append new rows
    combined = existing_rows + new_rows

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(combined)

    logger.info("Manifest updated: %s (+%d rows, total %d sequences)", manifest_path, len(new_rows), len(combined))
    return len(combined)


# ---------------------------------------------------------------------------
# CLI & Orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest avian flu data from BV-BRC (Option A) and Nextstrain (Option B).")
    parser.add_argument("--config", default=str(_ROOT / "configs" / "pipeline_config.yaml"), help="Path to pipeline_config.yaml")
    parser.add_argument("--source", choices=["all", "bvbrc", "nextstrain"], default="all", help="Source to ingest")
    parser.add_argument("--limit-per-subtype", type=int, default=50, help="Per-subtype genome limit for BV-BRC")
    parser.add_argument("--max-total-bvbrc", type=int, default=500, help="Max total sequences to ingest from BV-BRC")
    parser.add_argument("--dry-run", action="store_true", help="Inspect availability without saving")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg: Dict = {}
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    paths_cfg = cfg.get("paths", {})
    canonical_dir = _ROOT / paths_cfg.get("canonical_dir", "dataset/canonical")
    raw_staging_dir = _ROOT / paths_cfg.get("staging_dir", "dataset/raw_sources")
    manifest_path = canonical_dir / "manifest.csv"
    canonical_fasta_dir = canonical_dir / "fasta"

    checkpoint_path = raw_staging_dir / "external_sources_checkpoint.json"
    checkpoint = ExternalIngestCheckpoint(checkpoint_path)

    # Load existing accessions and SHAs for zero-leakage deduplication
    seen_accessions: Set[str] = set()
    seen_sha256: Set[str] = set()
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                acc = r.get("insdc_accession", "").strip()
                sha = r.get("sha256", "").strip()
                if acc:
                    seen_accessions.add(acc)
                if sha:
                    seen_sha256.add(sha)
        logger.info("Loaded %d existing accessions and %d sequence SHAs from canonical manifest.",
                    len(seen_accessions), len(seen_sha256))

    if args.dry_run:
        logger.info("[Dry Run] Checking Nextstrain connectivity ...")
        for lineage in ["h5n1", "h7n9", "h9n2"]:
            u = f"{NEXTSTRAIN_CONFIG_URL}/{lineage}/reference_{lineage}_ha.gb"
            r = requests.head(u, timeout=10)
            logger.info("  Nextstrain %s HA: HTTP %d", lineage, r.status_code)

        logger.info("[Dry Run] Checking BV-BRC connectivity ...")
        headers = {"Content-Type": "application/rqlquery+x-www-form-urlencoded", "Accept": "application/json"}
        q = "and(eq(taxon_id,11320),eq(host_group,Avian),eq(subtype,H5N1))&limit(1)"
        r = requests.post(BVBRC_API_GENOME, data=q, headers=headers, timeout=10)
        logger.info("  BV-BRC Avian H5N1 count header: %s", r.headers.get("Content-Range"))
        return 0

    all_new_rows: List[Dict] = []

    # 1. Option B: Nextstrain Reference Ingestion
    if args.source in ("all", "nextstrain"):
        nextstrain_rows = ingest_nextstrain(
            canonical_fasta_dir=canonical_fasta_dir,
            raw_staging_dir=raw_staging_dir,
            seen_sha256=seen_sha256,
            seen_accessions=seen_accessions,
            checkpoint=checkpoint,
        )
        all_new_rows.extend(nextstrain_rows)

    # 2. Option A: BV-BRC Ingestion
    if args.source in ("all", "bvbrc"):
        bvbrc_rows = ingest_bvbrc(
            canonical_fasta_dir=canonical_fasta_dir,
            raw_staging_dir=raw_staging_dir,
            seen_sha256=seen_sha256,
            seen_accessions=seen_accessions,
            subtypes_list=ALL_AVIAN_SUBTYPES,
            limit_per_subtype=args.limit_per_subtype,
            max_total=args.max_total_bvbrc,
            checkpoint=checkpoint,
        )
        all_new_rows.extend(bvbrc_rows)

    # 3. Update manifest
    if all_new_rows:
        append_to_manifest(all_new_rows, manifest_path)
        logger.info("==================================================================")
        logger.info("EXTERNAL INGESTION COMPLETE: %d new real biological sequences added.", len(all_new_rows))
        logger.info("==================================================================")
    else:
        logger.info("Ingestion completed: no new sequences needed (all already present).")

    return 0


if __name__ == "__main__":
    sys.exit(main())

