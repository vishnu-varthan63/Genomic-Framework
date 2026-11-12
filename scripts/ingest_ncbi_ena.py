"""
scripts/ingest_ncbi_ena.py
==========================
Stage 1 — Collect free public avian influenza nucleotide sequences across all
subtypes and substrands (genomic segments) from NCBI Entrez.

Sources & Features
------------------
- NCBI Entrez REST (eutils): TaxID 11320 (Influenza A virus).
- History server support (`usehistory=y`, `WebEnv`, `QueryKey`) for resilient,
  high-throughput streaming of hundreds of thousands of sequences.
- Comprehensive Avian Subtypes catalogue: covers all H1–H16 and N1–N9 combinations,
  particularly high-pathogenicity H5 (H5N1–H5N9), H7 (H7N1–H7N9), H9 (H9N2),
  H10 (H10N3–H10N8), and wild-bird reservoirs (H3N8, H4N6, H6N2, H11N9, etc.).
- Comprehensive Substrands (Segments): full representation across all 8 viral
  RNA segments: PB2, PB1, PA, HA, NP, NA, M, and NS.
- Automatic header metadata extraction: parses subtype, segment, host species,
  geographic origin, collection year, and isolate identifier directly.
- Checkpoint persistence and resume: stores progress so downloads resume
  seamlessly without duplicate fetches.
- Pure Python and `requests` — complies with system Application Control policy
  (no Biopython C-extensions required).

Output
------
  dataset/raw_sources/ncbi/         — raw batch FASTA dumps
  dataset/canonical/fasta/          — one FASTA per accession (normalized ACGTN)
  dataset/canonical/manifest.csv    — unified metadata for all accepted records
  dataset/raw_sources/ncbi/ingest_checkpoint.json — download state

Usage
-----
  # Download across all avian influenza subtypes
  python scripts/ingest_ncbi_ena.py --live --all-subtypes --per-subtype-limit 500

  # Download across all 8 viral substrands / segments
  python scripts/ingest_ncbi_ena.py --live --all-substrands --per-substrand-limit 500

  # Connect and download all matching avian sequences
  python scripts/ingest_ncbi_ena.py --live --download-all --limit 5000

  # Dry run to inspect NCBI hit counts across all subtypes and substrands
  python scripts/ingest_ncbi_ena.py --dry-run
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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple

import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest")

_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Comprehensive Avian Influenza Subtype & Substrand Catalogues
# ---------------------------------------------------------------------------

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

VIRAL_SUBSTRANDS: List[Tuple[str, str]] = [
    ("PB2", "segment 1 OR PB2 OR \"polymerase basic 2\""),
    ("PB1", "segment 2 OR PB1 OR \"polymerase basic 1\""),
    ("PA",  "segment 3 OR PA OR \"polymerase acidic\""),
    ("HA",  "segment 4 OR HA OR hemagglutinin"),
    ("NP",  "segment 5 OR NP OR nucleoprotein"),
    ("NA",  "segment 6 OR NA OR neuraminidase"),
    ("M",   "segment 7 OR \"matrix protein\" OR MP"),
    ("NS",  "segment 8 OR nonstructural OR NS1 OR NEP"),
]

AVIAN_HOST_KEYWORDS = [
    "avian", "bird", "poultry", "chicken", "duck", "goose",
    "gull", "quail", "turkey", "swan", "teal", "mallard",
    "wild bird", "pigeon", "dove", "pheasant", "egret", "heron",
]

# ---------------------------------------------------------------------------
# NCBI Entrez REST (eutils) — history & streaming support
# ---------------------------------------------------------------------------

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_RETMAX = 200          # records per NCBI efetch call
_MAX_RETRIES = 4


def _ncbi_esearch_history(
    query: str,
    email: str,
    api_key: Optional[str] = None,
    delay: float = 0.34,
) -> Tuple[int, str, str]:
    """
    Execute an NCBI Nucleotide esearch with usehistory=y.
    Returns (hit_count, WebEnv, QueryKey).
    """
    params = {
        "db": "nucleotide",
        "term": query,
        "usehistory": "y",
        "retmax": 0,
        "tool": "avian_genomic_intel",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{_EUTILS_BASE}/esearch.fcgi"
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=45)
            if resp.status_code == 429:
                wait_sec = delay * (2 ** attempt) + 1.0
                logger.warning("[NCBI esearch] 429 Rate limit hit, backing off %.1fs", wait_sec)
                time.sleep(wait_sec)
                continue
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            count = int(root.findtext("Count", "0"))
            webenv = root.findtext("WebEnv", "")
            query_key = root.findtext("QueryKey", "")
            time.sleep(delay)
            return count, webenv, query_key
        except (requests.RequestException, ET.ParseError) as exc:
            if attempt == _MAX_RETRIES:
                logger.error("[NCBI esearch] Failed after %d attempts: %s", attempt, exc)
                raise
            time.sleep(delay * attempt)
    return 0, "", ""


def _ncbi_efetch_fasta_history(
    webenv: str,
    query_key: str,
    retstart: int,
    retmax: int,
    email: str,
    api_key: Optional[str] = None,
    delay: float = 0.34,
) -> str:
    """Fetch a FASTA batch using NCBI history server."""
    params = {
        "db": "nucleotide",
        "query_key": query_key,
        "WebEnv": webenv,
        "rettype": "fasta",
        "retmode": "text",
        "retstart": retstart,
        "retmax": retmax,
        "tool": "avian_genomic_intel",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{_EUTILS_BASE}/efetch.fcgi"
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(url, data=params, timeout=120)
            if resp.status_code == 429:
                wait_sec = delay * (2 ** attempt) + 1.5
                logger.warning("[NCBI efetch] 429 Rate limit hit, backing off %.1fs", wait_sec)
                time.sleep(wait_sec)
                continue
            resp.raise_for_status()
            time.sleep(delay)
            return resp.text
        except requests.RequestException as exc:
            if attempt == _MAX_RETRIES:
                logger.error("[NCBI efetch] Batch starting at %d failed: %s", retstart, exc)
                return ""
            time.sleep(delay * attempt + 1.0)
    return ""


# ---------------------------------------------------------------------------
# Sequence Normalisation & Biological Parsing
# ---------------------------------------------------------------------------

_VALID_BASES = frozenset("ACGTN")


def _parse_fasta_text(text: str) -> List[Tuple[str, str, str]]:
    """Parse multi-record FASTA text into (accession, header, sequence)."""
    records: List[Tuple[str, str, str]] = []
    acc = header = None
    buf: List[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if acc is not None and buf:
                records.append((acc, header or "", "".join(buf)))
            header = line[1:]
            acc = header.split()[0] if header else "UNKNOWN"
            buf = []
        else:
            buf.append(line.upper())

    if acc is not None and buf:
        records.append((acc, header or "", "".join(buf)))
    return records


def _normalise_seq(seq: str) -> str:
    """Replace non-IUPAC characters with N; uppercase."""
    return "".join(c if c in _VALID_BASES else "N" for c in seq.upper())


def _sha256(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()


def _looks_synthetic(header: str) -> bool:
    kw = {"synthetic", "construct", "patent", "artificial", "recombinant expression", "vector"}
    low = header.lower()
    return any(k in low for k in kw)


def _parse_influenza_header(header: str, default_subtype: str = "") -> Dict[str, str]:
    """
    Extract structured influenza metadata from NCBI FASTA header:
    subtype, segment (substrand), host species, country/location, date, isolate_id.
    """
    meta = {
        "subtype": default_subtype,
        "segment": "",
        "host": "",
        "country": "",
        "collection_date": "",
        "isolate_id": "",
    }

    # 1. Subtype extraction
    sub_m = re.search(r"\(?(H\d{1,2}N\d{1,2})\)?", header, re.IGNORECASE)
    if sub_m:
        meta["subtype"] = sub_m.group(1).upper()

    # 2. Segment / substrand extraction
    seg_m = re.search(r"segment\s+([1-8])", header, re.IGNORECASE)
    if seg_m:
        seg_map = {
            "1": "PB2", "2": "PB1", "3": "PA", "4": "HA",
            "5": "NP",  "6": "NA",  "7": "M",  "8": "NS",
        }
        meta["segment"] = seg_map.get(seg_m.group(1), "")
    else:
        seg_named = re.search(
            r"\b(polymerase basic 2|pb2|polymerase basic 1|pb1|polymerase acidic|pa|"
            r"hemagglutinin|ha|nucleoprotein|np|neuraminidase|na|matrix|m1|m2|"
            r"nonstructural|ns1|nep|ns)\b",
            header,
            re.IGNORECASE,
        )
        if seg_named:
            name = seg_named.group(1).lower()
            if "pb2" in name or "polymerase basic 2" in name:
                meta["segment"] = "PB2"
            elif "pb1" in name or "polymerase basic 1" in name:
                meta["segment"] = "PB1"
            elif "pa" in name or "polymerase acidic" in name:
                meta["segment"] = "PA"
            elif "hemagglutinin" in name or name == "ha":
                meta["segment"] = "HA"
            elif "nucleoprotein" in name or name == "np":
                meta["segment"] = "NP"
            elif "neuraminidase" in name or name == "na":
                meta["segment"] = "NA"
            elif "matrix" in name or name in ("m", "m1", "m2"):
                meta["segment"] = "M"
            elif "nonstructural" in name or name in ("ns", "ns1", "nep"):
                meta["segment"] = "NS"

    # 3. Strain format: A/host/location/strain_id/year(subtype)
    strain_m = re.search(r"\((A/[^()]+(?:\([^()]+\))?[^()]*)\)", header) or re.search(r"\b(A/[A-Za-z0-9_\-\.\/]+)", header)
    if strain_m:
        s = strain_m.group(1)
        meta["isolate_id"] = s
        parts = [p.strip() for p in s.split("/")]
        if len(parts) >= 2:
            meta["host"] = parts[1]
        if len(parts) >= 3:
            meta["country"] = parts[2]
        ym = re.search(r"(19\d\d|20\d\d)", s)
        if ym:
            meta["collection_date"] = ym.group(1)

    # 4. Host fallback if not extracted from strain
    if not meta["host"]:
        low_h = header.lower()
        for kw in AVIAN_HOST_KEYWORDS:
            if kw in low_h:
                meta["host"] = kw.title()
                break

    return meta


# ---------------------------------------------------------------------------
# Manifest Fields & Checkpoint Manager
# ---------------------------------------------------------------------------

MANIFEST_FIELDS = [
    "record_id", "insdc_accession", "source", "organism_taxid",
    "subtype", "segment", "host", "country", "collection_date",
    "seq_len", "sha256", "pathotype", "label_confidence", "split",
]


class IngestCheckpoint:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict = {"completed_subtypes": [], "stats": {}}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Could not read checkpoint file %s: %s", path, e)

    def is_completed(self, key: str) -> bool:
        return key in self.data.get("completed_subtypes", [])

    def record_completed(self, key: str, count: int) -> None:
        if "completed_subtypes" not in self.data:
            self.data["completed_subtypes"] = []
        if key not in self.data["completed_subtypes"]:
            self.data["completed_subtypes"].append(key)
        self.data.setdefault("stats", {})[key] = count
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")


def write_manifest(rows: List[Dict], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Manifest updated: %s  (%d total sequences)", manifest_path, len(rows))


# ---------------------------------------------------------------------------
# Processing Raw Batch into Canonical FASTA + Manifest
# ---------------------------------------------------------------------------

def _process_fasta_batch(
    fasta_text: str,
    canonical_fasta_dir: Path,
    seen_sha256: Set[str],
    seen_accessions: Set[str],
    min_len: int,
    max_len: int,
    max_n_fraction: float,
    default_subtype: str = "",
) -> List[Dict]:
    """Normalize, filter, and write canonical FASTA files; return manifest rows."""
    new_rows: List[Dict] = []
    records = _parse_fasta_text(fasta_text)

    for acc, header, seq in records:
        if acc in seen_accessions:
            continue
        if _looks_synthetic(header):
            continue

        seq_norm = _normalise_seq(seq)
        if not (min_len <= len(seq_norm) <= max_len):
            continue

        n_frac = seq_norm.count("N") / max(len(seq_norm), 1)
        if n_frac > max_n_fraction:
            continue

        sha = _sha256(seq_norm)
        if sha in seen_sha256:
            continue

        meta = _parse_influenza_header(header, default_subtype=default_subtype)

        # Write canonical FASTA
        out_fasta = canonical_fasta_dir / f"{acc}.fasta"
        out_fasta.write_text(f">{acc} {header}\n{seq_norm}\n", encoding="utf-8")

        seen_sha256.add(sha)
        seen_accessions.add(acc)

        new_rows.append({
            "record_id": acc,
            "insdc_accession": acc,
            "source": "ncbi",
            "organism_taxid": 11320,
            "subtype": meta["subtype"] or default_subtype,
            "segment": meta["segment"],
            "host": meta["host"],
            "country": meta["country"],
            "collection_date": meta["collection_date"],
            "seq_len": len(seq_norm),
            "sha256": sha,
            "pathotype": "",       # filled in label_qc stage
            "label_confidence": "",
            "split": "",           # filled in accession_split stage
        })

    return new_rows


# ---------------------------------------------------------------------------
# Streaming Ingestion Routines
# ---------------------------------------------------------------------------

def stream_ncbi_query(
    query: str,
    email: str,
    api_key: Optional[str],
    delay: float,
    min_len: int,
    max_len: int,
    max_n_fraction: float,
    raw_staging_dir: Path,
    canonical_fasta_dir: Path,
    seen_sha256: Set[str],
    seen_accessions: Set[str],
    limit: Optional[int] = None,
    tag: str = "batch",
    default_subtype: str = "",
) -> List[Dict]:
    """Execute an NCBI query via history server and stream sequences in batches."""
    logger.info("[NCBI Stream] Query: %s", query)
    count, webenv, query_key = _ncbi_esearch_history(query, email, api_key, delay=delay)
    logger.info("[NCBI Stream] Query returned %d total hits on NCBI history server", count)

    if count == 0:
        return []

    fetch_total = min(count, limit) if limit else count
    logger.info("[NCBI Stream] Will stream %d sequences (tag: %s) in batches of %d", fetch_total, tag, _RETMAX)

    raw_staging_dir.mkdir(parents=True, exist_ok=True)
    canonical_fasta_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    retstart = 0

    while retstart < fetch_total:
        batch_size = min(_RETMAX, fetch_total - retstart)
        batch_num = retstart // _RETMAX + 1
        logger.info("[NCBI Stream: %s] Fetching records %d..%d of %d", tag, retstart, retstart + batch_size, fetch_total)

        fasta_text = _ncbi_efetch_fasta_history(
            webenv, query_key, retstart, batch_size, email, api_key, delay=delay
        )
        if not fasta_text:
            logger.warning("[NCBI Stream: %s] Empty FASTA returned at %d; skipping batch", tag, retstart)
            retstart += batch_size
            continue

        # Save raw batch dump
        raw_file = raw_staging_dir / f"ncbi_{tag}_batch_{batch_num:04d}.fasta"
        raw_file.write_text(fasta_text, encoding="utf-8")

        # Process and save canonical
        batch_rows = _process_fasta_batch(
            fasta_text=fasta_text,
            canonical_fasta_dir=canonical_fasta_dir,
            seen_sha256=seen_sha256,
            seen_accessions=seen_accessions,
            min_len=min_len,
            max_len=max_len,
            max_n_fraction=max_n_fraction,
            default_subtype=default_subtype,
        )
        rows.extend(batch_rows)
        retstart += batch_size

    logger.info("[NCBI Stream: %s] Completed. Accepted %d new unique sequences.", tag, len(rows))
    return rows


def ingest_all_subtypes(
    cfg: Dict,
    canonical_fasta_dir: Path,
    raw_staging_dir: Path,
    seen_sha256: Set[str],
    seen_accessions: Set[str],
    subtypes_list: List[str],
    per_subtype_limit: int = 500,
    checkpoint: Optional[IngestCheckpoint] = None,
    manifest_path: Optional[Path] = None,
    existing_rows: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Download across all catalogued avian influenza subtypes."""
    ncbi_cfg = cfg["ingest"]["ncbi"]
    email = ncbi_cfg.get("email") or os.environ.get("NCBI_EMAIL", "vishnuvarthansenthilkumar@gmail.com")
    api_key = ncbi_cfg.get("api_key") or os.environ.get("NCBI_API_KEY")
    delay = ncbi_cfg.get("inter_request_delay_s", 0.34)
    min_len = cfg["ingest"]["min_len"]
    max_len = cfg["ingest"]["max_len"]
    max_n_frac = cfg["qc"]["max_n_fraction"]

    all_new_rows: List[Dict] = []

    logger.info("==================================================================")
    logger.info("STARTING MULTI-SUBTYPE INGEST: %d subtypes, target %d seqs/subtype", len(subtypes_list), per_subtype_limit)
    logger.info("==================================================================")

    for idx, subtype in enumerate(subtypes_list, 1):
        if checkpoint and checkpoint.is_completed(f"subtype_{subtype}"):
            logger.info("[%d/%d] Subtype %s already marked completed in checkpoint; skipping.", idx, len(subtypes_list), subtype)
            continue

        logger.info("[%d/%d] Ingesting subtype %s ...", idx, len(subtypes_list), subtype)
        query = f"txid11320[Organism:exp] AND {subtype}[All Fields] AND {min_len}:{max_len}[SLEN]"

        new_rows = stream_ncbi_query(
            query=query,
            email=email,
            api_key=api_key,
            delay=delay,
            min_len=min_len,
            max_len=max_len,
            max_n_fraction=max_n_frac,
            raw_staging_dir=raw_staging_dir,
            canonical_fasta_dir=canonical_fasta_dir,
            seen_sha256=seen_sha256,
            seen_accessions=seen_accessions,
            limit=per_subtype_limit,
            tag=subtype,
            default_subtype=subtype,
        )
        all_new_rows.extend(new_rows)

        if checkpoint:
            checkpoint.record_completed(f"subtype_{subtype}", len(new_rows))

        # Incremental manifest write to protect progress
        if manifest_path and existing_rows is not None and new_rows:
            existing_rows.extend(new_rows)
            write_manifest(existing_rows, manifest_path)

    return all_new_rows


def ingest_all_substrands(
    cfg: Dict,
    canonical_fasta_dir: Path,
    raw_staging_dir: Path,
    seen_sha256: Set[str],
    seen_accessions: Set[str],
    per_substrand_limit: int = 500,
    checkpoint: Optional[IngestCheckpoint] = None,
    manifest_path: Optional[Path] = None,
    existing_rows: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Download across all 8 viral RNA substrands/segments."""
    ncbi_cfg = cfg["ingest"]["ncbi"]
    email = ncbi_cfg.get("email") or os.environ.get("NCBI_EMAIL", "vishnuvarthansenthilkumar@gmail.com")
    api_key = ncbi_cfg.get("api_key") or os.environ.get("NCBI_API_KEY")
    delay = ncbi_cfg.get("inter_request_delay_s", 0.34)
    min_len = cfg["ingest"]["min_len"]
    max_len = cfg["ingest"]["max_len"]
    max_n_frac = cfg["qc"]["max_n_fraction"]

    all_new_rows: List[Dict] = []
    avian_hosts = " OR ".join(AVIAN_HOST_KEYWORDS)

    logger.info("==================================================================")
    logger.info("STARTING MULTI-SUBSTRAND INGEST: All 8 segments, target %d seqs/segment", per_substrand_limit)
    logger.info("==================================================================")

    for seg_name, seg_query in VIRAL_SUBSTRANDS:
        key = f"segment_{seg_name}"
        if checkpoint and checkpoint.is_completed(key):
            logger.info("Segment %s already marked completed in checkpoint; skipping.", seg_name)
            continue

        logger.info("Ingesting viral substrand / segment: %s ...", seg_name)
        query = f"txid11320[Organism:exp] AND ({avian_hosts}) AND ({seg_query}) AND {min_len}:{max_len}[SLEN]"

        new_rows = stream_ncbi_query(
            query=query,
            email=email,
            api_key=api_key,
            delay=delay,
            min_len=min_len,
            max_len=max_len,
            max_n_fraction=max_n_frac,
            raw_staging_dir=raw_staging_dir,
            canonical_fasta_dir=canonical_fasta_dir,
            seen_sha256=seen_sha256,
            seen_accessions=seen_accessions,
            limit=per_substrand_limit,
            tag=f"seg_{seg_name}",
        )
        all_new_rows.extend(new_rows)

        if checkpoint:
            checkpoint.record_completed(key, len(new_rows))

        if manifest_path and existing_rows is not None and new_rows:
            existing_rows.extend(new_rows)
            write_manifest(existing_rows, manifest_path)

    return all_new_rows


def ingest_download_all(
    cfg: Dict,
    canonical_fasta_dir: Path,
    raw_staging_dir: Path,
    seen_sha256: Set[str],
    seen_accessions: Set[str],
    limit: Optional[int] = None,
    manifest_path: Optional[Path] = None,
    existing_rows: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Connect and download all matching avian influenza sequences from NCBI."""
    ncbi_cfg = cfg["ingest"]["ncbi"]
    email = ncbi_cfg.get("email") or os.environ.get("NCBI_EMAIL", "vishnuvarthansenthilkumar@gmail.com")
    api_key = ncbi_cfg.get("api_key") or os.environ.get("NCBI_API_KEY")
    delay = ncbi_cfg.get("inter_request_delay_s", 0.34)
    min_len = cfg["ingest"]["min_len"]
    max_len = cfg["ingest"]["max_len"]
    max_n_frac = cfg["qc"]["max_n_fraction"]

    avian_hosts = " OR ".join(AVIAN_HOST_KEYWORDS)
    query = f"txid11320[Organism:exp] AND ({avian_hosts}) AND {min_len}:{max_len}[SLEN]"

    logger.info("==================================================================")
    logger.info("STARTING COMPREHENSIVE AVIAN INGESTION (Download All)")
    logger.info("Query: %s", query)
    logger.info("==================================================================")

    new_rows = stream_ncbi_query(
        query=query,
        email=email,
        api_key=api_key,
        delay=delay,
        min_len=min_len,
        max_len=max_len,
        max_n_fraction=max_n_frac,
        raw_staging_dir=raw_staging_dir,
        canonical_fasta_dir=canonical_fasta_dir,
        seen_sha256=seen_sha256,
        seen_accessions=seen_accessions,
        limit=limit,
        tag="all_avian",
    )

    if manifest_path and existing_rows is not None and new_rows:
        existing_rows.extend(new_rows)
        write_manifest(existing_rows, manifest_path)

    return new_rows


# ---------------------------------------------------------------------------
# Dry Run / Hit Count Inspection
# ---------------------------------------------------------------------------

def run_dry_run(cfg: Dict) -> None:
    """Print available sequence counts across all subtypes and substrands on NCBI."""
    ncbi_cfg = cfg["ingest"]["ncbi"]
    email = ncbi_cfg.get("email") or os.environ.get("NCBI_EMAIL", "vishnuvarthansenthilkumar@gmail.com")
    api_key = ncbi_cfg.get("api_key") or os.environ.get("NCBI_API_KEY")
    delay = ncbi_cfg.get("inter_request_delay_s", 0.34)
    min_len = cfg["ingest"]["min_len"]
    max_len = cfg["ingest"]["max_len"]

    print("\n" + "=" * 70)
    print("NCBI ENTREZ: AVIAN INFLUENZA HIT COUNT SURVEY")
    print("=" * 70)

    print("\n[1] Subtypes Catalogue:")
    for sub in ALL_AVIAN_SUBTYPES:
        q = f"txid11320[Organism:exp] AND {sub}[All Fields] AND {min_len}:{max_len}[SLEN]"
        cnt, _, _ = _ncbi_esearch_history(q, email, api_key, delay=delay)
        print(f"  {sub:<8} : {cnt:>7} sequences")

    print("\n[2] Substrands (Genomic Segments):")
    avian_hosts = " OR ".join(AVIAN_HOST_KEYWORDS)
    for seg_name, seg_q in VIRAL_SUBSTRANDS:
        q = f"txid11320[Organism:exp] AND ({avian_hosts}) AND ({seg_q}) AND {min_len}:{max_len}[SLEN]"
        cnt, _, _ = _ncbi_esearch_history(q, email, api_key, delay=delay)
        print(f"  Segment {seg_name:<4} : {cnt:>7} sequences")

    overall_q = f"txid11320[Organism:exp] AND ({avian_hosts}) AND {min_len}:{max_len}[SLEN]"
    total_cnt, _, _ = _ncbi_esearch_history(overall_q, email, api_key, delay=delay)
    print(f"\nTotal Avian Influenza Sequences in NCBI: {total_cnt:,}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Local Raw Files Ingestion (Preserved)
# ---------------------------------------------------------------------------

def ingest_from_raw_local(
    raw_dir: Path,
    canonical_fasta_dir: Path,
    seen_sha256: Set[str],
    seen_accessions: Set[str],
    min_len: int = 500,
    max_len: int = 2500,
    max_n_fraction: float = 0.05,
) -> List[Dict]:
    """Import existing raw FASTA files from dataset/raw/ into canonical format."""
    canonical_fasta_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    if not raw_dir.exists():
        return rows

    fasta_files = sorted(list(raw_dir.rglob("*.fasta")) + list(raw_dir.rglob("*.fa")))
    logger.info("[Local Raw] Scanning %d FASTA files from %s ...", len(fasta_files), raw_dir)

    for fp in fasta_files:
        parent_name = fp.parent.name
        subtype_hint = parent_name if re.match(r"^H\d+N\d+$", parent_name, re.I) else ""

        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        records = _parse_fasta_text(text)
        for acc, header, seq in records:
            if not acc or acc in seen_accessions:
                continue
            if _looks_synthetic(header):
                continue
            seq_norm = _normalise_seq(seq)
            if not (min_len <= len(seq_norm) <= max_len):
                continue
            n_frac = seq_norm.count("N") / max(len(seq_norm), 1)
            if n_frac > max_n_fraction:
                continue

            sha = _sha256(seq_norm)
            if sha in seen_sha256:
                continue

            meta = _parse_influenza_header(header, default_subtype=subtype_hint)

            out_fasta = canonical_fasta_dir / f"{acc}.fasta"
            out_fasta.write_text(f">{acc} {header}\n{seq_norm}\n", encoding="utf-8")

            seen_sha256.add(sha)
            seen_accessions.add(acc)

            rows.append({
                "record_id": acc,
                "insdc_accession": acc,
                "source": "local_raw",
                "organism_taxid": 11320,
                "subtype": meta["subtype"] or subtype_hint,
                "segment": meta["segment"],
                "host": meta["host"],
                "country": meta["country"],
                "collection_date": meta["collection_date"],
                "seq_len": len(seq_norm),
                "sha256": sha,
                "pathotype": "",
                "label_confidence": "",
                "split": "",
            })

    logger.info("[Local Raw] Ingested %d canonical sequences from %s", len(rows), raw_dir)
    return rows


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingest NCBI avian influenza sequences across all subtypes and substrands")
    p.add_argument("--config", type=Path, default=Path("configs/pipeline_config.yaml"))
    p.add_argument("--from-raw", action="store_true", help="Ingest exclusively from existing local dataset/raw files")
    p.add_argument("--live", action="store_true", help="Download live from NCBI public API")
    p.add_argument("--all-subtypes", action="store_true", help="Download across all catalogued avian influenza subtypes")
    p.add_argument("--all-substrands", "--all-segments", action="store_true", help="Download across all 8 viral RNA substrands/segments")
    p.add_argument("--download-all", action="store_true", help="Connect and download all matching avian sequences")
    p.add_argument("--subtypes", type=str, default=None, help="Comma-separated list of subtypes to download (e.g. H5N1,H7N9,H9N2)")
    p.add_argument("--per-subtype-limit", type=int, default=500, help="Max sequences to download per subtype (default: 500)")
    p.add_argument("--per-substrand-limit", type=int, default=500, help="Max sequences to download per substrand (default: 500)")
    p.add_argument("--limit", type=int, default=None, help="Cap maximum total records to download in this run")
    p.add_argument("--dry-run", action="store_true", help="Survey NCBI hit counts across all subtypes without downloading")
    args = p.parse_args(argv)

    with args.config.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if args.dry_run:
        run_dry_run(cfg)
        return 0

    root = Path(cfg["paths"]["dataset_root"])
    canonical_dir = Path(cfg["paths"]["canonical_dir"])
    staging_dir = Path(cfg["paths"]["staging_dir"])

    canonical_fasta_dir = canonical_dir / "fasta"
    manifest_path = canonical_dir / "manifest.csv"
    raw_ncbi = staging_dir / "ncbi"
    checkpoint_file = raw_ncbi / "ingest_checkpoint.json"

    checkpoint = IngestCheckpoint(checkpoint_file)

    # Load existing manifest to avoid duplicate work
    seen_sha256: Set[str] = set()
    seen_accessions: Set[str] = set()
    existing_rows: List[Dict] = []

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                existing_rows.append(row)
                seen_sha256.add(row["sha256"])
                seen_accessions.add(row["insdc_accession"])
        logger.info("Resuming: loaded %d existing manifest records", len(existing_rows))

    # 1. Local raw files import if requested
    should_ingest_local = args.from_raw or (not args.live and not manifest_path.exists() and (root / "raw").exists())
    if should_ingest_local:
        raw_local_dir = root / "raw"
        local_rows = ingest_from_raw_local(
            raw_local_dir,
            canonical_fasta_dir,
            seen_sha256,
            seen_accessions,
            min_len=cfg["ingest"]["min_len"],
            max_len=cfg["ingest"]["max_len"],
            max_n_fraction=cfg["qc"]["max_n_fraction"],
        )
        existing_rows.extend(local_rows)
        write_manifest(existing_rows, manifest_path)

    # 2. Live NCBI multi-subtype / multi-substrand / download-all ingestion
    if args.live:
        subtypes_to_fetch = (
            [s.strip().upper() for s in args.subtypes.split(",") if s.strip()]
            if args.subtypes
            else ALL_AVIAN_SUBTYPES
        )

        if args.all_subtypes or args.subtypes:
            ingest_all_subtypes(
                cfg=cfg,
                canonical_fasta_dir=canonical_fasta_dir,
                raw_staging_dir=raw_ncbi,
                seen_sha256=seen_sha256,
                seen_accessions=seen_accessions,
                subtypes_list=subtypes_to_fetch,
                per_subtype_limit=args.per_subtype_limit,
                checkpoint=checkpoint,
                manifest_path=manifest_path,
                existing_rows=existing_rows,
            )

        if args.all_substrands:
            ingest_all_substrands(
                cfg=cfg,
                canonical_fasta_dir=canonical_fasta_dir,
                raw_staging_dir=raw_ncbi,
                seen_sha256=seen_sha256,
                seen_accessions=seen_accessions,
                per_substrand_limit=args.per_substrand_limit,
                checkpoint=checkpoint,
                manifest_path=manifest_path,
                existing_rows=existing_rows,
            )

        if args.download_all:
            ingest_download_all(
                cfg=cfg,
                canonical_fasta_dir=canonical_fasta_dir,
                raw_staging_dir=raw_ncbi,
                seen_sha256=seen_sha256,
                seen_accessions=seen_accessions,
                limit=args.limit,
                manifest_path=manifest_path,
                existing_rows=existing_rows,
            )

        # Default fallback if only --live was passed: stream all avian sequences up to max_records or 2,000
        if not (args.all_subtypes or args.all_substrands or args.download_all or args.subtypes):
            limit = args.limit or cfg["ingest"].get("max_records") or 2000
            ingest_download_all(
                cfg=cfg,
                canonical_fasta_dir=canonical_fasta_dir,
                raw_staging_dir=raw_ncbi,
                seen_sha256=seen_sha256,
                seen_accessions=seen_accessions,
                limit=limit,
                manifest_path=manifest_path,
                existing_rows=existing_rows,
            )

    write_manifest(existing_rows, manifest_path)
    logger.info("Ingest stage complete. Canonical dataset now contains %d total sequences.", len(existing_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
