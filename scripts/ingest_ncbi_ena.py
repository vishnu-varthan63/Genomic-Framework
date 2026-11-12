"""
scripts/ingest_ncbi_ena.py
==========================
Stage 1 — Collect all free public avian influenza nucleotide sequences.

Sources
-------
- NCBI Datasets / Virus API  : TaxID 11320, host Aves, all segments, paginated
- ENA Portal API             : tax_tree(11320) + avian host filter

Does NOT use Biopython (blocked by Application Control policy on this system).
Uses only the ``requests`` library + standard Python.

Output
------
  dataset/raw_sources/{ncbi,ena}/   — untouched dumps
  dataset/canonical/fasta/          — one FASTA per accession (normalised)
  dataset/canonical/manifest.csv    — unified metadata for all accepted records

Run
---
  python scripts/ingest_ncbi_ena.py --config configs/pipeline_config.yaml

Environment
-----------
  NCBI_EMAIL    required
  NCBI_API_KEY  optional (raises rate limit 3 → 10 req/s)
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
from typing import Dict, Generator, List, Optional, Tuple
from urllib.parse import urlencode

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
# NCBI Entrez REST (eutils) — no Biopython
# ---------------------------------------------------------------------------

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_RETMAX = 500          # records per NCBI efetch call


def _ncbi_esearch(
    query: str,
    email: str,
    api_key: Optional[str],
    delay: float = 0.34,
    retmax: int = 10_000,
) -> List[str]:
    """Return list of NCBI Nucleotide UIDs matching *query*."""
    params: Dict = {
        "db": "nucleotide",
        "term": query,
        "retmax": retmax,
        "usehistory": "y",
        "tool": "avian_genomic_intel",
        "email": email,
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{_EUTILS_BASE}/esearch.fcgi"
    logger.info("[NCBI esearch] query=%r  retmax=%d", query, retmax)
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    ids = data.get("esearchresult", {}).get("idlist", [])
    count = int(data.get("esearchresult", {}).get("count", 0))
    logger.info("[NCBI esearch] total hits=%d  returned=%d UIDs", count, len(ids))
    time.sleep(delay)
    return ids


def _ncbi_efetch_fasta_batch(
    uids: List[str],
    email: str,
    api_key: Optional[str],
    delay: float = 0.34,
) -> str:
    """Fetch FASTA text for a batch of UIDs (max ~500 at a time)."""
    params: Dict = {
        "db": "nucleotide",
        "id": ",".join(uids),
        "rettype": "fasta",
        "retmode": "text",
        "tool": "avian_genomic_intel",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{_EUTILS_BASE}/efetch.fcgi"
    resp = requests.post(url, data=params, timeout=120)
    resp.raise_for_status()
    time.sleep(delay)
    return resp.text


def _ncbi_efetch_summary(
    uids: List[str],
    email: str,
    api_key: Optional[str],
    delay: float = 0.34,
) -> List[Dict]:
    """Return DocSummary JSON for a batch of UIDs (for metadata)."""
    params: Dict = {
        "db": "nucleotide",
        "id": ",".join(uids),
        "retmode": "json",
        "tool": "avian_genomic_intel",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{_EUTILS_BASE}/esummary.fcgi"
    resp = requests.post(url, data=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    result = data.get("result", {})
    uids_ret = result.get("uids", [])
    summaries = [result[uid] for uid in uids_ret if uid in result]
    time.sleep(delay)
    return summaries


# ---------------------------------------------------------------------------
# ENA Portal REST
# ---------------------------------------------------------------------------

_ENA_PORTAL = "https://www.ebi.ac.uk/ena/portal/api"
_ENA_FASTA  = "https://www.ebi.ac.uk/ena/browser/api/fasta"


def _ena_search(
    taxid: int = 11320,
    min_len: int = 700,
    max_len: int = 2500,
    page_size: int = 500,
) -> Generator[Dict, None, None]:
    """Yield ENA search result dicts page by page."""
    offset = 0
    fields = "accession,description,tax_id,host,country,collection_date,base_count,strain"
    query = (
        f"tax_tree({taxid})"
        f" AND base_count>={min_len}"
        f" AND base_count<={max_len}"
    )
    while True:
        url = f"{_ENA_PORTAL}/search"
        params = {
            "result": "sequence",
            "query": query,
            "fields": fields,
            "format": "json",
            "limit": page_size,
            "offset": offset,
        }
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("[ENA] Request failed at offset %d: %s", offset, exc)
            break

        records = resp.json()
        if not records:
            break
        for rec in records:
            yield rec
        offset += page_size
        if len(records) < page_size:
            break
        time.sleep(0.5)


def _ena_fetch_fasta(accession: str, timeout: int = 30) -> Optional[str]:
    """Fetch single FASTA from ENA browser API."""
    url = f"{_ENA_FASTA}/{accession}"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        logger.warning("[ENA] FASTA fetch failed for %s: %s", accession, exc)
        return None


# ---------------------------------------------------------------------------
# FASTA normalisation
# ---------------------------------------------------------------------------

_VALID_BASES = frozenset("ACGTN")


def _parse_fasta_text(text: str) -> List[Tuple[str, str, str]]:
    """
    Parse multi-record FASTA text.
    Returns list of (accession, header, sequence).
    """
    records: List[Tuple[str, str, str]] = []
    acc = header = None
    buf: List[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if acc is not None and buf:
                records.append((acc, header, "".join(buf)))
            header = line[1:]
            acc = header.split()[0] if header else "UNKNOWN"
            buf = []
        else:
            buf.append(line.upper())

    if acc is not None and buf:
        records.append((acc, header, "".join(buf)))
    return records


def _normalise_seq(seq: str) -> str:
    """Replace non-IUPAC chars with N; uppercase."""
    return "".join(c if c in _VALID_BASES else "N" for c in seq.upper())


def _sha256(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Metadata extraction helpers
# ---------------------------------------------------------------------------

_SUBTYPE_RE = re.compile(r"\b(H\d{1,2}N\d{1,2})\b", re.IGNORECASE)
_SEGMENT_MAP = {
    "1": "PB2", "2": "PB1", "3": "PA", "4": "HA",
    "5": "NP",  "6": "NA",  "7": "M", "8": "NS",
    "pb2": "PB2", "pb1": "PB1", "pa": "PA", "ha": "HA",
    "np": "NP",   "na": "NA",   "m": "M",   "ns": "NS",
    "hemagglutinin": "HA", "neuraminidase": "NA",
    "polymerase basic 2": "PB2", "polymerase basic 1": "PB1",
    "nucleoprotein": "NP", "matrix": "M", "nonstructural": "NS",
}
_SEG_RE = re.compile(
    r"segment\s+(\d)|"
    r"\b(pb2|pb1|pa|ha|np|na|m|ns|hemagglutinin|neuraminidase"
    r"|nucleoprotein|polymerase basic \d|matrix|nonstructural)\b",
    re.IGNORECASE,
)


def _extract_subtype(header: str) -> Optional[str]:
    m = _SUBTYPE_RE.search(header)
    return m.group(1).upper() if m else None


def _extract_segment(header: str) -> Optional[str]:
    for m in _SEG_RE.finditer(header):
        if m.group(1):
            return _SEGMENT_MAP.get(m.group(1))
        if m.group(2):
            return _SEGMENT_MAP.get(m.group(2).lower())
    return None


def _is_avian_host(host_str: str) -> bool:
    if not host_str:
        return True  # unknown host: include conservatively
    avian_kw = {
        "duck", "chicken", "goose", "quail", "pheasant", "turkey",
        "pigeon", "dove", "swan", "teal", "mallard", "egret", "heron",
        "gull", "seagull", "tern", "wild bird", "avian", "poultry",
        "peacock", "ostrich", "emu", "parrot", "sparrow", "crow",
        "falcon", "hawk", "eagle",
    }
    low = host_str.lower()
    return any(kw in low for kw in avian_kw)


def _is_viral_length(seq_len: int, min_len: int, max_len: int) -> bool:
    return min_len <= seq_len <= max_len


def _looks_synthetic(header: str) -> bool:
    kw = {"synthetic", "construct", "patent", "artificial", "recombinant expression"}
    low = header.lower()
    return any(k in low for k in kw)


# ---------------------------------------------------------------------------
# Main ingest logic
# ---------------------------------------------------------------------------

MANIFEST_FIELDS = [
    "record_id", "insdc_accession", "source", "organism_taxid",
    "subtype", "segment", "host", "country", "collection_date",
    "seq_len", "sha256", "pathotype", "label_confidence", "split",
]


def ingest_ncbi(
    cfg: Dict,
    canonical_fasta_dir: Path,
    raw_staging_dir: Path,
    seen_sha256: set,
    seen_accessions: set,
) -> List[Dict]:
    """Download NCBI sequences matching config; return manifest rows."""
    ncbi_cfg = cfg["ingest"]["ncbi"]
    email = ncbi_cfg.get("email") or os.environ.get("NCBI_EMAIL", "")
    api_key = ncbi_cfg.get("api_key") or os.environ.get("NCBI_API_KEY")
    delay = ncbi_cfg.get("inter_request_delay_s", 0.34)
    min_len = cfg["ingest"]["min_len"]
    max_len = cfg["ingest"]["max_len"]

    if not email:
        logger.warning("[NCBI] NCBI_EMAIL not set. Set env var NCBI_EMAIL for polite access.")
        email = "anonymous@example.com"

    # Build query: TaxID 11320 AND Aves host AND length filter
    # Using eSearch length filter with SLEN field
    query = (
        f"txid11320[Organism:exp] AND aves[Host] "
        f"AND {min_len}:{max_len}[SLEN]"
    )

    # Optionally constrain by date
    date_from = cfg["ingest"].get("date_from")
    if date_from:
        year_from = date_from[:4]
        query += f' AND {year_from}:2026[PDAT]'

    uids = _ncbi_esearch(query, email, api_key, delay=delay)
    if not uids:
        logger.warning("[NCBI] No UIDs returned. Trying broader query.")
        query_broad = f"txid11320[Organism:exp] AND {min_len}:{max_len}[SLEN]"
        uids = _ncbi_esearch(query_broad, email, api_key, delay=delay)

    max_records = cfg["ingest"].get("max_records")
    if max_records and isinstance(max_records, int):
        uids = uids[:max_records]
        logger.info("[NCBI] Capped to %d UIDs via max_records setting", len(uids))

    logger.info("[NCBI] Will fetch %d sequences in batches of %d", len(uids), _RETMAX)

    raw_staging_dir.mkdir(parents=True, exist_ok=True)
    canonical_fasta_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for batch_start in range(0, len(uids), _RETMAX):
        batch_uids = uids[batch_start: batch_start + _RETMAX]
        batch_num = batch_start // _RETMAX + 1
        logger.info("[NCBI] Fetching batch %d (%d UIDs)", batch_num, len(batch_uids))

        try:
            fasta_text = _ncbi_efetch_fasta_batch(batch_uids, email, api_key, delay)
        except requests.RequestException as exc:
            logger.error("[NCBI] Batch %d fetch failed: %s", batch_num, exc)
            continue

        # Save raw dump
        raw_file = raw_staging_dir / f"ncbi_batch_{batch_num:04d}.fasta"
        raw_file.write_text(fasta_text, encoding="utf-8")

        records = _parse_fasta_text(fasta_text)
        for acc, header, seq in records:
            if acc in seen_accessions:
                continue
            if _looks_synthetic(header):
                continue
            seq_norm = _normalise_seq(seq)
            if not _is_viral_length(len(seq_norm), min_len, max_len):
                continue
            # N-fraction filter
            n_frac = seq_norm.count("N") / max(len(seq_norm), 1)
            if n_frac > cfg["qc"]["max_n_fraction"]:
                continue

            sha = _sha256(seq_norm)
            if sha in seen_sha256:
                continue

            subtype = _extract_subtype(header)
            segment = _extract_segment(header)

            # Save canonical FASTA
            out_fasta = canonical_fasta_dir / f"{acc}.fasta"
            out_fasta.write_text(f">{acc} {header}\n{seq_norm}\n", encoding="utf-8")

            seen_sha256.add(sha)
            seen_accessions.add(acc)

            rows.append({
                "record_id": acc,
                "insdc_accession": acc,
                "source": "ncbi",
                "organism_taxid": 11320,
                "subtype": subtype or "",
                "segment": segment or "",
                "host": "",
                "country": "",
                "collection_date": "",
                "seq_len": len(seq_norm),
                "sha256": sha,
                "pathotype": "",       # filled in label_qc stage
                "label_confidence": "",
                "split": "",           # filled in accession_split stage
            })

    logger.info("[NCBI] Accepted %d sequences", len(rows))
    return rows


def ingest_ena(
    cfg: Dict,
    canonical_fasta_dir: Path,
    raw_staging_dir: Path,
    seen_sha256: set,
    seen_accessions: set,
) -> List[Dict]:
    """Download ENA sequences; return manifest rows."""
    min_len = cfg["ingest"]["min_len"]
    max_len = cfg["ingest"]["max_len"]
    page_size = cfg["ingest"].get("ena", {}).get("page_size", 500)

    raw_staging_dir.mkdir(parents=True, exist_ok=True)
    canonical_fasta_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    logger.info("[ENA] Starting ENA Portal search (tax_tree(11320) + length filter)")

    for i, record in enumerate(_ena_search(taxid=11320, min_len=min_len, max_len=max_len, page_size=page_size)):
        acc = record.get("accession", "").strip()
        if not acc or acc in seen_accessions:
            continue
        host = record.get("host", "")
        if not _is_avian_host(host):
            continue

        seq_len_meta = int(record.get("base_count", 0))
        if seq_len_meta and not _is_viral_length(seq_len_meta, min_len, max_len):
            continue

        # Fetch FASTA
        fasta_text = _ena_fetch_fasta(acc)
        if not fasta_text:
            continue

        parsed = _parse_fasta_text(fasta_text)
        if not parsed:
            continue

        _, header, seq = parsed[0]
        if _looks_synthetic(header):
            continue

        seq_norm = _normalise_seq(seq)
        if not _is_viral_length(len(seq_norm), min_len, max_len):
            continue
        n_frac = seq_norm.count("N") / max(len(seq_norm), 1)
        if n_frac > cfg["qc"]["max_n_fraction"]:
            continue

        sha = _sha256(seq_norm)
        if sha in seen_sha256:
            continue

        subtype = _extract_subtype(header) or _extract_subtype(record.get("description", ""))
        segment = _extract_segment(header) or _extract_segment(record.get("description", ""))

        out_fasta = canonical_fasta_dir / f"{acc}.fasta"
        out_fasta.write_text(f">{acc} {header}\n{seq_norm}\n", encoding="utf-8")

        seen_sha256.add(sha)
        seen_accessions.add(acc)

        rows.append({
            "record_id": acc,
            "insdc_accession": acc,
            "source": "ena",
            "organism_taxid": 11320,
            "subtype": subtype or "",
            "segment": segment or "",
            "host": host,
            "country": record.get("country", ""),
            "collection_date": record.get("collection_date", ""),
            "seq_len": len(seq_norm),
            "sha256": sha,
            "pathotype": "",
            "label_confidence": "",
            "split": "",
        })

        max_records = cfg["ingest"].get("max_records")
        if max_records and isinstance(max_records, int) and len(rows) >= max_records:
            logger.info("[ENA] Reached cap of %d records", max_records)
            break

        if (i + 1) % 500 == 0:
            logger.info("[ENA] Processed %d ENA records so far (%d accepted)", i + 1, len(rows))

    logger.info("[ENA] Accepted %d sequences", len(rows))
    return rows


def ingest_from_raw_local(
    raw_dir: Path,
    canonical_fasta_dir: Path,
    seen_sha256: set,
    seen_accessions: set,
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
            if not _is_viral_length(len(seq_norm), min_len, max_len):
                continue
            n_frac = seq_norm.count("N") / max(len(seq_norm), 1)
            if n_frac > max_n_fraction:
                continue

            sha = _sha256(seq_norm)
            if sha in seen_sha256:
                continue

            subtype = _extract_subtype(header) or subtype_hint
            segment = _extract_segment(header)

            out_fasta = canonical_fasta_dir / f"{acc}.fasta"
            out_fasta.write_text(f">{acc} {header}\n{seq_norm}\n", encoding="utf-8")

            seen_sha256.add(sha)
            seen_accessions.add(acc)

            rows.append({
                "record_id": acc,
                "insdc_accession": acc,
                "source": "local_raw",
                "organism_taxid": 11320,
                "subtype": subtype or "",
                "segment": segment or "",
                "host": "",
                "country": "",
                "collection_date": "",
                "seq_len": len(seq_norm),
                "sha256": sha,
                "pathotype": "",
                "label_confidence": "",
                "split": "",
            })

    logger.info("[Local Raw] Ingested %d canonical sequences from %s", len(rows), raw_dir)
    return rows


def write_manifest(rows: List[Dict], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Manifest written: %s  (%d rows)", manifest_path, len(rows))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingest NCBI + ENA avian influenza sequences")
    p.add_argument("--config", type=Path, default=Path("configs/pipeline_config.yaml"))
    p.add_argument("--from-raw", action="store_true", help="Ingest exclusively from existing local dataset/raw files")
    p.add_argument("--live", action="store_true", help="Download live from NCBI and ENA public APIs")
    p.add_argument("--limit", type=int, default=None, help="Cap maximum records to download per source")
    args = p.parse_args(argv)

    with args.config.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if args.limit:
        cfg["ingest"]["max_records"] = args.limit

    root = Path(cfg["paths"]["dataset_root"])
    canonical_dir = Path(cfg["paths"]["canonical_dir"])
    staging_dir = Path(cfg["paths"]["staging_dir"])

    canonical_fasta_dir = canonical_dir / "fasta"
    manifest_path = canonical_dir / "manifest.csv"
    raw_ncbi = staging_dir / "ncbi"
    raw_ena = staging_dir / "ena"

    # Load existing manifest to avoid re-downloading
    seen_sha256: set = set()
    seen_accessions: set = set()
    existing_rows: List[Dict] = []

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                existing_rows.append(row)
                seen_sha256.add(row["sha256"])
                seen_accessions.add(row["insdc_accession"])
        logger.info("Resuming: found %d existing manifest entries", len(existing_rows))

    all_rows = list(existing_rows)

    # 1. Local raw ingestion if requested or bootstrapping an empty manifest without --live flag
    should_ingest_local = args.from_raw or (not args.live and not manifest_path.exists() and (root / "raw").exists())
    if should_ingest_local:
        raw_local_dir = root / "raw"
        logger.info("--- Local raw files ingest ---")
        local_rows = ingest_from_raw_local(
            raw_local_dir,
            canonical_fasta_dir,
            seen_sha256,
            seen_accessions,
            min_len=cfg["ingest"]["min_len"],
            max_len=cfg["ingest"]["max_len"],
            max_n_fraction=cfg["qc"]["max_n_fraction"],
        )
        all_rows.extend(local_rows)

    sources = cfg["ingest"].get("sources", {})

    # 2. Live remote sources (NCBI & ENA)
    should_ingest_live = args.live or (not args.from_raw and sources.get("ncbi", False))
    if should_ingest_live and sources.get("ncbi", False):
        logger.info("--- Live NCBI ingest ---")
        ncbi_rows = ingest_ncbi(cfg, canonical_fasta_dir, raw_ncbi, seen_sha256, seen_accessions)
        all_rows.extend(ncbi_rows)

    if should_ingest_live and sources.get("ena", False):
        logger.info("--- Live ENA ingest ---")
        ena_rows = ingest_ena(cfg, canonical_fasta_dir, raw_ena, seen_sha256, seen_accessions)
        all_rows.extend(ena_rows)

    write_manifest(all_rows, manifest_path)

    logger.info(
        "Ingest complete. Total accepted sequences: %d  (deduplicated)",
        len(all_rows),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

