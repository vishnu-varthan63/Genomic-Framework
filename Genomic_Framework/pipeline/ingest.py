"""Public INSDC ingest: NCBI Entrez (tax-constrained) + ENA Portal + BV-BRC metadata."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from xml.etree import ElementTree as ET

from Genomic_Framework.pipeline.taxonomy import (
    isolate_id_from_header,
    normalize_subtype,
    parse_segment,
)

logger = logging.getLogger(__name__)

NCBI_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ENA_SEARCH_URL = "https://www.ebi.ac.uk/ena/portal/api/search"
BVBRC_GENOME_URL = "https://www.bv-brc.org/api/genome/"

MANIFEST_FIELDS = [
    "record_id", "insdc_accession", "source", "organism_taxid", "subtype",
    "segment", "host", "country", "collection_date", "seq_len", "sha256",
    "header", "fasta_path", "isolate_id",
]


def _http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "GenomicFramework/2.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _clean_seq(seq: str) -> str:
    return "".join(ch for ch in seq.upper().replace("U", "T") if ch in "ACGTN")


def _sha256(seq: str) -> str:
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()


def _record(
    accession: str,
    sequence: str,
    source: str,
    header: str = "",
    host: str = "",
    country: str = "",
    collection_date: str = "",
    fasta_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    seq = _clean_seq(sequence)
    acc = accession.split(".")[0]
    rec = {
        "record_id": acc,
        "insdc_accession": acc,
        "source": source,
        "organism_taxid": "11320",
        "subtype": normalize_subtype(header) or normalize_subtype(accession) or "",
        "segment": parse_segment(header) or "",
        "host": host,
        "country": country,
        "collection_date": collection_date,
        "seq_len": len(seq),
        "sha256": _sha256(seq),
        "header": header.strip(),
        "fasta_path": "",
        "isolate_id": isolate_id_from_header(header, acc),
        "sequence": seq,
    }
    if fasta_dir is not None and seq:
        fasta_dir.mkdir(parents=True, exist_ok=True)
        path = fasta_dir / f"{acc}.fasta"
        path.write_text(f">{acc} {header}\n{seq}\n", encoding="utf-8")
        rec["fasta_path"] = str(path)
    return rec


class PublicSequenceIngestor:
    """Download avian Influenza A nucleotide records from free public APIs."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        paths = config.get("paths", {})
        ingest = config.get("ingest", {})
        self.dataset_root = Path(paths.get("dataset_root", "dataset"))
        self.staging_dir = Path(paths.get("staging_dir", self.dataset_root / "raw_sources"))
        self.canonical_fasta = Path(paths.get("canonical_dir", self.dataset_root / "canonical")) / "fasta"
        bulk = paths.get("bulk_staging_dir")
        if bulk:
            self.staging_dir = Path(bulk) / "raw_sources"
        self.taxid = int(ingest.get("taxid", 11320))
        self.host_taxid = int(ingest.get("host_taxid", 8782))
        self.min_len = int(ingest.get("min_len", 700))
        self.max_len = int(ingest.get("max_len", 2500))
        self.page_size = int(ingest.get("page_size", 200))
        self.max_records = ingest.get("max_records")
        self.delay = float(ingest.get("ncbi", {}).get("inter_request_delay_s", 0.34))
        ncbi = ingest.get("ncbi", {})
        self.email = ncbi.get("email") or os.environ.get("NCBI_EMAIL") or "research@example.com"
        self.api_key = ncbi.get("api_key") or os.environ.get("NCBI_API_KEY")
        self.sources = ingest.get("sources", {"ncbi": True, "ena": True, "bvbrc": True})

    def _cap_reached(self, n: int) -> bool:
        return self.max_records is not None and n >= int(self.max_records)

    def _ncbi_query(self) -> str:
        return (
            f"txid{self.taxid}[Organism:exp] AND txid{self.host_taxid}[Host] "
            f'AND ("2010/01/01"[PDAT] : "3000"[PDAT])'
        )

    def ingest(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.canonical_fasta.mkdir(parents=True, exist_ok=True)

        if self.sources.get("ncbi", True):
            records.extend(self._ingest_ncbi())
        if self.sources.get("ena", True) and not self._cap_reached(len(records)):
            remaining = None if self.max_records is None else int(self.max_records) - len(records)
            records.extend(self._ingest_ena(limit=remaining))
        if self.sources.get("bvbrc", True):
            self._ingest_bvbrc_metadata()

        self._write_raw_manifest(records)
        logger.info("Ingest collected %d raw records (pre-QC).", len(records))
        return records

    def _ncbi_params(self, extra: Dict[str, str]) -> str:
        params = {"email": self.email, "tool": "genomic_framework", **extra}
        if self.api_key:
            params["api_key"] = self.api_key
        return urllib.parse.urlencode(params)

    def _ingest_ncbi(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        query = self._ncbi_query()
        logger.info("NCBI Entrez search: %s", query)
        retstart = 0
        while True:
            if self._cap_reached(len(records)):
                break
            page = self.page_size
            if self.max_records is not None:
                page = min(page, int(self.max_records) - len(records))
            url = f"{NCBI_SEARCH_URL}?{self._ncbi_params({'db': 'nuccore', 'term': query, 'retmax': str(page), 'retstart': str(retstart), 'retmode': 'json'})}"
            try:
                payload = json.loads(_http_get(url).decode("utf-8"))
            except (urllib.error.URLError, json.JSONDecodeError) as exc:
                logger.error("NCBI esearch failed: %s", exc)
                break
            ids = payload.get("esearchresult", {}).get("idlist", [])
            if not ids:
                break
            batch = self._efetch_fasta(ids)
            records.extend(batch)
            retstart += len(ids)
            time.sleep(self.delay)
            if len(ids) < page:
                break
        (self.staging_dir / "ncbi_count.json").write_text(
            json.dumps({"n_records": len(records)}, indent=2), encoding="utf-8"
        )
        return records

    def _efetch_fasta(self, ids: List[str]) -> List[Dict[str, Any]]:
        url = f"{NCBI_FETCH_URL}?{self._ncbi_params({'db': 'nuccore', 'id': ','.join(ids), 'rettype': 'fasta', 'retmode': 'text'})}"
        try:
            text = _http_get(url).decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            logger.error("NCBI efetch failed: %s", exc)
            return []
        return list(self._parse_fasta_text(text, source="ncbi"))

    def _parse_fasta_text(self, text: str, source: str) -> Iterable[Dict[str, Any]]:
        header = ""
        chunks: List[str] = []
        for line in text.splitlines():
            if line.startswith(">"):
                if header and chunks:
                    rec = self._from_header_seq(header, "".join(chunks), source)
                    if rec:
                        yield rec
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
        if header and chunks:
            rec = self._from_header_seq(header, "".join(chunks), source)
            if rec:
                yield rec

    def _from_header_seq(self, header: str, sequence: str, source: str) -> Optional[Dict[str, Any]]:
        accession = header.split()[0] if header else ""
        seq = _clean_seq(sequence)
        if not accession or not seq:
            return None
        if not (self.min_len <= len(seq) <= self.max_len):
            return None
        return _record(accession, seq, source, header=header, fasta_dir=self.canonical_fasta)

    def _ingest_ena(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        fields = "accession,sequence,host,country,collection_date,strain,tag,description"
        query = f"tax_tree({self.taxid}) AND host_tax_id={self.host_taxid}"
        page = int(self.config.get("ingest", {}).get("ena", {}).get("page_size", 500))
        if limit is not None:
            page = min(page, max(0, limit))
        if page == 0:
            return []
        params = urllib.parse.urlencode({
            "result": "sequence",
            "query": query,
            "fields": fields,
            "format": "tsv",
            "limit": str(page),
        })
        url = f"{ENA_SEARCH_URL}?{params}"
        logger.info("ENA Portal search: %s", query)
        try:
            text = _http_get(url, timeout=120).decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            logger.warning("ENA search failed (continuing with NCBI only): %s", exc)
            return []
        records: List[Dict[str, Any]] = []
        lines = text.strip().splitlines()
        if len(lines) < 2:
            return records
        cols = lines[0].split("\t")
        for line in lines[1:]:
            parts = line.split("\t")
            row = {cols[i]: parts[i] if i < len(parts) else "" for i in range(len(cols))}
            acc = row.get("accession", "")
            seq = row.get("sequence", "")
            header = f"{acc} {row.get('description', '')} {row.get('strain', '')} {row.get('tag', '')}"
            rec = self._from_header_seq(header, seq, "ena")
            if rec is None:
                continue
            rec["host"] = row.get("host", "")
            rec["country"] = row.get("country", "")
            rec["collection_date"] = row.get("collection_date", "")
            rec["isolate_id"] = isolate_id_from_header(header, acc)
            records.append(rec)
            if self._cap_reached(len(records)):
                break
        return records

    def _ingest_bvbrc_metadata(self) -> None:
        query = urllib.parse.quote("taxon_id:11320 AND host_group:Avian")
        url = f"{BVBRC_GENOME_URL}?{urllib.parse.urlencode({'http_accept': 'application/json', 'limit': 100})}&eq({query})"
        dest = self.staging_dir / "bvbrc_genomes.json"
        try:
            payload = _http_get(
                f"{BVBRC_GENOME_URL}?http_accept=application/json&limit=50",
                timeout=60,
            )
            dest.write_bytes(payload)
            logger.info("Wrote BV-BRC metadata snapshot to %s", dest)
        except urllib.error.URLError as exc:
            logger.warning("BV-BRC metadata fetch skipped: %s", exc)

    def _write_raw_manifest(self, records: List[Dict[str, Any]]) -> Path:
        path = self.staging_dir / "raw_manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)
        return path


def write_fixture_dataset(root: Path, n_per_combo: int = 2) -> Path:
    """Offline synthetic avian-influenza-like FASTA + raw manifest for tests/smoke training."""
    from Genomic_Framework.pipeline.taxonomy import DEFAULT_SUBTYPES, SEGMENT_LABELS

    ingestor_cfg = {
        "paths": {
            "dataset_root": str(root),
            "staging_dir": str(root / "raw_sources"),
            "canonical_dir": str(root / "canonical"),
        },
        "ingest": {"min_len": 50, "max_len": 5000, "max_records": 10, "sources": {}},
    }
    fasta_dir = root / "canonical" / "fasta"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    (root / "raw_sources").mkdir(parents=True, exist_ok=True)

    lpai_tail = "CAAAGAGAAACAAGAGGATTATTT"  # QRETRGLF
    hpai_tail = "AGAAAAAAAAGAAGAGGACTATTT"  # RKKRRGLF
    records: List[Dict[str, Any]] = []
    idx = 0
    subtypes = DEFAULT_SUBTYPES[:4]
    segments = SEGMENT_LABELS
    for subtype in subtypes:
        for segment in segments:
            for patho, motif in (("LPAI", lpai_tail), ("HPAI", hpai_tail)):
                for k in range(n_per_combo):
                    idx += 1
                    acc = f"FIX{idx:06d}"
                    core = ("ATGC" * 80) + (motif if segment == "HA" else "ATGC" * 6)
                    host = "mallard" if patho == "LPAI" else "chicken"
                    header = (
                        f"{acc} Influenza A virus ({subtype}) segment {segment} "
                        f"{subtype} host={host}"
                    )
                    rec = _record(acc, core, "fixture", header=header, host=host, fasta_dir=fasta_dir)
                    rec["subtype"] = subtype
                    rec["segment"] = segment
                    rec["isolate_id"] = f"{subtype}_{patho}_{k}"
                    records.append(rec)

    ingestor = PublicSequenceIngestor(ingestor_cfg)
    ingestor.staging_dir = root / "raw_sources"
    ingestor._write_raw_manifest(records)
    return root
