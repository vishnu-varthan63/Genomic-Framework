"""
Genomic_Framework/pipeline/healthy_downloader.py
=================================================
Healthy Poultry Reference Genome Downloader
============================================

Downloads official NCBI reference genomes for five healthy poultry species
using the **NCBI Datasets REST API v2** exclusively — no CLI binary, no FTP
server.  Every network call goes to ``api.ncbi.nlm.nih.gov`` over HTTPS.

Species catalogue
-----------------
  Gallus gallus       (Chicken)        GCF_016699485.2
  Anas platyrhynchos  (Duck)           GCF_015476345.1
  Meleagris gallopavo (Turkey)         GCF_000146605.3
  Anser anser         (Goose)          GCF_011075055.1
  Coturnix japonica   (Japanese Quail) GCF_001577835.2

API endpoints used
------------------
  Download ZIP :
    GET https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{acc}/download
        ?include_annotation_type=GENOME_FASTA,GENOME_GFF,CDS_FASTA,PROT_FASTA,SEQUENCE_REPORT

  Metadata :
    GET https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{acc}/dataset_report

Output layout
-------------
dataset/healthy/
  chicken/  genome.fna  annotation.gff  cds.fna  protein.faa  metadata.json
  duck/     ...
  turkey/   ...
  goose/    ...
  quail/    ...
  download_summary.json

Design
------
* No CLI binary required — pure Python + requests.
* Resume-safe  — skips species where all four target files already exist.
* Retry logic  — up to MAX_RETRIES attempts with exponential back-off.
* Streaming    — genome ZIPs are streamed to disk (not loaded into RAM).
* Progress     — tqdm bytes bar during download + file extraction bar.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ===========================================================================
# CONFIGURATION
# ===========================================================================

_MODULE_DIR    = Path(__file__).resolve().parent   # pipeline/
_FRAMEWORK_DIR = _MODULE_DIR.parent                # Genomic_Framework/
PROJECT_ROOT   = _FRAMEWORK_DIR.parent             # project root

HEALTHY_DIR: Path = PROJECT_ROOT / "dataset" / "healthy"

# NCBI Datasets REST API v2
_API_BASE = "https://api.ncbi.nlm.nih.gov/datasets/v2"

# Data types to include in the download package
_INCLUDE_TYPES: List[str] = [
    "GENOME_FASTA",      # -> genomic .fna
    "GENOME_GFF",        # -> annotation .gff
    "CDS_FASTA",         # -> cds .fna
    "PROT_FASTA",        # -> protein .faa
    "SEQUENCE_REPORT",   # -> metadata report inside zip
]

# Retry / network parameters
MAX_RETRIES:        int   = 3
RETRY_DELAY_S:      float = 10.0
BACKOFF_FACTOR:     float = 2.0
CONNECT_TIMEOUT_S:  int   = 30
READ_TIMEOUT_S:     int   = 900     # large genomes can take 10-15 minutes

# HTTP headers that help NCBI identify the client
_HEADERS: Dict[str, str] = {
    "User-Agent":    "HealthyGenomeDownloader/2.0 (Python/requests; genomic-research)",
    "Accept":        "application/zip",
}

# Ordered list of (suffix_or_exact_name, dest_filename) pairs.
# More-specific patterns MUST appear before shorter/overlapping ones.
# e.g. "cds_from_genomic.fna" must precede "_genomic.fna" so the CDS
# file is not accidentally matched by the genome rule.
_FILE_PATTERNS: List[Tuple[str, str]] = [
    ("cds_from_genomic.fna",  "cds.fna"),        # exact CDS filename
    ("_genomic.fna",          "genome.fna"),      # primary genome assembly
    ("_genomic.gff",          "annotation.gff"), # GFF annotation
    ("protein.faa",           "protein.faa"),    # protein FASTA
]

# ---------------------------------------------------------------------------
# Species catalogue
# ---------------------------------------------------------------------------
SPECIES_CATALOGUE: List[Dict] = [
    {
        "label":      "chicken",
        "common":     "Chicken",
        "scientific": "Gallus gallus",
        "accession":  "GCF_016699485.2",
        "assembly":   "bGalGal1.mat.broiler.GRCg7b",
    },
    {
        "label":      "duck",
        "common":     "Duck",
        "scientific": "Anas platyrhynchos",
        "accession":  "GCF_015476345.1",
        "assembly":   "CAU_duck1.0",
    },
    {
        "label":      "turkey",
        "common":     "Turkey",
        "scientific": "Meleagris gallopavo",
        "accession":  "GCF_000146605.3",
        "assembly":   "Turkey_5.1",
    },
    {
        "label":      "goose",
        "common":     "Goose",
        "scientific": "Anser anser",
        "accession":  "GCF_011075055.1",
        "assembly":   "GooseV2.0",
    },
    {
        "label":      "quail",
        "common":     "Japanese Quail",
        "scientific": "Coturnix japonica",
        "accession":  "GCF_001577835.2",
        "assembly":   "Coturnix_japonica_2.0",
    },
]


# ===========================================================================
# MODULE-LEVEL UTILITIES
# ===========================================================================

def _sizeof_fmt(num_bytes: int) -> str:
    """Return a human-readable file-size string (e.g. '1.4 GB')."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes //= 1024          # type: ignore[assignment]
    return f"{num_bytes:.1f} TB"


def _count_fasta_sequences(filepath: Path) -> int:
    """Count '>' header lines in a FASTA file without reading it fully."""
    count = 0
    try:
        with filepath.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith(">"):
                    count += 1
    except OSError:
        pass
    return count


def _retry(func, *args, retries: int = MAX_RETRIES,
           delay: float = RETRY_DELAY_S, **kwargs):
    """
    Retry *func* up to *retries* times with exponential back-off.

    Returns the function's return value on first success.
    Raises the last exception if every attempt fails.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            wait = delay * (BACKOFF_FACTOR ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d failed: %s  -- retrying in %.0f s...",
                attempt, retries, exc, wait,
            )
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def _stream_download(url: str, dest: Path, desc: str,
                     params: Optional[Dict] = None) -> int:
    """
    Stream-download *url* to *dest* showing a tqdm bytes progress bar.

    Parameters
    ----------
    url    : str          -- Source URL.
    dest   : Path         -- Destination file path.
    desc   : str          -- Label for the tqdm bar.
    params : dict | None  -- Query-string parameters.

    Returns
    -------
    int -- Number of bytes written.
    """
    resp = requests.get(
        url,
        params=params,
        headers=_HEADERS,
        stream=True,
        timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
    )
    resp.raise_for_status()

    content_len = int(resp.headers.get("content-length", 0))

    written = 0
    with tqdm(
        total=content_len or None,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"  {desc[:42]}",
        leave=False,
        dynamic_ncols=True,
    ) as bar, dest.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=131_072):   # 128 KB chunks
            if chunk:
                fh.write(chunk)
                bar.update(len(chunk))
                written += len(chunk)

    return written


# ===========================================================================
# METADATA CLIENT
# ===========================================================================

class _NCBIMetadataClient:
    """
    Thin wrapper around the NCBI Datasets REST API v2 for assembly metadata.
    """

    def fetch_assembly_metadata(self, accession: str) -> Dict:
        """
        Fetch assembly-level metadata for *accession*.

        Parameters
        ----------
        accession : str -- RefSeq assembly accession, e.g. ``GCF_016699485.2``.

        Returns
        -------
        dict
            Parsed JSON body from NCBI, or an empty dict on failure.
        """
        url = f"{_API_BASE}/genome/accession/{accession}/dataset_report"
        try:
            resp = _retry(
                requests.get,
                url,
                headers={**_HEADERS, "Accept": "application/json"},
                timeout=(CONNECT_TIMEOUT_S, 60),
                retries=2,
                delay=5.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning(
                "Metadata fetch failed for %s: %s", accession, exc
            )
            return {}


# ===========================================================================
# MAIN DOWNLOADER CLASS
# ===========================================================================

class HealthyGenomeDownloader:
    """
    Downloads healthy poultry reference genomes from NCBI into a structured
    ``dataset/healthy/`` directory tree using the NCBI Datasets REST API.

    Parameters
    ----------
    healthy_dir  : Path            -- Root output directory.
    max_retries  : int             -- Download retry attempts per species.
    species_list : list[dict]|None -- Species to process (default: all 5).

    Examples
    --------
    >>> dl = HealthyGenomeDownloader()
    >>> dl.download_all()
    """

    def __init__(
        self,
        healthy_dir: Path = HEALTHY_DIR,
        max_retries: int = MAX_RETRIES,
        species_list: Optional[List[Dict]] = None,
    ) -> None:
        self.healthy_dir  = Path(healthy_dir)
        self.max_retries  = max_retries
        self.species_list = species_list or SPECIES_CATALOGUE
        self._meta_client = _NCBIMetadataClient()
        self._results: List[Dict] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def create_directories(self) -> None:
        """
        Create ``dataset/healthy/`` and one sub-directory per species.

        Safe to call multiple times (idempotent).
        """
        self.healthy_dir.mkdir(parents=True, exist_ok=True)
        for sp in self.species_list:
            sp_dir = self.healthy_dir / sp["label"]
            sp_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Directory ready: %s", sp_dir)
        logger.info("Directories ready: %s", self.healthy_dir.resolve())

    def download_species(self, species: Dict) -> Tuple[bool, str]:
        """
        Download, extract, and place all genome files for one *species*.

        Uses the NCBI Datasets API download endpoint to retrieve a ZIP
        archive, then extracts and renames its contents into the correct
        species subdirectory.

        Parameters
        ----------
        species : dict -- Catalogue entry with keys ``label``, ``accession``.

        Returns
        -------
        (success, status) : tuple[bool, str]
            status is one of ``"downloaded"``, ``"skipped"``, or a failure
            description.
        """
        label     = species["label"]
        accession = species["accession"]
        sp_dir    = self.healthy_dir / label

        logger.info(
            "==> [%s] %s  accession=%s",
            label.upper(), species["scientific"], accession,
        )

        # Resume check
        if self._all_files_present(sp_dir):
            logger.info("[SKIP] %s: all files already present.", label)
            return True, "skipped"

        # Build the API download URL and query parameters
        url = f"{_API_BASE}/genome/accession/{accession}/download"
        params = {
            "include_annotation_type": ",".join(_INCLUDE_TYPES),
            "filename":                f"{accession}.zip",
        }

        # Download into a temp directory (prevents partial-file corruption)
        with tempfile.TemporaryDirectory(prefix=f"ncbi_{label}_") as tmp_str:
            tmp_dir  = Path(tmp_str)
            zip_path = tmp_dir / f"{accession}.zip"

            logger.info(
                "[%s] Downloading from NCBI Datasets API...", label.upper()
            )
            try:
                _retry(
                    _stream_download,
                    url,
                    zip_path,
                    f"Downloading {species['common']}",
                    params,
                    retries=self.max_retries,
                    delay=RETRY_DELAY_S,
                )
            except Exception as exc:
                return False, f"Download failed: {exc}"

            size_mb = zip_path.stat().st_size / 1_048_576
            logger.info(
                "[%s] Archive received: %.1f MB", label.upper(), size_mb
            )

            # Extract
            logger.info("[%s] Extracting archive...", label.upper())
            extract_dir = tmp_dir / "extracted"
            try:
                self._extract_zip(zip_path, extract_dir, label)
            except Exception as exc:
                return False, f"Extraction failed: {exc}"

            # Rename and move to final location
            try:
                placed = self._rename_and_place(
                    extract_dir, sp_dir, accession, label
                )
                if not placed:
                    return False, "No expected files found inside the archive"
            except Exception as exc:
                return False, f"File placement failed: {exc}"

        return True, "downloaded"

    def verify_download(self, species: Dict) -> Tuple[bool, Dict]:
        """
        Verify integrity of all downloaded files for *species*.

        Checks
        ------
        1. File exists.
        2. File size > 0.
        3. FASTA files contain at least one ``>`` header.
        4. GFF file contains at least one non-comment data line.

        Parameters
        ----------
        species : dict -- Catalogue entry.

        Returns
        -------
        (passed, report) : tuple[bool, dict]
        """
        label  = species["label"]
        sp_dir = self.healthy_dir / label
        report: Dict = {}
        all_ok = True

        target_files = {
            "genome.fna":     "fasta",
            "annotation.gff": "gff",
            "cds.fna":        "fasta",
            "protein.faa":    "fasta",
        }

        for fname, ftype in target_files.items():
            fpath = sp_dir / fname
            entry: Dict = {"exists": fpath.exists()}

            if not fpath.exists():
                entry["status"] = "MISSING"
                all_ok = False
            elif fpath.stat().st_size == 0:
                entry.update({"status": "EMPTY", "size_bytes": 0})
                all_ok = False
            else:
                entry["size_bytes"] = fpath.stat().st_size

                if ftype == "fasta":
                    n = _count_fasta_sequences(fpath)
                    entry["sequences"] = n
                    entry["status"] = "OK" if n > 0 else "NO_SEQUENCES"
                    if n == 0:
                        all_ok = False
                elif ftype == "gff":
                    n = self._count_gff_data_lines(fpath)
                    entry["data_lines"] = n
                    entry["status"] = "OK" if n > 0 else "EMPTY_GFF"
                    if n == 0:
                        all_ok = False

            report[fname] = entry

        verdict = "PASS" if all_ok else "FAIL"
        logger.info("[%s] Verification: %s", label.upper(), verdict)
        for fname, e in report.items():
            size_str = (
                f"  ({_sizeof_fmt(e['size_bytes'])})" if "size_bytes" in e else ""
            )
            logger.info("    %-22s : %s%s", fname, e.get("status", "?"), size_str)

        return all_ok, report

    def rename_files(
        self, extract_dir: Path, sp_dir: Path, accession: str
    ) -> None:
        """
        Public alias for :meth:`_rename_and_place`.

        Parameters
        ----------
        extract_dir : Path -- Extracted archive root directory.
        sp_dir      : Path -- Destination species directory.
        accession   : str  -- Assembly accession string.
        """
        self._rename_and_place(extract_dir, sp_dir, accession, sp_dir.name)

    def save_metadata(self, species: Dict, verify_report: Dict) -> None:
        """
        Fetch NCBI metadata and write ``metadata.json`` into the species dir.

        Parameters
        ----------
        species       : dict -- Catalogue entry.
        verify_report : dict -- Output of :meth:`verify_download`.
        """
        label     = species["label"]
        accession = species["accession"]
        sp_dir    = self.healthy_dir / label

        raw = self._meta_client.fetch_assembly_metadata(accession)

        meta: Dict = {
            "species":            species["scientific"],
            "common_name":        species["common"],
            "label":              label,
            "assembly_accession": accession,
            "assembly_name":      species["assembly"],
            "source":             "NCBI RefSeq (Datasets API v2)",
            "download_date":      datetime.now(timezone.utc).isoformat(),
            "files": {
                fname: {
                    "size_bytes": e.get("size_bytes", 0),
                    "status":     e.get("status", "UNKNOWN"),
                }
                for fname, e in verify_report.items()
            },
        }

        # Enrich with API response data when available
        if raw:
            reports_list = raw.get("reports", [])
            if reports_list:
                asm   = reports_list[0].get("assembly_info", {})
                stats = reports_list[0].get("assembly_stats", {})
                meta.update({
                    "assembly_level":         asm.get("assembly_level", ""),
                    "refseq_category":        asm.get("refseq_category", ""),
                    "total_sequence_length":  stats.get("total_sequence_length"),
                    "number_of_chromosomes":  stats.get("number_of_chromosomes"),
                    "contig_n50":             stats.get("contig_n50"),
                    "scaffold_n50":           stats.get("scaffold_n50"),
                })

        meta_path = sp_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("[%s] metadata.json written.", label.upper())

    def download_all(self) -> None:
        """
        Run the full download pipeline for all configured species.

        Steps
        -----
        1. Create directory structure.
        2. For each species: download -> verify -> save metadata.
        3. Write ``download_summary.json``.
        4. Log final summary table.
        """
        logger.info("=" * 68)
        logger.info(
            "Healthy Poultry Genome Downloader  (%d species)", len(self.species_list)
        )
        logger.info("Download engine : NCBI Datasets REST API v2")
        logger.info("Output          : %s", self.healthy_dir.resolve())
        logger.info("=" * 68)

        self.create_directories()

        with tqdm(
            self.species_list,
            desc="Overall progress",
            unit="species",
            position=0,
            dynamic_ncols=True,
        ) as bar:
            for species in bar:
                label = species["label"]
                bar.set_postfix_str(label)

                entry: Dict = {
                    "species":            species["scientific"],
                    "common_name":        species["common"],
                    "label":              label,
                    "assembly_accession": species["accession"],
                    "assembly_name":      species["assembly"],
                    "download_date":      datetime.now(timezone.utc).isoformat(),
                    "status":             "pending",
                    "files":              {},
                    "error":              None,
                }

                # --- Download ---
                try:
                    ok, status = self.download_species(species)
                    entry["status"] = status if ok else "failed"
                except Exception as exc:
                    logger.error(
                        "[%s] Unhandled error: %s", label.upper(), exc,
                        exc_info=True,
                    )
                    entry["status"] = "error"
                    entry["error"]  = str(exc)
                    self._results.append(entry)
                    continue

                # --- Verify ---
                verify_report: Dict = {}
                try:
                    passed, verify_report = self.verify_download(species)
                    entry["files"]             = verify_report
                    entry["verification_pass"] = passed
                    if not passed and entry["status"] == "downloaded":
                        entry["status"] = "verification_failed"
                except Exception as exc:
                    logger.warning(
                        "[%s] Verification error: %s", label.upper(), exc
                    )
                    entry["verification_pass"] = False

                # --- Metadata ---
                try:
                    self.save_metadata(species, verify_report)
                except Exception as exc:
                    logger.warning(
                        "[%s] Metadata error: %s", label.upper(), exc
                    )

                # --- Record genome file size for summary ---
                genome_path = self.healthy_dir / label / "genome.fna"
                if genome_path.exists():
                    entry["genome_file_size_bytes"] = genome_path.stat().st_size

                self._results.append(entry)

                final = entry["status"]
                if final in ("downloaded", "skipped"):
                    logger.info(
                        "[OK] %s  --  %s", species["common"], final.upper()
                    )
                else:
                    logger.warning(
                        "[!!] %s  --  %s", species["common"], final.upper()
                    )

        self._write_summary()
        self._log_final_summary()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _all_files_present(self, sp_dir: Path) -> bool:
        """Return True iff all four target files exist and are non-empty."""
        for name in ("genome.fna", "annotation.gff", "cds.fna", "protein.faa"):
            p = sp_dir / name
            if not p.exists() or p.stat().st_size == 0:
                return False
        return True

    def _extract_zip(
        self, zip_path: Path, extract_dir: Path, label: str
    ) -> None:
        """
        Extract *zip_path* to *extract_dir* with a tqdm file-count bar.

        Parameters
        ----------
        zip_path    : Path -- Source ZIP.
        extract_dir : Path -- Destination root (created if absent).
        label       : str  -- Species label for tqdm description.
        """
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            with tqdm(
                members,
                desc=f"  Extracting {label}",
                unit="file",
                position=1,
                leave=False,
                dynamic_ncols=True,
            ) as bar:
                for member in bar:
                    zf.extract(member, extract_dir)

    def _rename_and_place(
        self,
        extract_dir: Path,
        sp_dir: Path,
        accession: str,
        label: str,
    ) -> bool:
        """
        Walk the extracted archive tree and copy files with standardised names
        into *sp_dir*.

        The NCBI Datasets API ZIP layout is::

            ncbi_dataset/
              data/
                {accession}/
                  {accession}_*_genomic.fna
                  {accession}_*_genomic.gff
                  cds_from_genomic.fna
                  protein.faa
              assembly_data_report.jsonl
              dataset_catalog.json

        Returns
        -------
        bool -- True if at least one expected file was placed.
        """
        sp_dir.mkdir(parents=True, exist_ok=True)
        placed = 0
        chosen: Dict[str, Path] = {}    # dest_name -> best source path

        for fpath in extract_dir.rglob("*"):
            if not fpath.is_file():
                continue
            name = fpath.name

            for pattern, dest_name in _FILE_PATTERNS:
                if name.endswith(pattern) or name == pattern:
                    # Prefer files whose name contains the accession ID
                    # (main assembly file vs. unplaced scaffold files)
                    prev = chosen.get(dest_name)
                    if prev is None:
                        chosen[dest_name] = fpath
                    elif accession in name and accession not in prev.name:
                        chosen[dest_name] = fpath
                    break

        for dest_name, src in chosen.items():
            dest = sp_dir / dest_name
            shutil.copy2(src, dest)
            logger.info(
                "[%s] Placed: %-22s  %s",
                label.upper(), dest_name, _sizeof_fmt(dest.stat().st_size),
            )
            placed += 1

        if placed == 0:
            all_files = [
                str(p) for p in extract_dir.rglob("*") if p.is_file()
            ]
            logger.error(
                "[%s] No matching files found in archive.\nContents:\n  %s",
                label.upper(), "\n  ".join(all_files),
            )

        return placed > 0

    @staticmethod
    def _count_gff_data_lines(gff_path: Path) -> int:
        """Count non-comment, non-blank lines in a GFF3 file."""
        count = 0
        try:
            with gff_path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    s = line.strip()
                    if s and not s.startswith("#"):
                        count += 1
        except OSError:
            pass
        return count

    def _write_summary(self) -> None:
        """Write ``dataset/healthy/download_summary.json``."""
        summary_path = self.healthy_dir / "download_summary.json"
        payload = {
            "generated_at":  datetime.now(timezone.utc).isoformat(),
            "download_engine": "NCBI Datasets REST API v2",
            "total_species": len(self._results),
            "species":       self._results,
        }
        summary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Summary written: %s", summary_path.resolve())

    def _log_final_summary(self) -> None:
        """Print a summary table to the log."""
        ok     = [r for r in self._results if r["status"] in ("downloaded", "skipped")]
        failed = [r for r in self._results if r["status"] not in ("downloaded", "skipped")]

        logger.info("")
        logger.info("=" * 68)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 68)
        logger.info("  %-10s  %-30s  %s", "Label", "Scientific Name", "Status")
        logger.info("  %s", "-" * 60)
        for r in self._results:
            logger.info(
                "  %-10s  %-30s  %s",
                r["label"], r["species"], r["status"].upper(),
            )
        logger.info("")
        logger.info("  Succeeded : %d / %d", len(ok), len(self._results))
        if failed:
            logger.warning("  Failed    : %d species", len(failed))
            for r in failed:
                logger.warning("    - %s  (%s)", r["label"], r.get("error") or r["status"])
        logger.info("=" * 68)

    def __repr__(self) -> str:
        return (
            f"HealthyGenomeDownloader("
            f"species={len(self.species_list)}, "
            f"healthy_dir={str(self.healthy_dir)!r})"
        )
