"""
ncbi_downloader.py
==================
Automated Batch Genomic Sequence Downloader
============================================

Downloads FASTA sequences for a predefined list of influenza subtypes from
the NCBI Nucleotide database using Biopython's Entrez API wrapper.

Key Features
------------
* Resume-capable  — already-downloaded batch files are skipped.
* Fault-tolerant  — per-chunk retry loop with exponential back-off.
* Rate-compliant  — enforces NCBI's recommended inter-request delay.
* Structured I/O  — one sub-directory per subtype, one file per batch.

Usage
-----
1. Set your NCBI email in the ``EMAIL`` constant below.
2. Run:  python ncbi_downloader.py

Dependencies
------------
    pip install biopython

Directory output layout
-----------------------
dataset/
└── raw/
    ├── H5N1/
    │   ├── batch_001.fasta
    │   ├── batch_002.fasta
    │   └── ...
    ├── H5N2/
    │   └── ...
    └── ...
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
try:
    from Bio import Entrez, SeqIO
except ImportError:
    sys.exit(
        "[FATAL] Biopython is not installed.\n"
        "        Install it with:  pip install biopython"
    )

# ===========================================================================
# ── CONFIGURATION  (edit these values before running) ──────────────────────
# ===========================================================================

EMAIL: str = "vishnuvarthangn@gmail.com"   # ← REQUIRED: NCBI policy mandates a valid e-mail
API_KEY: Optional[str] = None           # ← Optional: NCBI API key (10 req/s instead of 3)

# Subtypes to download
TARGET_SUBTYPES: List[str] = [
    "H5N1", "H5N2", "H5N3", "H5N6", "H5N8", "H5N9",
    "H7N1", "H7N2", "H7N3", "H7N7", "H7N9",
    "H9N2",
    "H10N7", "H10N8",
]

RETMAX: int = 1000          # Maximum total records to retrieve per subtype
BATCH_SIZE: int = 100       # Number of records per individual network call

BASE_DIR: Path = Path("dataset") / "raw"   # Root output directory

# Network resilience
MAX_RETRIES: int = 3            # Retry attempts per failed chunk
RETRY_DELAY_S: float = 5.0     # Seconds to wait before the first retry
BACKOFF_FACTOR: float = 2.0    # Each retry waits RETRY_DELAY * BACKOFF_FACTOR^attempt
INTER_REQUEST_DELAY_S: float = 0.5   # Mandatory sleep between successive API calls

# ===========================================================================
# ── LOGGING SETUP ──────────────────────────────────────────────────────────
# ===========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ncbi_download.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ===========================================================================
# ── HELPER UTILITIES ───────────────────────────────────────────────────────
# ===========================================================================

def _chunk(lst: List[str], size: int) -> Iterator[List[str]]:
    """
    Yield successive fixed-size sub-lists from *lst*.

    Parameters
    ----------
    lst  : list of str — the full list to partition.
    size : int         — maximum items per chunk.

    Yields
    ------
    list[str]
        Each chunk of at most *size* elements.
    """
    for start in range(0, len(lst), size):
        yield lst[start : start + size]


def _count_fasta_records(filepath: Path) -> int:
    """
    Count the number of FASTA records in an existing file without loading
    the full sequences into memory.

    Parameters
    ----------
    filepath : Path — path to a .fasta file that must already exist.

    Returns
    -------
    int
        Number of '>' header lines found in the file.
    """
    count = 0
    try:
        with filepath.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith(">"):
                    count += 1
    except OSError:
        count = 0
    return count


# ===========================================================================
# ── MAIN DOWNLOADER CLASS ──────────────────────────────────────────────────
# ===========================================================================

class NCBIDownloader:
    """
    Automated NCBI Nucleotide batch downloader for influenza subtypes.

    Workflow per subtype
    --------------------
    1. ``search()``      — Entrez.esearch → returns up to RETMAX GI/accession IDs.
    2. ``fetch_batch()`` — Entrez.efetch in BATCH_SIZE chunks → raw FASTA text.
    3. ``save_batch()``  — write FASTA text to disk; skip if file exists and
                           is non-empty (resume safety).

    Parameters
    ----------
    email    : str            — NCBI-required contact e-mail.
    api_key  : str | None     — Optional NCBI API key for higher rate limits.
    retmax   : int            — Maximum records to retrieve per subtype.
    batch_size : int          — Records per Entrez.efetch call.
    base_dir : Path           — Root directory for all output files.
    max_retries : int         — Network retry attempts per chunk.
    retry_delay : float       — Initial wait (seconds) before first retry.
    backoff_factor : float    — Multiplier applied to *retry_delay* on each attempt.
    inter_request_delay : float — Sleep between successive API calls.

    Examples
    --------
    >>> downloader = NCBIDownloader(email="user@example.com")
    >>> downloader.run(subtypes=["H5N1", "H7N9"])
    """

    def __init__(
        self,
        email: str = EMAIL,
        api_key: Optional[str] = API_KEY,
        retmax: int = RETMAX,
        batch_size: int = BATCH_SIZE,
        base_dir: Path = BASE_DIR,
        max_retries: int = MAX_RETRIES,
        retry_delay: float = RETRY_DELAY_S,
        backoff_factor: float = BACKOFF_FACTOR,
        inter_request_delay: float = INTER_REQUEST_DELAY_S,
    ) -> None:
        # ── Validate configuration ────────────────────────────────────────
        if not email or email == "your_email@example.com":
            logger.error(
                "[CONFIG] EMAIL is not set. "
                "NCBI requires a valid e-mail address. "
                "Edit the EMAIL constant at the top of ncbi_downloader.py."
            )
            sys.exit(1)

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {batch_size}.")
        if retmax < 1:
            raise ValueError(f"retmax must be >= 1; got {retmax}.")

        # ── Configure Biopython Entrez ────────────────────────────────────
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key

        self.email = email
        self.retmax = retmax
        self.batch_size = batch_size
        self.base_dir = Path(base_dir)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.inter_request_delay = inter_request_delay

        logger.info("NCBIDownloader initialised.")
        logger.info("  email      : %s", email)
        logger.info("  retmax     : %d", retmax)
        logger.info("  batch_size : %d", batch_size)
        logger.info("  base_dir   : %s", self.base_dir.resolve())

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, subtypes: Optional[List[str]] = None) -> None:
        """
        Execute the full download pipeline for all *subtypes*.

        Parameters
        ----------
        subtypes : list[str] | None
            Influenza subtype labels to process (e.g. ``["H5N1", "H7N9"]``).
            Defaults to the module-level ``TARGET_SUBTYPES`` list.
        """
        if subtypes is None:
            subtypes = TARGET_SUBTYPES

        logger.info("=" * 64)
        logger.info("Starting download run for %d subtype(s).", len(subtypes))
        logger.info("=" * 64)

        total_saved = 0
        total_skipped = 0
        failed_subtypes: List[str] = []

        for subtype_idx, subtype in enumerate(subtypes, start=1):
            logger.info(
                "[%d/%d] Processing subtype: %s",
                subtype_idx, len(subtypes), subtype,
            )

            try:
                saved, skipped = self._process_subtype(subtype)
                total_saved += saved
                total_skipped += skipped
            except Exception as exc:
                logger.error(
                    "[ERROR] Subtype '%s' failed with unexpected error: %s",
                    subtype, exc, exc_info=True,
                )
                failed_subtypes.append(subtype)

            logger.info("-" * 64)

        # ── Final summary ─────────────────────────────────────────────────
        logger.info("=" * 64)
        logger.info("Download run complete.")
        logger.info("  Batches saved   : %d", total_saved)
        logger.info("  Batches skipped : %d (already existed)", total_skipped)
        if failed_subtypes:
            logger.warning(
                "  Failed subtypes : %s", ", ".join(failed_subtypes)
            )
        logger.info("=" * 64)

    # ------------------------------------------------------------------
    # Per-subtype orchestration
    # ------------------------------------------------------------------

    def _process_subtype(self, subtype: str) -> tuple[int, int]:
        """
        Search, fetch, and save all batches for a single *subtype*.

        Parameters
        ----------
        subtype : str — Influenza subtype label (e.g. ``"H5N1"``).

        Returns
        -------
        (saved, skipped) : tuple[int, int]
            Number of batch files written vs. skipped for this subtype.
        """
        # ── 1. Create output directory ────────────────────────────────────
        subtype_dir = self.base_dir / subtype
        subtype_dir.mkdir(parents=True, exist_ok=True)

        # ── 2. Search NCBI ────────────────────────────────────────────────
        id_list = self.search(subtype)
        if not id_list:
            logger.warning(
                "[%s] No records returned from NCBI search. "
                "The subtype may not exist in the database.",
                subtype,
            )
            return 0, 0

        total_ids = len(id_list)
        n_batches = math.ceil(total_ids / self.batch_size)
        logger.info(
            "[%s] Found %d record ID(s) — will create %d batch file(s).",
            subtype, total_ids, n_batches,
        )

        # ── 3. Fetch & save each batch ────────────────────────────────────
        saved = 0
        skipped = 0

        for batch_num, chunk in enumerate(_chunk(id_list, self.batch_size), start=1):
            batch_filename = subtype_dir / f"batch_{batch_num:03d}.fasta"

            # Resume check — skip if file already exists and is non-empty
            if batch_filename.exists() and batch_filename.stat().st_size > 0:
                n_existing = _count_fasta_records(batch_filename)
                logger.info(
                    "[SKIP] %s already exists (%d sequences). Skipping.",
                    batch_filename, n_existing,
                )
                skipped += 1
                continue

            # Fetch the FASTA text for this chunk
            fasta_text = self.fetch_batch(
                id_list=chunk,
                subtype=subtype,
                batch_num=batch_num,
            )

            if fasta_text is None:
                logger.error(
                    "[FAIL] Could not retrieve batch %03d for %s after "
                    "%d retries. Skipping this chunk.",
                    batch_num, subtype, self.max_retries,
                )
                continue

            # Write to disk and log success
            self.save_batch(
                fasta_text=fasta_text,
                filepath=batch_filename,
                subtype=subtype,
                batch_num=batch_num,
            )
            saved += 1

            # Mandatory inter-request delay
            time.sleep(self.inter_request_delay)

        return saved, skipped

    # ------------------------------------------------------------------
    # Module 2 — Search & ID Retrieval
    # ------------------------------------------------------------------

    def search(self, subtype: str) -> List[str]:
        """
        Query NCBI Nucleotide for all records matching *subtype*.

        Uses ``Entrez.esearch`` with ``usehistory="y"`` to retrieve up to
        ``self.retmax`` GI IDs for the given subtype label.

        Parameters
        ----------
        subtype : str
            Influenza subtype string used as the search term
            (e.g. ``"H5N1"`` → query: ``'"H5N1"[All Fields]'``).

        Returns
        -------
        list[str]
            Unique sequence accession / GI IDs returned by NCBI.
            Returns an empty list if no results are found or if an
            error occurs.

        Notes
        -----
        The query is deliberately broad (All Fields) so that both
        partial and complete genome records are captured. Narrow the
        query by adding e.g. ``AND "complete genome"[Title]`` if only
        full-length sequences are required.
        """
        query = f'"{subtype}"[All Fields]'
        logger.info("[%s] Searching NCBI Nucleotide: %s", subtype, query)

        for attempt in range(1, self.max_retries + 1):
            try:
                handle = Entrez.esearch(
                    db="nucleotide",
                    term=query,
                    retmax=self.retmax,
                    usehistory="y",
                )
                record = Entrez.read(handle)
                handle.close()

                id_list: List[str] = record.get("IdList", [])
                count_found = int(record.get("Count", 0))

                logger.info(
                    "[%s] NCBI reports %d total matching records; "
                    "retrieved %d ID(s) (retmax=%d).",
                    subtype, count_found, len(id_list), self.retmax,
                )

                time.sleep(self.inter_request_delay)
                return id_list

            except Exception as exc:
                wait = self.retry_delay * (self.backoff_factor ** (attempt - 1))
                logger.warning(
                    "[%s] Search attempt %d/%d failed: %s. "
                    "Retrying in %.1f s...",
                    subtype, attempt, self.max_retries, exc, wait,
                )
                time.sleep(wait)

        logger.error(
            "[%s] All %d search attempts exhausted. Returning empty ID list.",
            subtype, self.max_retries,
        )
        return []

    # ------------------------------------------------------------------
    # Module 3 — Fault-Tolerant Batch Fetching
    # ------------------------------------------------------------------

    def fetch_batch(
        self,
        id_list: List[str],
        subtype: str,
        batch_num: int,
    ) -> Optional[str]:
        """
        Fetch FASTA sequences for a chunk of accession IDs from NCBI.

        Wraps ``Entrez.efetch`` in a retry loop with exponential back-off.
        Returns ``None`` if all retry attempts are exhausted.

        Parameters
        ----------
        id_list   : list[str] — Accession / GI IDs to fetch.
        subtype   : str       — Subtype label (used only for logging).
        batch_num : int       — Batch number (used only for logging).

        Returns
        -------
        str | None
            Raw FASTA-format text for all requested IDs, or ``None`` if
            the fetch failed after all retries.

        Notes
        -----
        NCBI's E-utilities rate limit is 3 requests/second without an API
        key and 10 requests/second with one.  The ``inter_request_delay``
        sleep in ``_process_subtype`` handles this; the sleep here is
        only for retry back-off.
        """
        id_str = ",".join(id_list)

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "[%s] batch %03d — fetching %d ID(s) (attempt %d/%d).",
                    subtype, batch_num, len(id_list), attempt, self.max_retries,
                )

                handle = Entrez.efetch(
                    db="nucleotide",
                    id=id_str,
                    rettype="fasta",
                    retmode="text",
                )
                fasta_text: str = handle.read()
                handle.close()

                # Sanity-check: a valid response must contain at least one '>'
                if not fasta_text.strip() or ">" not in fasta_text:
                    raise ValueError(
                        f"Empty or invalid FASTA response for batch {batch_num:03d}."
                    )

                return fasta_text

            except Exception as exc:
                wait = self.retry_delay * (self.backoff_factor ** (attempt - 1))
                logger.warning(
                    "[%s] batch %03d — fetch attempt %d/%d failed: %s. "
                    "Waiting %.1f s before retry...",
                    subtype, batch_num, attempt, self.max_retries, exc, wait,
                )
                time.sleep(wait)

        logger.error(
            "[%s] batch %03d — all %d fetch attempts failed.",
            subtype, batch_num, self.max_retries,
        )
        return None

    # ------------------------------------------------------------------
    # Module 4 — File I/O & Organization
    # ------------------------------------------------------------------

    def save_batch(
        self,
        fasta_text: str,
        filepath: Path,
        subtype: str,
        batch_num: int,
    ) -> None:
        """
        Write raw FASTA text to disk and log the outcome.

        Parameters
        ----------
        fasta_text : str  — Raw FASTA-format text (may contain multiple records).
        filepath   : Path — Absolute or relative destination file path.
        subtype    : str  — Subtype label (for logging).
        batch_num  : int  — Batch number (for logging).

        Side Effects
        ------------
        * Creates or overwrites *filepath* with *fasta_text*.
        * Logs a ``[SUCCESS]`` line with the sequence count.

        Raises
        ------
        OSError
            Propagated if the file cannot be written (e.g. permissions).
        """
        # Count records before writing so the log line is informative
        n_sequences = fasta_text.count(">")

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(fasta_text, encoding="utf-8")

        logger.info(
            "[SUCCESS] Saved %s  (Contains %d sequence%s).",
            filepath,
            n_sequences,
            "s" if n_sequences != 1 else "",
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"NCBIDownloader("
            f"email={self.email!r}, "
            f"retmax={self.retmax}, "
            f"batch_size={self.batch_size}, "
            f"base_dir={str(self.base_dir)!r})"
        )


# ===========================================================================
# ── ENTRY POINT ────────────────────────────────────────────────────────────
# ===========================================================================

def main() -> None:
    """
    Script entry point.

    Instantiates ``NCBIDownloader`` with the module-level configuration
    constants and runs the full download pipeline for all target subtypes.

    Edit ``EMAIL``, ``RETMAX``, and ``BATCH_SIZE`` at the top of this file
    before executing.
    """
    logger.info("Automated NCBI Genomic Sequence Downloader")
    logger.info("Biopython version : %s", __import__("Bio").__version__)

    downloader = NCBIDownloader(
        email=EMAIL,
        api_key=API_KEY,
        retmax=RETMAX,
        batch_size=BATCH_SIZE,
        base_dir=BASE_DIR,
        max_retries=MAX_RETRIES,
        retry_delay=RETRY_DELAY_S,
        backoff_factor=BACKOFF_FACTOR,
        inter_request_delay=INTER_REQUEST_DELAY_S,
    )

    downloader.run(subtypes=TARGET_SUBTYPES)


if __name__ == "__main__":
    main()
