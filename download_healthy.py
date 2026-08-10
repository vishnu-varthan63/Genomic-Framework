"""
download_healthy.py
===================
Project-root entry point for the Healthy Poultry Genome Downloader.

Downloads official NCBI reference genomes for five poultry species using
the NCBI Datasets REST API (api.ncbi.nlm.nih.gov). No CLI binary or FTP
access is required.

Usage
-----
    python download_healthy.py [OPTIONS]

Options
-------
  --species  LABEL [LABEL ...]
        Download only the listed species.
        Valid labels: chicken duck turkey goose quail

  --retries  N
        Maximum retry attempts per species (default: 3).

  --dry-run
        Create directories and show file status without downloading.

  --list-accessions
        Print the NCBI accession numbers and exit.

  -h / --help
        Show this message and exit.

Examples
--------
    # Download all 5 species
    python download_healthy.py

    # Download chicken and turkey only
    python download_healthy.py --species chicken turkey

    # Check what is already on disk (no network)
    python download_healthy.py --dry-run

    # Show accession table
    python download_healthy.py --list-accessions

Output
------
  dataset/healthy/
    chicken/  genome.fna  annotation.gff  cds.fna  protein.faa  metadata.json
    duck/     ...
    turkey/   ...
    goose/    ...
    quail/    ...
    download_summary.json
  healthy_download.log
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make project root importable when run directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Genomic_Framework.pipeline.healthy_downloader import (
    HEALTHY_DIR,
    MAX_RETRIES,
    SPECIES_CATALOGUE,
    HealthyGenomeDownloader,
    _sizeof_fmt,
)

# ===========================================================================
# LOGGING
# ===========================================================================

_LOG_FILE = _PROJECT_ROOT / "healthy_download.log"


def _configure_logging() -> None:
    """
    Attach INFO console handler and DEBUG file handler to the root logger.

    Both handlers use an identical timestamp format so the log file and
    the terminal output are easy to compare.
    """
    fmt      = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # ---- stdout ----
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))

    # ---- file ----
    fh = logging.FileHandler(_LOG_FILE, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))

    root.addHandler(ch)
    root.addHandler(fh)


logger = logging.getLogger(__name__)

# ===========================================================================
# ARGUMENT PARSING
# ===========================================================================

_VALID_LABELS = [sp["label"] for sp in SPECIES_CATALOGUE]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="download_healthy.py",
        description=(
            "Download healthy poultry reference genomes from NCBI "
            "(Datasets REST API v2)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--species",
        nargs="+",
        metavar="LABEL",
        choices=_VALID_LABELS,
        default=None,
        help=(
            "Species labels to download "
            f"(choices: {', '.join(_VALID_LABELS)}). "
            "Omit to download all."
        ),
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=MAX_RETRIES,
        metavar="N",
        help=f"Max retry attempts per species (default: {MAX_RETRIES}).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create directories and show file status without downloading.",
    )

    parser.add_argument(
        "--list-accessions",
        action="store_true",
        help="Print NCBI accession table and exit.",
    )

    return parser.parse_args(argv)


# ===========================================================================
# HELPERS
# ===========================================================================

def _print_accessions() -> None:
    """Print the species / accession reference table."""
    print()
    print(f"  {'Label':<10}  {'Common Name':<16}  {'Scientific Name':<25}  "
          f"{'Accession':<20}  Assembly")
    print(f"  {'-'*100}")
    for sp in SPECIES_CATALOGUE:
        print(
            f"  {sp['label']:<10}  {sp['common']:<16}  {sp['scientific']:<25}  "
            f"{sp['accession']:<20}  {sp['assembly']}"
        )
    print()


def _dry_run_report(species_list: list[dict]) -> None:
    """Print a per-file existence table without downloading anything."""
    target_files = ["genome.fna", "annotation.gff", "cds.fna", "protein.faa"]

    print()
    header = (f"  {'Label':<10}  {'Accession':<20}  "
              + "  ".join(f"{f:<18}" for f in target_files))
    print(header)
    print(f"  {'-'*100}")

    for sp in species_list:
        sp_dir = HEALTHY_DIR / sp["label"]
        cols = []
        for fname in target_files:
            fpath = sp_dir / fname
            if fpath.exists() and fpath.stat().st_size > 0:
                cols.append(f"OK({_sizeof_fmt(fpath.stat().st_size)})")
            else:
                cols.append("MISSING")

        print(
            f"  {sp['label']:<10}  {sp['accession']:<20}  "
            + "  ".join(f"{c:<18}" for c in cols)
        )
    print()


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main() -> int:
    """
    Main entry point — configures logging, parses arguments, and runs the
    appropriate action (dry-run, accession list, or full download).

    Returns
    -------
    int -- Exit code: 0 = all OK, 1 = partial failure, 2 = total failure.
    """
    _configure_logging()
    args = _parse_args()

    # --- List accessions and exit ---
    if args.list_accessions:
        _print_accessions()
        return 0

    # --- Filter species ---
    selected = (
        [sp for sp in SPECIES_CATALOGUE if sp["label"] in args.species]
        if args.species
        else SPECIES_CATALOGUE
    )

    if not selected:
        logger.error("No matching species found for: %s", args.species)
        return 2

    logger.info("=" * 60)
    logger.info("Healthy Poultry Genome Downloader")
    logger.info("Engine   : NCBI Datasets REST API v2 (no CLI required)")
    logger.info("Log file : %s", _LOG_FILE)
    logger.info("Output   : %s", HEALTHY_DIR.resolve())
    logger.info("Species  : %s", [sp["label"] for sp in selected])
    logger.info("=" * 60)

    # --- Dry-run mode ---
    if args.dry_run:
        logger.info("[DRY-RUN] No downloads will be performed.")
        # Still create directories so the user can see the intended structure
        for sp in selected:
            (HEALTHY_DIR / sp["label"]).mkdir(parents=True, exist_ok=True)
        _dry_run_report(selected)
        return 0

    # --- Full download ---
    downloader = HealthyGenomeDownloader(
        healthy_dir=HEALTHY_DIR,
        max_retries=args.retries,
        species_list=selected,
    )
    downloader.download_all()

    results = downloader._results
    if not results:
        return 2

    failed = [
        r for r in results
        if r["status"] not in ("downloaded", "skipped")
    ]
    if len(failed) == len(results):
        return 2
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
