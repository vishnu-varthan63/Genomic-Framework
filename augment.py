"""
augment.py
==========
CLI entry point for the Genomic Data Augmentation Pipeline.

Usage
-----
    # Augment the full dataset (healthy + all viral subtypes)
    python augment.py

    # Dry-run: show what would be generated, write nothing
    python augment.py --dry-run

    # Augment a single subdirectory
    python augment.py --input-dir dataset/raw/H5N1 --output-dir dataset/raw/H5N1
                      --strategies random_mutation simulate_reassortment

    # Create a class-balanced dataset (target 5000 sequences per class)
    python augment.py --balance --target-size 5000

    # Augment only healthy genomes
    python augment.py --category healthy

CLI Arguments
-------------
    --dataset-root   PATH    Root of the dataset directory (default: dataset/)
    --input-dir      PATH    Single source directory (overrides dataset-root discovery)
    --output-dir     PATH    Output directory when --input-dir is used
    --strategies     STR...  Augmentation strategy names (space-separated)
    --category       STR     Restrict to "healthy" or "affected"
    --balance        FLAG    Run create_balanced_dataset()
    --target-size    INT     Target sequences per class when --balance is set
    --dry-run        FLAG    Log planned operations; write nothing to disk
    --seed           INT     Random seed (default: 42)
    --samples        INT     Augmented copies per original sequence (default: 3)

Output
------
    Augmented FASTA files written alongside originals as:
        aug_<strategy>_<original_filename>.fasta
    Metadata summary:
        dataset/augmentation_metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is importable when executed directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Genomic_Framework.pipeline.augmentor import (
    AUGMENTATION_CONFIG,
    GenomicAugmenter,
)

# ===========================================================================
# LOGGING
# ===========================================================================

_LOG_FILE = _PROJECT_ROOT / "augmentation.log"


def _configure_logging(log_file: Path = _LOG_FILE) -> None:
    fmt      = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    root     = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))

    root.addHandler(ch)
    root.addHandler(fh)


logger = logging.getLogger(__name__)

# ===========================================================================
# CLI PARSER
# ===========================================================================

ALL_STRATEGIES = [
    "reverse_complement", "random_mutation", "random_insertion",
    "random_deletion", "sliding_window", "sequence_shuffling",
    "simulate_reassortment", "simulate_recombination",
    "host_adaptation_simulation", "codon_optimization",
    "gc_content_shift", "transition_bias", "transversion_bias",
    "mixup", "cutmix",
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="augment.py",
        description="Genomic Sequence Data Augmentation Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset-root", type=Path, default=Path("dataset"),
        help="Root directory containing raw/ and healthy/ subdirectories.",
    )
    p.add_argument(
        "--input-dir", type=Path, default=None,
        help="Single source directory (skips full-dataset discovery).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (used with --input-dir).",
    )
    p.add_argument(
        "--strategies", nargs="+", default=None,
        choices=ALL_STRATEGIES + ["all"],
        metavar="STRATEGY",
        help="Augmentation strategy names. Use 'all' to run all strategies.",
    )
    p.add_argument(
        "--category", choices=["healthy", "affected", "all"], default="all",
        help="Which dataset category to augment.",
    )
    p.add_argument(
        "--balance", action="store_true",
        help="Run create_balanced_dataset() after augmentation.",
    )
    p.add_argument(
        "--target-size", type=int, default=5000,
        help="Target sequences per class when --balance is used.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Log planned operations without writing any files.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Master random seed.",
    )
    p.add_argument(
        "--samples", type=int, default=3,
        help="Augmented copies generated per original sequence.",
    )
    return p


# ===========================================================================
# MAIN
# ===========================================================================


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point.

    Returns
    -------
    int — 0 = success, 1 = error.
    """
    _configure_logging()
    args   = _build_parser().parse_args(argv)

    # Build config with CLI overrides
    cfg = AUGMENTATION_CONFIG.copy()
    if args.samples:
        cfg["healthy"]["samples_per_original"]  = args.samples
        cfg["affected"]["samples_per_original"] = args.samples

    augmenter = GenomicAugmenter(cfg, seed=args.seed)

    logger.info("=" * 64)
    logger.info("Genomic Data Augmentation Pipeline")
    logger.info("  Dataset root : %s", args.dataset_root.resolve())
    logger.info("  Dry-run      : %s", args.dry_run)
    logger.info("  Seed         : %d", args.seed)
    logger.info("=" * 64)

    # ---- Mode 1: single-directory augmentation ----
    if args.input_dir is not None:
        output_dir = args.output_dir or args.input_dir
        strategies = args.strategies or ["random_mutation"]
        if "all" in strategies:
            strategies = ALL_STRATEGIES

        logger.info(
            "Single-directory mode: %s → %s",
            args.input_dir, output_dir,
        )
        records = augmenter.augment_dataset(
            input_dir=args.input_dir,
            output_dir=output_dir,
            augmentation_strategies=strategies,
            samples_per_original=args.samples,
            dry_run=args.dry_run,
        )
        logger.info("Generated %d augmented sequences.", len(records))
        _print_summary_table(records, args)
        return 0

    # ---- Mode 2: full-dataset augmentation ----
    if args.category != "all":
        # Filter config to requested category only
        keep_keys = {"common"}
        if args.category == "healthy":
            keep_keys.add("healthy")
        else:
            keep_keys.add("affected")
        cfg = {k: v for k, v in cfg.items() if k in keep_keys}
        augmenter = GenomicAugmenter(cfg, seed=args.seed)

    # Override strategies if provided
    if args.strategies:
        strategies = args.strategies
        if "all" in strategies:
            strategies = ALL_STRATEGIES
        for key in ("healthy", "affected"):
            if key in cfg:
                cfg[key]["strategies"] = strategies

    try:
        summary = augmenter.augment_all(
            dataset_root=args.dataset_root,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        logger.error("Dataset error: %s", exc)
        return 1

    # ---- Mode 3: optional class-balancing ----
    if args.balance:
        raw_dir     = args.dataset_root / "raw"
        healthy_dir = args.dataset_root / "healthy"
        affected_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
        achieved = augmenter.create_balanced_dataset(
            healthy_dir=healthy_dir,
            affected_dirs=affected_dirs,
            target_size=args.target_size,
            output_dir=args.dataset_root / "balanced",
            dry_run=args.dry_run,
        )
        logger.info("Balancing achieved: %s", achieved)

    _print_full_summary(summary, args)
    return 0


def _print_summary_table(records: list, args: argparse.Namespace) -> None:
    """Print a compact summary of single-directory augmentation."""
    from collections import Counter
    strategy_counts = Counter(r.strategy for r in records)
    print()
    print("=" * 56)
    print("  AUGMENTATION SUMMARY")
    print("=" * 56)
    print(f"  Total sequences generated : {len(records)}")
    print(f"  Output directory          : {args.output_dir or args.input_dir}")
    print()
    print(f"  {'Strategy':<30}  {'Count':>6}")
    print(f"  {'-' * 38}")
    for strategy, cnt in sorted(strategy_counts.items()):
        print(f"  {strategy:<30}  {cnt:>6}")
    print("=" * 56)
    print()


def _print_full_summary(summary, args: argparse.Namespace) -> None:
    """Print a full run summary table."""
    print()
    print("=" * 56)
    print("  AUGMENTATION PIPELINE COMPLETE")
    print("=" * 56)
    print(f"  Total augmented sequences : {summary.total_augmented}")
    print(f"  Run timestamp             : {summary.run_timestamp}")
    print(f"  Dry-run                   : {args.dry_run}")
    print()
    if summary.strategy_counts:
        print(f"  {'Strategy':<30}  {'Count':>6}")
        print(f"  {'-' * 38}")
        for strategy, cnt in sorted(summary.strategy_counts.items()):
            print(f"  {strategy:<30}  {cnt:>6}")
    print()
    if not args.dry_run:
        meta_path = args.dataset_root / "augmentation_metadata.json"
        print(f"  Metadata JSON : {meta_path.resolve()}")
    print(f"  Log file      : {_LOG_FILE.resolve()}")
    print("=" * 56)
    print()


if __name__ == "__main__":
    sys.exit(main())
