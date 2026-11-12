"""
run_pipeline.py
===============
Single-command orchestrator for the Avian Influenza Genomic Intelligence Framework.

Usage
-----
  python run_pipeline.py --stage all
  python run_pipeline.py --stage ingest
  python run_pipeline.py --stage qc
  python run_pipeline.py --stage split
  python run_pipeline.py --stage train
  python run_pipeline.py --stage eval

Stages
------
  ingest  → download & normalise NCBI + ENA sequences into dataset/canonical/
  qc      → contaminant filter, HA cleavage labeling, manifest CSV
  split   → accession-level stratified train/val/test split (no leakage)
  train   → train multi-task model to best checkpoint
  eval    → evaluate best checkpoint on held-out test set, write metrics JSON

Environment variables
---------------------
  NCBI_EMAIL    required for NCBI Entrez polite access
  NCBI_API_KEY  optional; raises NCBI rate limit 3 → 10 req/s
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_pipeline")

_ROOT = Path(__file__).resolve().parent

STAGES = ["ingest", "qc", "split", "augment", "train", "eval"]

STAGE_COMMANDS = {
    "ingest":  [sys.executable, str(_ROOT / "scripts" / "ingest_ncbi_ena.py"),
                "--config", str(_ROOT / "configs" / "pipeline_config.yaml")],
    "qc":      [sys.executable, str(_ROOT / "scripts" / "label_qc.py"),
                "--config", str(_ROOT / "configs" / "pipeline_config.yaml")],
    "split":   [sys.executable, str(_ROOT / "scripts" / "accession_split.py"),
                "--config", str(_ROOT / "configs" / "pipeline_config.yaml")],
    "augment": [sys.executable, str(_ROOT / "scripts" / "augment_training_set.py"),
                "--config", str(_ROOT / "configs" / "pipeline_config.yaml")],
    "train":   [sys.executable, str(_ROOT / "train_advanced.py"),
                "--config", str(_ROOT / "configs" / "train_config.yaml")],
    "eval":    [sys.executable, str(_ROOT / "evaluate_advanced.py"),
                "--config", str(_ROOT / "configs" / "train_config.yaml")],
}


def run_stage(name: str) -> int:
    """Run a single pipeline stage as a subprocess; return exit code."""
    cmd = STAGE_COMMANDS[name]
    logger.info("=" * 60)
    logger.info("STAGE: %s", name.upper())
    logger.info("CMD  : %s", " ".join(cmd))
    logger.info("=" * 60)
    result = subprocess.run(cmd, cwd=str(_ROOT))
    if result.returncode != 0:
        logger.error("Stage '%s' failed with exit code %d", name, result.returncode)
    else:
        logger.info("Stage '%s' completed OK", name)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Avian Influenza pipeline orchestrator")
    p.add_argument(
        "--stage",
        choices=STAGES + ["all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    p.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingest stage even when running --stage all (useful when data is on disk)",
    )
    args = p.parse_args(argv)

    stages_to_run = STAGES if args.stage == "all" else [args.stage]

    if args.skip_ingest and "ingest" in stages_to_run:
        stages_to_run = [s for s in stages_to_run if s != "ingest"]
        logger.info("Skipping ingest stage (--skip-ingest flag set)")

    for stage in stages_to_run:
        rc = run_stage(stage)
        if rc != 0:
            logger.error("Pipeline aborted at stage '%s'", stage)
            return rc

    logger.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

