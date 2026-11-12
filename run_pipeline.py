"""
run_pipeline.py
===============
Single-command orchestrator for the Avian Influenza Genomic Intelligence Framework.

Usage
-----
  python run_pipeline.py --stage all
  python run_pipeline.py --stage ingest --live --all-subtypes --per-subtype-limit 500
  python run_pipeline.py --stage ingest --live --all-substrands
  python run_pipeline.py --stage ingest --live --download-all --limit 5000
  python run_pipeline.py --stage ingest --dry-run
  python run_pipeline.py --stage qc
  python run_pipeline.py --stage split
  python run_pipeline.py --stage augment
  python run_pipeline.py --stage train
  python run_pipeline.py --stage eval

Stages
------
  ingest  → download & normalise NCBI sequences across all subtypes & substrands
  qc      → contaminant filter, HA cleavage labeling, manifest CSV
  split   → accession-level stratified train/val/test split (no leakage)
  augment → biological mutation operators on training split only
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
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_pipeline")

_ROOT = Path(__file__).resolve().parent

STAGES = ["ingest", "qc", "split", "augment", "train", "eval"]


def build_stage_command(name: str, args: Optional[argparse.Namespace] = None) -> List[str]:
    """Construct command line arguments for a pipeline stage."""
    base_commands = {
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
    cmd = list(base_commands[name])

    if name == "ingest" and args:
        if getattr(args, "live", False):
            cmd.append("--live")
        if getattr(args, "all_subtypes", False):
            cmd.append("--all-subtypes")
        if getattr(args, "all_substrands", False):
            cmd.append("--all-substrands")
        if getattr(args, "download_all", False):
            cmd.append("--download-all")
        if getattr(args, "subtypes", None):
            cmd.extend(["--subtypes", args.subtypes])
        if getattr(args, "per_subtype_limit", None):
            cmd.extend(["--per-subtype-limit", str(args.per_subtype_limit)])
        if getattr(args, "limit", None):
            cmd.extend(["--limit", str(args.limit)])
        if getattr(args, "dry_run", False):
            cmd.append("--dry-run")
        if getattr(args, "from_raw", False):
            cmd.append("--from-raw")

    return cmd


def run_stage(name: str, args: Optional[argparse.Namespace] = None) -> int:
    """Run a single pipeline stage as a subprocess; return exit code."""
    cmd = build_stage_command(name, args)
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
    # Forwarding flags for ingest
    p.add_argument("--live", action="store_true", help="Pass --live to ingest stage")
    p.add_argument("--all-subtypes", action="store_true", help="Download all avian influenza subtypes in ingest stage")
    p.add_argument("--all-substrands", "--all-segments", action="store_true", help="Download all 8 viral RNA substrands in ingest stage")
    p.add_argument("--download-all", action="store_true", help="Stream all matching avian sequences in ingest stage")
    p.add_argument("--subtypes", type=str, default=None, help="Comma-separated list of subtypes to download")
    p.add_argument("--per-subtype-limit", type=int, default=None, help="Max sequences per subtype")
    p.add_argument("--limit", type=int, default=None, help="Cap total sequences to download")
    p.add_argument("--dry-run", action="store_true", help="Run hit count survey without downloading")
    p.add_argument("--from-raw", action="store_true", help="Ingest from local raw directory")

    args = p.parse_args(argv)

    stages_to_run = STAGES if args.stage == "all" else [args.stage]

    if args.skip_ingest and "ingest" in stages_to_run:
        stages_to_run = [s for s in stages_to_run if s != "ingest"]
        logger.info("Skipping ingest stage (--skip-ingest flag set)")

    for stage in stages_to_run:
        rc = run_stage(stage, args)
        if rc != 0:
            logger.error("Pipeline aborted at stage '%s'", stage)
            return rc

    logger.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
