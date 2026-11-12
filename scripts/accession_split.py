"""
scripts/accession_split.py
==========================
Stage 3 — Leakage-free accession-level stratified train / val / test split.

Reads  : dataset/canonical/manifest.csv
Writes : dataset/splits/train_accessions.txt
         dataset/splits/val_accessions.txt
         dataset/splits/test_accessions.txt
         dataset/splits/split_manifest.csv    (manifest with 'split' filled)

Design
------
Groups are formed on ``isolate_id`` (approximated as the first 8 chars of the
INSDC accession + subtype).  Entire isolate groups move together so no
accession from the same isolate appears in both train and test sets.

Stratification is on ``subtype × pathotype`` to maintain class ratios.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("accession_split")


def _stratified_group_split(
    groups: Dict[str, List[str]],       # group_key → [accession, ...]
    strata: Dict[str, str],             # accession → stratum label
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split groups into train/val/test maintaining stratum distribution.
    Returns (train_accessions, val_accessions, test_accessions).
    """
    rng = random.Random(seed)

    # Bucket groups by their dominant stratum
    stratum_groups: Dict[str, List[str]] = defaultdict(list)
    for group_key, accs in groups.items():
        # Dominant stratum = most common stratum in this group
        counts: Dict[str, int] = defaultdict(int)
        for acc in accs:
            counts[strata.get(acc, "UNKNOWN")] += 1
        dominant = max(counts, key=lambda k: counts[k])
        stratum_groups[dominant].append(group_key)

    train_groups: List[str] = []
    val_groups:   List[str] = []
    test_groups:  List[str] = []

    for stratum, group_keys in stratum_groups.items():
        rng.shuffle(group_keys)
        n = len(group_keys)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        # Rest goes to test
        train_groups.extend(group_keys[:n_train])
        val_groups.extend(group_keys[n_train: n_train + n_val])
        test_groups.extend(group_keys[n_train + n_val:])

    def _expand(gk_list: List[str]) -> List[str]:
        accs: List[str] = []
        for gk in gk_list:
            accs.extend(groups[gk])
        return accs

    return _expand(train_groups), _expand(val_groups), _expand(test_groups)


def run_split(cfg: Dict) -> int:
    canonical_dir = Path(cfg["paths"]["canonical_dir"])
    splits_dir    = Path(cfg["paths"]["splits_dir"])
    manifest_path = canonical_dir / "manifest.csv"

    if not manifest_path.exists():
        logger.error("Manifest not found at %s. Run label_qc stage first.", manifest_path)
        return 1

    with manifest_path.open("r", encoding="utf-8") as fh:
        all_rows = list(csv.DictReader(fh))

    # Strictly filter for real biological isolates (never assign augmented records to test/val)
    real_rows = [
        r for r in all_rows
        if not r.get("source", "").startswith("augmented_") and "_aug" not in r.get("insdc_accession", "")
    ]

    logger.info("Loaded %d manifest rows (%d real biological isolates for splitting)", len(all_rows), len(real_rows))

    split_cfg = cfg.get("split", {})
    train_ratio = split_cfg.get("train_ratio", 0.70)
    val_ratio   = split_cfg.get("val_ratio",   0.15)
    seed        = split_cfg.get("seed",        42)
    group_by    = split_cfg.get("group_by",    "isolate_id")

    # Build groups: isolate proxy = first 8 chars of accession + subtype
    groups: Dict[str, List[str]] = defaultdict(list)
    strata: Dict[str, str] = {}  # accession → subtype_pathotype stratum

    for row in real_rows:
        acc     = row["insdc_accession"]
        subtype = row.get("subtype", "UNKNOWN")
        pathotype = row.get("pathotype", "INDETERMINATE")

        # Isolate proxy key
        group_key = acc[:8] + "_" + subtype
        groups[group_key].append(acc)

        strata[acc] = f"{subtype}_{pathotype}"

    logger.info("Found %d isolate groups across %d real sequences", len(groups), len(real_rows))

    train_accs, val_accs, test_accs = _stratified_group_split(
        dict(groups), strata, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    # Verify zero leakage
    train_set = set(train_accs)
    val_set   = set(val_accs)
    test_set  = set(test_accs)

    tv_overlap = train_set & val_set
    tt_overlap = train_set & test_set
    vt_overlap = val_set   & test_set
    if tv_overlap or tt_overlap or vt_overlap:
        logger.error(
            "LEAKAGE DETECTED! train∩val=%d  train∩test=%d  val∩test=%d",
            len(tv_overlap), len(tt_overlap), len(vt_overlap),
        )
        return 1
    logger.info(
        "Split: train=%d  val=%d  test=%d  (zero leakage verified)",
        len(train_accs), len(val_accs), len(test_accs),
    )

    splits_dir.mkdir(parents=True, exist_ok=True)

    # Write accession lists
    for name, accs in [("train", train_accs), ("val", val_accs), ("test", test_accs)]:
        path = splits_dir / f"{name}_accessions.txt"
        path.write_text("\n".join(accs), encoding="utf-8")
        logger.info("  %s → %s (%d accessions)", name, path, len(accs))

    # Update manifest with split column
    split_map: Dict[str, str] = {}
    for acc in train_accs: split_map[acc] = "train"
    for acc in val_accs:   split_map[acc] = "val"
    for acc in test_accs:  split_map[acc] = "test"

    for row in real_rows:
        row["split"] = split_map.get(row["insdc_accession"], "unassigned")

    split_manifest = splits_dir / "split_manifest.csv"
    fieldnames = list(real_rows[0].keys()) if real_rows else []
    with split_manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(real_rows)
    logger.info("Split manifest: %s (%d real records)", split_manifest, len(real_rows))

    # Also update canonical manifest in place
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(real_rows)
    logger.info("Canonical manifest updated: %s (%d real records)", manifest_path, len(real_rows))

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Accession-level stratified split")
    p.add_argument("--config", type=Path, default=Path("configs/pipeline_config.yaml"))
    args = p.parse_args(argv)

    with args.config.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    return run_split(cfg)


if __name__ == "__main__":
    sys.exit(main())

