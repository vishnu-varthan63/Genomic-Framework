"""
scripts/augment_training_set.py
===============================
Leakage-Free Biological Augmentation of the Training Partition for Avian Influenza.

Strict Scientific & Data Hygiene Constraints:
----------------------------------------------
1. Augments ONLY sequences assigned to the 'train' split (pure real isolates).
2. Leaves validation and test sets 100% REAL, untouched, and unpolluted.
3. Zero data leakage: parent real sequence and all its augmented derivatives
   stay strictly within the training partition.
4. Uses realistic biological mutation operators for Influenza A:
   - Viral RdRp transition bias (purine-purine / pyrimidine-pyrimidine drift)
   - Avian host adaptation bias (codon drift across chicken, duck, goose, quail, turkey)
   - Transversion bias (low-frequency transversion substitutions)
   - Synonymous codon optimization (avian host translational efficiency)
   - Point mutation drift (low-rate viral genetic drift)
5. Protects polybasic cleavage site motifs in HPAI strains to maintain pathotype integrity.
6. Adaptive class & subtype balancing to enrich rare subtypes (e.g. H5N4, H5N9, H7N4, H14N5, H16N3).

Output:
-------
  dataset/canonical/fasta/{acc}_aug{i}_{strat}.fasta — augmented sequences
  dataset/canonical/manifest.csv                     — unified manifest with augmented rows
  dataset/splits/train_accessions.txt                — updated train accessions list
  dataset/splits/split_manifest.csv                  — updated split manifest

Usage:
------
  python scripts/augment_training_set.py --clean --copies 2 --adaptive
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Genomic_Framework.pipeline.augmentor import (
    AUGMENTATION_CONFIG,
    GenomicAugmenter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("augment_train")

_VALID_BASES = frozenset("ACGTN")
_BASIC_CODONS = frozenset({"CGT", "CGC", "CGA", "CGG", "AGA", "AGG", "AAA", "AAG"})


def _sha256(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()


def _read_fasta_seq(fpath: Path) -> Tuple[str, str]:
    """Return (header, sequence) from single FASTA file."""
    lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
    header = lines[0][1:].strip() if lines and lines[0].startswith(">") else fpath.stem
    seq = "".join(ln.strip().upper() for ln in lines if not ln.startswith(">"))
    return header, seq


def clean_existing_augmentations(fasta_dir: Path, manifest_path: Path, split_manifest_path: Path) -> int:
    """Remove previous augmented FASTA files and reset manifests to real-only rows."""
    logger.info("Cleaning previous augmented FASTA files from %s ...", fasta_dir)
    aug_files = list(fasta_dir.glob("*_aug*.fasta"))
    for f in aug_files:
        try:
            f.unlink()
        except OSError:
            pass
    logger.info("Removed %d previous augmented FASTA files.", len(aug_files))

    # Reset manifests to real isolates only
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        real_rows = [
            r for r in rows
            if not r.get("source", "").startswith("augmented_") and "_aug" not in r.get("insdc_accession", "")
        ]
        fieldnames = list(rows[0].keys()) if rows else []
        with manifest_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(real_rows)
        logger.info("Manifest reset to %d pure real isolates: %s", len(real_rows), manifest_path)

        if split_manifest_path.exists():
            with split_manifest_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(real_rows)
            logger.info("Split manifest reset to %d pure real isolates: %s", len(real_rows), split_manifest_path)

    return len(aug_files)


def run_augmentation(
    cfg: Dict,
    copies_per_seq: int = 2,
    adaptive: bool = True,
    clean: bool = True,
    seed: int = 42,
) -> int:
    canonical_dir = Path(cfg["paths"]["canonical_dir"])
    splits_dir = Path(cfg["paths"]["splits_dir"])
    manifest_path = canonical_dir / "manifest.csv"
    split_manifest_path = splits_dir / "split_manifest.csv"
    train_acc_file = splits_dir / "train_accessions.txt"
    val_acc_file = splits_dir / "val_accessions.txt"
    test_acc_file = splits_dir / "test_accessions.txt"
    fasta_dir = canonical_dir / "fasta"

    if clean:
        clean_existing_augmentations(fasta_dir, manifest_path, split_manifest_path)

    if not train_acc_file.exists() or not manifest_path.exists():
        logger.error("Required split files missing. Run accession_split stage first.")
        return 1

    # Load pure real training accessions
    raw_train_accs = [
        line.strip()
        for line in train_acc_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_accs = [acc for acc in raw_train_accs if "_aug" not in acc]
    logger.info("Loaded %d real training accessions to augment", len(train_accs))

    # Load existing manifest
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest_rows = list(csv.DictReader(fh))

    # Filter to real rows
    real_manifest_rows = [
        r for r in manifest_rows
        if not r.get("source", "").startswith("augmented_") and "_aug" not in r.get("insdc_accession", "")
    ]
    row_by_acc: Dict[str, Dict] = {r["insdc_accession"]: r for r in real_manifest_rows}
    seen_sha: Set[str] = {r["sha256"] for r in real_manifest_rows}

    # Count subtype distribution in training set for adaptive balancing
    train_subtypes = [row_by_acc[acc].get("subtype", "UNKNOWN") for acc in train_accs if acc in row_by_acc]
    subtype_counts = Counter(train_subtypes)
    logger.info("Training subtype distribution (top 10): %s", subtype_counts.most_common(10))

    augmenter = GenomicAugmenter(config=AUGMENTATION_CONFIG, seed=seed)
    rng = random.Random(seed)

    strategies = ["transition_bias", "host_adaptation", "transversion_bias", "codon_opt", "point_mut"]
    hosts = ["chicken", "duck", "goose", "quail", "turkey"]

    new_rows: List[Dict] = []
    new_train_accs: List[str] = []

    logger.info(
        "Beginning biological training set augmentation (base copies: %d, adaptive=%s) ...",
        copies_per_seq, adaptive
    )

    for idx, acc in enumerate(train_accs, 1):
        fpath = fasta_dir / f"{acc}.fasta"
        if not fpath.exists():
            continue

        try:
            header, seq = _read_fasta_seq(fpath)
        except OSError:
            continue

        if len(seq) < 200:
            continue

        base_meta = row_by_acc.get(acc, {})
        subtype = base_meta.get("subtype", "UNKNOWN")
        pathotype = base_meta.get("pathotype", "INDETERMINATE")

        # Determine number of copies adaptively
        num_copies = copies_per_seq
        if adaptive:
            st_count = subtype_counts.get(subtype, 0)
            if st_count < 150:
                num_copies = max(copies_per_seq, 3)
            elif st_count < 350:
                num_copies = max(copies_per_seq, 2)
            else:
                num_copies = min(copies_per_seq, 2)

            # Extra multiplier for HPAI pathotype to mitigate class imbalance
            if pathotype == "HPAI":
                num_copies = max(num_copies, 3)

        for copy_i in range(1, num_copies + 1):
            strat = strategies[(idx + copy_i) % len(strategies)]

            if strat == "host_adaptation":
                host_target = hosts[rng.randint(0, len(hosts) - 1)]
                aug_seq = augmenter.host_adaptation_simulation(seq, host=host_target, adaptation_rate=0.003)
                strat_tag = f"host_{host_target}"
            elif strat == "transition_bias":
                aug_seq = augmenter.transition_bias(seq, trans_rate=0.80, overall_rate=0.003)
                strat_tag = "trans_bias"
            elif strat == "transversion_bias":
                aug_seq = augmenter.transversion_bias(seq, transv_rate=0.80, overall_rate=0.002)
                strat_tag = "tv_bias"
            elif strat == "codon_opt":
                aug_seq = augmenter.codon_optimization(seq, target_organism="avian")
                strat_tag = "codon_opt"
            else:
                aug_seq = augmenter.random_mutation(seq, mutation_rate=0.002)
                strat_tag = "point_mut"

            if not aug_seq or len(aug_seq) < 100:
                continue

            sha = _sha256(aug_seq)
            if sha in seen_sha:
                continue
            seen_sha.add(sha)

            aug_acc = f"{acc}_aug{copy_i}_{strat_tag}"
            aug_fpath = fasta_dir / f"{aug_acc}.fasta"
            aug_fpath.write_text(f">{aug_acc} [aug:{strat_tag}] parent={acc}\n{aug_seq}\n", encoding="utf-8")

            aug_row = dict(base_meta)
            aug_row["record_id"] = aug_acc
            aug_row["insdc_accession"] = aug_acc
            aug_row["source"] = f"augmented_{strat_tag}"
            aug_row["seq_len"] = len(aug_seq)
            aug_row["sha256"] = sha
            aug_row["split"] = "train"   # Strictly training partition
            aug_row["label_confidence"] = base_meta.get("label_confidence", "high")

            new_rows.append(aug_row)
            new_train_accs.append(aug_acc)

        if idx % 1000 == 0 or idx == len(train_accs):
            logger.info("  Processed %d / %d training sequences (%d new augmented records generated)", idx, len(train_accs), len(new_rows))

    logger.info("Augmentation complete: generated %d new biological training sequences", len(new_rows))

    # Update manifest
    all_manifest_rows = real_manifest_rows + new_rows
    fieldnames = list(real_manifest_rows[0].keys())

    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_manifest_rows)
    logger.info("Updated canonical manifest: %s (%d total rows)", manifest_path, len(all_manifest_rows))

    if split_manifest_path.exists():
        with split_manifest_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_manifest_rows)
        logger.info("Updated split manifest: %s", split_manifest_path)

    # Update train_accessions.txt
    all_train = train_accs + new_train_accs
    train_acc_file.write_text("\n".join(all_train) + "\n", encoding="utf-8")
    logger.info(
        "Updated training accessions: %s (%d accessions = %d real + %d augmented)",
        train_acc_file, len(all_train), len(train_accs), len(new_train_accs)
    )

    # Zero-leakage verification assertion
    if val_acc_file.exists() and test_acc_file.exists():
        val_accs = [l.strip() for l in val_acc_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        test_accs = [l.strip() for l in test_acc_file.read_text(encoding="utf-8").splitlines() if l.strip()]

        aug_in_val = [a for a in val_accs if "_aug" in a]
        aug_in_test = [a for a in test_accs if "_aug" in a]
        if aug_in_val or aug_in_test:
            logger.error("DATA POLLUTION DETECTED! %d aug in val, %d aug in test", len(aug_in_val), len(aug_in_test))
            return 1
        logger.info("Data Hygiene Verification PASSED: Val (%d) and Test (%d) contain 0%% augmented sequences.", len(val_accs), len(test_accs))

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Leakage-free biological data augmentation")
    p.add_argument("--config", type=Path, default=Path("configs/pipeline_config.yaml"))
    p.add_argument("--copies", type=int, default=2, help="Base augmented copies per training sequence (default: 2)")
    p.add_argument("--adaptive", action="store_true", default=True, help="Use adaptive subtype and pathotype balancing")
    p.add_argument("--no-adaptive", action="store_false", dest="adaptive", help="Disable adaptive balancing")
    p.add_argument("--clean", action="store_true", default=True, help="Clean previous augmented FASTAs before running")
    p.add_argument("--no-clean", action="store_false", dest="clean", help="Do not clean previous augmentations")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = p.parse_args(argv)

    with args.config.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    copies = args.copies or cfg.get("augmentation", {}).get("copies_per_sequence", 2)
    seed = args.seed or cfg.get("split", {}).get("seed", 42)

    return run_augmentation(
        cfg,
        copies_per_seq=copies,
        adaptive=args.adaptive,
        clean=args.clean,
        seed=seed,
    )


if __name__ == "__main__":
    sys.exit(main())
