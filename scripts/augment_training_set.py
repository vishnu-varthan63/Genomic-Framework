"""
scripts/augment_training_set.py
===============================
Leakage-Free Biological Augmentation of the Training Partition.

Strict Scientific Constraints:
------------------------------
1. Augments ONLY sequences assigned to the 'train' split (7,539 accessions).
2. Leaves validation (1,602) and test (1,656) sets 100% real and unpolluted.
3. Uses realistic biological mutation operators for Influenza A:
   - Transition bias (purine-purine / pyrimidine-pyrimidine drift from viral RdRp)
   - Avian host adaptation bias (chicken/duck/goose codon drift)
   - Low-rate random point mutations (genetic drift)
4. Writes augmented FASTAs to dataset/canonical/fasta/ and updates manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def _sha256(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()


def _read_fasta_seq(fpath: Path) -> Tuple[str, str]:
    """Return (header, sequence) from single FASTA file."""
    lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
    header = lines[0][1:].strip() if lines and lines[0].startswith(">") else fpath.stem
    seq = "".join(ln.strip().upper() for ln in lines if not ln.startswith(">"))
    return header, seq


def run_augmentation(cfg: Dict, copies_per_seq: int = 2, seed: int = 42) -> int:
    canonical_dir = Path(cfg["paths"]["canonical_dir"])
    splits_dir = Path(cfg["paths"]["splits_dir"])
    manifest_path = canonical_dir / "manifest.csv"
    split_manifest_path = splits_dir / "split_manifest.csv"
    train_acc_file = splits_dir / "train_accessions.txt"
    fasta_dir = canonical_dir / "fasta"

    if not train_acc_file.exists() or not manifest_path.exists():
        logger.error("Required split files missing. Run accession_split stage first.")
        return 1

    train_accs = [
        line.strip()
        for line in train_acc_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_acc_set = set(train_accs)
    logger.info("Loaded %d training accessions to augment", len(train_accs))

    # Read existing manifest
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest_rows = list(csv.DictReader(fh))

    # Index rows by accession
    row_by_acc: Dict[str, Dict] = {r["insdc_accession"]: r for r in manifest_rows}
    seen_sha = {r["sha256"] for r in manifest_rows}

    augmenter = GenomicAugmenter(config=AUGMENTATION_CONFIG, seed=seed)

    rng = random.Random(seed)
    strategies = ["transition_bias", "host_adaptation", "random_mutation", "transversion_bias"]
    hosts = ["chicken", "duck", "goose", "quail", "turkey"]

    new_rows: List[Dict] = []
    new_train_accs: List[str] = []

    logger.info("Beginning biological training set augmentation (%d copies/seq) ...", copies_per_seq)

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

        for copy_i in range(1, copies_per_seq + 1):
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
            aug_row["split"] = "train"   # Strictly training set only
            aug_row["label_confidence"] = base_meta.get("label_confidence", "high")

            new_rows.append(aug_row)
            new_train_accs.append(aug_acc)

        if idx % 1000 == 0:
            logger.info("  Augmented %d / %d training sequences (%d new records)", idx, len(train_accs), len(new_rows))

    logger.info("Augmentation complete: generated %d new training sequences", len(new_rows))

    # Append to manifest
    all_manifest_rows = manifest_rows + new_rows
    fieldnames = list(manifest_rows[0].keys())

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

    # Append to train_accessions.txt
    all_train = train_accs + new_train_accs
    train_acc_file.write_text("\n".join(all_train) + "\n", encoding="utf-8")
    logger.info(
        "Updated training accessions: %s (%d accessions = %d real + %d augmented)",
        train_acc_file, len(all_train), len(train_accs), len(new_train_accs)
    )

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train-only biological data augmentation")
    p.add_argument("--config", type=Path, default=Path("configs/pipeline_config.yaml"))
    p.add_argument("--copies", type=int, default=2, help="Augmented copies per training sequence")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = p.parse_args(argv)

    with args.config.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    copies = args.copies or cfg.get("augmentation", {}).get("copies_per_sequence", 2)
    seed = args.seed or cfg.get("split", {}).get("seed", 42)

    return run_augmentation(cfg, copies_per_seq=copies, seed=seed)


if __name__ == "__main__":
    sys.exit(main())
