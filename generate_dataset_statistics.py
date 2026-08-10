"""
generate_dataset_statistics.py
================================
Dynamic Dataset Auditor & JSON Generator for Genomic Framework

Scans all dataset directories:
  1. dataset/ (Raw surveillance & healthy FASTA files)
  2. dataset_augmented/raw/ & dataset_augmented/healthy/ (Full augmented set)
  3. dataset_augmented/balanced/ (4-class balanced training/testing set)

Parses every single FASTA record (starting with '>'), computes nucleotide counts,
average/median/min/max sequence lengths, class breakdowns, and train/val/test splits.

Saves output to:
  - dataset_statistics.json (Central source of truth for the website & framework)
  - website/content/dataset_statistics.json
"""

import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_JSON = PROJECT_ROOT / "dataset_statistics.json"
WEBSITE_JSON = PROJECT_ROOT / "website" / "content" / "dataset_statistics.json"

FASTA_EXTENSIONS = (".fasta", ".fa", ".fna")

CLASS_MAPPING = {
    "Benign": 0,
    "Likely_Benign": 1,
    "Likely_Pathogenic": 2,
    "Pathogenic": 3,
}

CLASS_LABELS = ["Benign", "Likely_Benign", "Likely_Pathogenic", "Pathogenic"]


def parse_fasta(file_path: Path):
    """Parse a FASTA file and return a list of sequence strings."""
    sequences = []
    current_seq = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append("".join(current_seq))
    return sequences


def scan_dataset():
    print("=" * 70)
    print("GENOMIC FRAMEWORK — DATASET AUDIT & STATISTICS GENERATOR")
    print("Project Root:", PROJECT_ROOT)
    print("=" * 70)

    # 1. Audit dataset/ (Raw Original)
    raw_dir = PROJECT_ROOT / "dataset"
    raw_files = list(raw_dir.rglob("*")) if raw_dir.exists() else []
    raw_fasta_files = [f for f in raw_files if f.suffix.lower() in FASTA_EXTENSIONS]

    raw_seq_count = 0
    raw_total_bases = 0
    raw_lengths = []

    for fp in raw_fasta_files:
        seqs = parse_fasta(fp)
        raw_seq_count += len(seqs)
        for s in seqs:
            l = len(s)
            raw_total_bases += l
            raw_lengths.append(l)

    # 2. Audit dataset_augmented/ (Full Augmented)
    aug_dir = PROJECT_ROOT / "dataset_augmented"
    aug_files = list(aug_dir.rglob("*")) if aug_dir.exists() else []
    aug_fasta_files = [f for f in aug_files if f.suffix.lower() in FASTA_EXTENSIONS]

    aug_seq_count = 0
    aug_total_bases = 0
    aug_lengths = []

    for fp in aug_fasta_files:
        seqs = parse_fasta(fp)
        aug_seq_count += len(seqs)
        for s in seqs:
            l = len(s)
            aug_total_bases += l
            aug_lengths.append(l)

    # 3. Audit dataset_augmented/balanced/ (Balanced 4-Class Partition)
    bal_dir = aug_dir / "balanced"
    class_stats = {}
    total_bal_files = 0
    total_bal_seqs = 0
    total_bal_bases = 0
    bal_lengths = []

    for cls_name in CLASS_LABELS:
        cls_dir = bal_dir / cls_name
        if cls_dir.exists():
            c_files = [f for f in cls_dir.glob("*") if f.suffix.lower() in FASTA_EXTENSIONS]
            c_seq_count = 0
            c_bases = 0
            c_lengths = []
            for fp in c_files:
                seqs = parse_fasta(fp)
                c_seq_count += len(seqs)
                for s in seqs:
                    l = len(s)
                    c_bases += l
                    c_lengths.append(l)

            total_bal_files += len(c_files)
            total_bal_seqs += c_seq_count
            total_bal_bases += c_bases
            bal_lengths.extend(c_lengths)

            class_stats[cls_name] = {
                "class_index": CLASS_MAPPING[cls_name],
                "num_files": len(c_files),
                "num_sequences": c_seq_count,
                "total_bases": c_bases,
                "avg_sequence_length": round(statistics.mean(c_lengths), 1) if c_lengths else 0,
                "min_sequence_length": min(c_lengths) if c_lengths else 0,
                "max_sequence_length": max(c_lengths) if c_lengths else 0,
            }
        else:
            class_stats[cls_name] = {
                "class_index": CLASS_MAPPING[cls_name],
                "num_files": 0,
                "num_sequences": 0,
                "total_bases": 0,
                "avg_sequence_length": 0,
                "min_sequence_length": 0,
                "max_sequence_length": 0,
            }

    # Add percentage of total sequences to class_stats
    for cls_name in CLASS_LABELS:
        cnt = class_stats[cls_name]["num_sequences"]
        pct = round((cnt / total_bal_seqs) * 100, 2) if total_bal_seqs > 0 else 0
        class_stats[cls_name]["percentage_of_total_sequences"] = pct

    # Stratified Split Breakdown for dataset_augmented/balanced
    # 70% Train, 15% Val, 15% Test with small-class reserve-test-first logic
    train_files, val_files, test_files = 0, 0, 0
    train_seqs, val_seqs, test_seqs = 0, 0, 0

    split_class_breakdown = {"train": {}, "val": {}, "test": {}}

    import random
    rng = random.Random(42)

    for cls_name in CLASS_LABELS:
        cls_dir = bal_dir / cls_name
        if not cls_dir.exists():
            continue
        c_files = sorted([f for f in cls_dir.glob("*") if f.suffix.lower() in FASTA_EXTENSIONS])
        shuffled = list(c_files)
        rng.shuffle(shuffled)
        n = len(shuffled)

        if n <= 4 and n >= 3:
            te_files = shuffled[-1:]
            v_files  = shuffled[-2:-1]
            t_files  = shuffled[:-2]
        elif n == 2:
            te_files = shuffled[-1:]
            v_files  = []
            t_files  = shuffled[:1]
        elif n == 1:
            t_files, v_files, te_files = shuffled[:], [], []
        else:
            n_train = max(1, int(n * 0.70))
            n_val   = max(1, int(n * 0.15))
            n_val   = min(n_val, n - n_train - 1)
            n_val   = max(1, n_val)
            t_files  = shuffled[:n_train]
            v_files  = shuffled[n_train: n_train + n_val]
            te_files = shuffled[n_train + n_val:]

        t_s = sum(len(parse_fasta(f)) for f in t_files)
        v_s = sum(len(parse_fasta(f)) for f in v_files)
        te_s = sum(len(parse_fasta(f)) for f in te_files)

        train_files += len(t_files)
        val_files += len(v_files)
        test_files += len(te_files)

        train_seqs += t_s
        val_seqs += v_s
        test_seqs += te_s

        split_class_breakdown["train"][cls_name] = {"files": len(t_files), "sequences": t_s}
        split_class_breakdown["val"][cls_name] = {"files": len(v_files), "sequences": v_s}
        split_class_breakdown["test"][cls_name] = {"files": len(te_files), "sequences": te_s}

    dataset_stats_dict = {
        "dataset_name": "AvianFlu Genomic Classification Dataset",
        "version": "1.0.0-Augmented",
        "license": "MIT",
        "description": "Bio-curated FASTA dataset of avian influenza viral genome segments categorized into 4 pathogenicity classes.",
        "summary": {
            "total_workspace_fasta_files": len(aug_fasta_files) + len(raw_fasta_files),
            "raw_dataset_files": len(raw_fasta_files),
            "raw_dataset_sequences": raw_seq_count,
            "raw_dataset_bases": raw_total_bases,
            "raw_avg_length": round(statistics.mean(raw_lengths), 1) if raw_lengths else 0,
            "raw_median_length": round(statistics.median(raw_lengths), 1) if raw_lengths else 0,
            "augmented_dataset_files": len(aug_fasta_files),
            "augmented_dataset_sequences": aug_seq_count,
            "augmented_dataset_bases": aug_total_bases,
            "augmentation_factor_files": round(len(aug_fasta_files) / len(raw_fasta_files), 2) if raw_fasta_files else 0,
            "augmentation_factor_seqs": round(aug_seq_count / raw_seq_count, 2) if raw_seq_count else 0,
            "balanced_dataset_files": total_bal_files,
            "balanced_dataset_sequences": total_bal_seqs,
            "balanced_dataset_bases": total_bal_bases,
            "balanced_avg_length": round(statistics.mean(bal_lengths), 1) if bal_lengths else 0,
            "balanced_median_length": round(statistics.median(bal_lengths), 1) if bal_lengths else 0,
            "balanced_min_length": min(bal_lengths) if bal_lengths else 0,
            "balanced_max_length": max(bal_lengths) if bal_lengths else 0,
        },
        "split_statistics": {
            "train": {
                "files": train_files,
                "sequences": train_seqs,
                "percentage_files": round((train_files / total_bal_files) * 100, 1) if total_bal_files else 0,
                "percentage_seqs": round((train_seqs / total_bal_seqs) * 100, 1) if total_bal_seqs else 0,
                "class_breakdown": split_class_breakdown["train"],
            },
            "val": {
                "files": val_files,
                "sequences": val_seqs,
                "percentage_files": round((val_files / total_bal_files) * 100, 1) if total_bal_files else 0,
                "percentage_seqs": round((val_seqs / total_bal_seqs) * 100, 1) if total_bal_seqs else 0,
                "class_breakdown": split_class_breakdown["val"],
            },
            "test": {
                "files": test_files,
                "sequences": test_seqs,
                "percentage_files": round((test_files / total_bal_files) * 100, 1) if total_bal_files else 0,
                "percentage_seqs": round((test_seqs / total_bal_seqs) * 100, 1) if total_bal_seqs else 0,
                "class_breakdown": split_class_breakdown["test"],
            },
        },
        "class_statistics": class_stats,
    }

    # Print summary report
    print("\n" + "=" * 70)
    print("DATASET AUDIT SUMMARY RESULTS")
    print("=" * 70)
    print(f"Total Workspace FASTA Files  : {dataset_stats_dict['summary']['total_workspace_fasta_files']}")
    print(f"Raw Dataset FASTA Files      : {len(raw_fasta_files)}")
    print(f"Raw Dataset Sequence Records : {raw_seq_count:,}")
    print(f"Raw Total Nucleotides        : {raw_total_bases:,} bp ({raw_total_bases / 1e6:.2f} MB)")
    print("-" * 70)
    print(f"Full Augmented FASTA Files   : {len(aug_fasta_files)}")
    print(f"Full Augmented Sequences     : {aug_seq_count:,}")
    print(f"Full Augmented Nucleotides   : {aug_total_bases:,} bp ({aug_total_bases / 1e6:.2f} MB)")
    print("-" * 70)
    print(f"Balanced Dataset FASTA Files : {total_bal_files}")
    print(f"Balanced Dataset Sequences   : {total_bal_seqs:,}")
    print(f"Balanced Total Nucleotides   : {total_bal_bases:,} bp ({total_bal_bases / 1e6:.2f} MB)")
    print(f"Balanced Seq Length (Min/Avg/Med/Max): {min(bal_lengths)} / {statistics.mean(bal_lengths):.1f} / {statistics.median(bal_lengths):.1f} / {max(bal_lengths)} bp")
    print("-" * 70)
    print(f"Stratified Train / Val / Test Split (Files) : {train_files} / {val_files} / {test_files}")
    print(f"Stratified Train / Val / Test Split (Seqs)  : {train_seqs:,} / {val_seqs:,} / {test_seqs:,}")
    print("=" * 70)

    # Save to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset_stats_dict, f, indent=2)
    print(f"\nSaved dataset statistics JSON to {OUTPUT_JSON}")

    if WEBSITE_JSON.parent.exists():
        with open(WEBSITE_JSON, "w", encoding="utf-8") as f:
            json.dump(dataset_stats_dict, f, indent=2)
        print(f"Saved website dataset statistics JSON to {WEBSITE_JSON}")

    return dataset_stats_dict


if __name__ == "__main__":
    scan_dataset()
