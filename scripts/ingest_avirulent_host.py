"""
scripts/ingest_avirulent_host.py
================================
Ingest healthy host genome sequences into the canonical dataset and split manifest
as Class 0 (Avirulent).

This completes the 4-class taxonomy:
  Class 0: Avirulent (Healthy host genomes)
  Class 1: Low_Pathogenicity (LPAI monobasic cleavage, wild bird subtypes)
  Class 2: Moderate_High_Pathogenicity (H5N6, H5N8, H7N1, H7N7, H10N7, H10N8)
  Class 3: Highly_Pathogenic (HPAI polybasic cleavage site, H5N1, H5N9, H7N9)
"""

from __future__ import annotations

import csv
import glob
import hashlib
import logging
from pathlib import Path
import random
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("ingest_avirulent")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_FASTA = _PROJECT_ROOT / "dataset" / "canonical" / "fasta"
CANONICAL_MANIFEST = _PROJECT_ROOT / "dataset" / "canonical" / "manifest.csv"
SPLIT_MANIFEST = _PROJECT_ROOT / "dataset" / "splits" / "split_manifest.csv"
BENIGN_DIR = _PROJECT_ROOT / "dataset_augmented" / "balanced" / "Benign"


def parse_fasta(file_path: Path):
    records = []
    with file_path.open("r", encoding="utf-8") as fp:
        cur_hdr = None
        cur_seq = []
        for line in fp:
            line = line.strip()
            if line.startswith(">"):
                if cur_hdr:
                    seq_str = "".join(cur_seq).upper()
                    records.append((cur_hdr, seq_str))
                cur_hdr = line[1:]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_hdr:
            seq_str = "".join(cur_seq).upper()
            records.append((cur_hdr, seq_str))
    return records


def main() -> int:
    CANONICAL_FASTA.mkdir(parents=True, exist_ok=True)
    if not BENIGN_DIR.exists():
        logger.error("Benign directory not found: %s", BENIGN_DIR)
        return 1

    benign_files = sorted(BENIGN_DIR.glob("*.fasta"))
    logger.info("Found %d benign source FASTA files.", len(benign_files))

    all_benign_records = []
    for bf in benign_files:
        recs = parse_fasta(bf)
        all_benign_records.extend(recs)

    logger.info("Total benign records loaded: %d", len(all_benign_records))
    if not all_benign_records:
        logger.error("No benign records found!")
        return 1

    # Shuffle deterministically to distribute host genes evenly
    rng = random.Random(42)
    indexed_records = list(enumerate(all_benign_records, 1))
    rng.shuffle(indexed_records)

    n_total = len(indexed_records)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    # Remaining to test
    n_test = n_total - n_train - n_val

    logger.info("Partitioning %d Avirulent records: train=%d, val=%d, test=%d", n_total, n_train, n_val, n_test)

    rows = []
    for i, (orig_idx, (hdr, seq)) in enumerate(indexed_records):
        rec_id = f"AVIR_{orig_idx:05d}"
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"

        sha = hashlib.sha256(seq.encode("utf-8")).hexdigest()
        fasta_path = CANONICAL_FASTA / f"{rec_id}.fasta"
        with fasta_path.open("w", encoding="utf-8") as fp:
            fp.write(f">{rec_id} | {hdr}\n")
            # Write in 80-char chunks
            for pos in range(0, len(seq), 80):
                fp.write(seq[pos:pos+80] + "\n")

        row = {
            "record_id": rec_id,
            "insdc_accession": rec_id,
            "source": "healthy_host",
            "organism_taxid": 8839,
            "subtype": "healthy",
            "segment": "host_cds",
            "host": "Avian",
            "country": "Unknown",
            "collection_date": "2024",
            "seq_len": len(seq),
            "sha256": sha,
            "pathotype": "AVIRULENT",
            "label_confidence": "gold_standard",
            "split": split,
        }
        rows.append(row)

    # Check existing manifest to avoid duplicates
    existing_records = set()
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for r in reader:
                existing_records.add(r["record_id"])

    new_rows = [r for r in rows if r["record_id"] not in existing_records]
    logger.info("New Avirulent rows to append: %d (already existed: %d)", len(new_rows), len(rows) - len(new_rows))

    if new_rows:
        fieldnames = [
            "record_id", "insdc_accession", "source", "organism_taxid",
            "subtype", "segment", "host", "country", "collection_date",
            "seq_len", "sha256", "pathotype", "label_confidence", "split"
        ]
        # Append to CANONICAL_MANIFEST
        with CANONICAL_MANIFEST.open("a", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            for r in new_rows:
                writer.writerow(r)

        # Append to SPLIT_MANIFEST
        with SPLIT_MANIFEST.open("a", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            for r in new_rows:
                writer.writerow(r)

    logger.info("Successfully ingested Avirulent host sequences into canonical dataset and split manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

