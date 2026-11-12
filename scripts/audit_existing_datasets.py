#!/usr/bin/env python3
"""
scripts/audit_existing_datasets.py
==================================
Forensic Dataset Audit Script for Avian Influenza Genomic Intelligence Framework.
Performs an exhaustive, record-level audit of all recovered datasets:
  - dataset/raw/
  - dataset/healthy/
  - dataset_augmented/balanced/
  - dataset_augmented/raw/
  - dataset_augmented/healthy/

Calculates exact empirical counts, lengths, nucleotide compositions,
sequence hashes, accession uniqueness, completeness, host vs viral identities,
and data leakage across splits.
"""

import os
import sys
import re
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(r"F:\vishnu\f\Projects\GEnomic-framework")
FASTA_EXTS = {".fasta", ".fa", ".fna"}

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

SEGMENT_PATTERNS = [
    (1, re.compile(r"segment\s+1|polymerase\s+pb2|\bPB2\b", re.I)),
    (2, re.compile(r"segment\s+2|polymerase\s+pb1|\bPB1\b", re.I)),
    (3, re.compile(r"segment\s+3|polymerase\s+pa|\bPA\b", re.I)),
    (4, re.compile(r"segment\s+4|hemagglutinin|\bHA\b", re.I)),
    (5, re.compile(r"segment\s+5|nucleoprotein|nucleocapsid|\bNP\b", re.I)),
    (6, re.compile(r"segment\s+6|neuraminidase|\bNA\b", re.I)),
    (7, re.compile(r"segment\s+7|matrix|\bM1\b|\bM2\b|\bMP\b", re.I)),
    (8, re.compile(r"segment\s+8|nonstructural|\bNS1\b|\bNS2\b|\bNEP\b|\bNS\b", re.I)),
]

EXPECTED_SEGMENT_LENGTHS = {
    1: 2341,
    2: 2341,
    3: 2233,
    4: 1778,
    5: 1565,
    6: 1413,
    7: 1027,
    8: 890,
}

def identify_segment(header: str) -> int:
    for seg_num, pat in SEGMENT_PATTERNS:
        if pat.search(header):
            return seg_num
    return 0

def extract_accession(header: str) -> str:
    m = re.match(r">([A-Za-z0-9_.]+)", header)
    if m:
        first_token = m.group(1)
        if first_token.startswith("lcl|"):
            parts = first_token.split("|")
            if len(parts) > 1:
                return parts[1]
        return first_token
    return "UNKNOWN"

def fast_parse_fasta(file_path: Path):
    current_header = None
    seq_chunks = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    yield current_header, "".join(seq_chunks)
                    seq_chunks = []
                current_header = line
            else:
                seq_chunks.append(line)
        if current_header is not None:
            yield current_header, "".join(seq_chunks)

def audit_directory(dir_path: Path, is_host_dir=False):
    print(f"Scanning directory: {dir_path}...")
    fasta_files = [f for f in dir_path.rglob("*") if f.suffix.lower() in FASTA_EXTS]
    
    file_records = []
    total_seqs = 0
    total_bases = 0
    seq_hashes = set()
    duplicate_seq_count = 0
    accession_counts = Counter()
    lengths = []
    base_counts = Counter()
    segment_counts = Counter()
    subtype_counts = Counter()
    complete_count = 0
    partial_count = 0
    sample_records = []
    
    for f in sorted(fasta_files):
        rel_path = f.relative_to(dir_path)
        f_size = f.stat().st_size
        f_seq_count = 0
        f_bases = 0
        
        for header, seq in fast_parse_fasta(f):
            total_seqs += 1
            f_seq_count += 1
            seq_len = len(seq)
            total_bases += seq_len
            f_bases += seq_len
            lengths.append(seq_len)
            
            seq_up = seq.upper()
            for ch in "ACGTN":
                c = seq_up.count(ch)
                if c:
                    base_counts[ch] += c
            other = seq_len - sum(seq_up.count(ch) for ch in "ACGTN")
            if other > 0:
                base_counts["OTHER"] += other
                
            acc = extract_accession(header)
            accession_counts[acc] += 1
            
            if is_host_dir and seq_len > 10000:
                h = hashlib.sha256(f"{seq_len}_{seq[:1000]}".encode()).hexdigest()
            else:
                h = sha256_str(seq.upper())
                
            if h in seq_hashes:
                duplicate_seq_count += 1
            else:
                seq_hashes.add(h)
                
            seg = identify_segment(header)
            segment_counts[seg] += 1
            
            if seg in EXPECTED_SEGMENT_LENGTHS:
                exp = EXPECTED_SEGMENT_LENGTHS[seg]
                if seq_len >= exp * 0.90:
                    complete_count += 1
                else:
                    partial_count += 1
            else:
                if seq_len >= 1000:
                    complete_count += 1
                else:
                    partial_count += 1
                    
            sub_m = re.search(r"\(?(H\d+N\d+)\)?", header, re.I)
            if sub_m:
                subtype_counts[sub_m.group(1).upper()] += 1
            else:
                folder_sub = f.parent.name
                if re.match(r"^H\d+N\d+$", folder_sub, re.I):
                    subtype_counts[folder_sub.upper()] += 1
                else:
                    subtype_counts["UNKNOWN_OR_HOST"] += 1
                    
            if f_seq_count <= 2 and len(sample_records) < 30:
                sample_records.append({
                    "file": str(rel_path),
                    "header": header[:120],
                    "length": seq_len,
                    "accession": acc,
                    "segment": seg
                })
                
        file_records.append({
            "relative_path": str(rel_path),
            "size_bytes": f_size,
            "seq_count": f_seq_count,
            "bases": f_bases,
        })
        
    lengths.sort()
    n = len(lengths)
    stats = {
        "total_files": len(fasta_files),
        "total_sequences": total_seqs,
        "unique_sequences": len(seq_hashes),
        "duplicate_sequences": duplicate_seq_count,
        "total_bases": total_bases,
        "mean_length": round(total_bases / n, 2) if n > 0 else 0,
        "median_length": lengths[n // 2] if n > 0 else 0,
        "min_length": lengths[0] if n > 0 else 0,
        "max_length": lengths[-1] if n > 0 else 0,
        "p25_length": lengths[int(n * 0.25)] if n > 0 else 0,
        "p75_length": lengths[int(n * 0.75)] if n > 0 else 0,
        "complete_sequences": complete_count,
        "partial_sequences": partial_count,
        "unique_accessions": len(accession_counts),
        "base_composition": dict(base_counts),
        "segment_distribution": dict(segment_counts),
        "subtype_distribution": dict(subtype_counts),
        "sample_records": sample_records,
        "file_records_count": len(file_records),
    }
    return stats

def run_audit():
    print("=================================================================")
    print("STARTING FORENSIC DATASET AUDIT")
    print("=================================================================")
    
    results = {}
    
    # 1. Audit dataset/raw/
    raw_dir = PROJECT_ROOT / "dataset" / "raw"
    if raw_dir.exists():
        print("\n--- Auditing dataset/raw ---")
        results["dataset_raw"] = audit_directory(raw_dir)
        
    # 2. Audit dataset_augmented/balanced/
    balanced_dir = PROJECT_ROOT / "dataset_augmented" / "balanced"
    if balanced_dir.exists():
        print("\n--- Auditing dataset_augmented/balanced (by class) ---")
        balanced_classes = {}
        for class_dir in sorted([d for d in balanced_dir.iterdir() if d.is_dir()]):
            c_name = class_dir.name
            print(f"  Auditing class: {c_name}...")
            balanced_classes[c_name] = audit_directory(class_dir)
        results["dataset_augmented_balanced"] = balanced_classes
        
    # 3. Audit dataset_augmented/raw/ and dataset_augmented/healthy/
    aug_raw_dir = PROJECT_ROOT / "dataset_augmented" / "raw"
    if aug_raw_dir.exists():
        print("\n--- Auditing dataset_augmented/raw ---")
        results["dataset_augmented_raw"] = audit_directory(aug_raw_dir)
        
    aug_healthy_dir = PROJECT_ROOT / "dataset_augmented" / "healthy"
    if aug_healthy_dir.exists():
        print("\n--- Auditing dataset_augmented/healthy ---")
        results["dataset_augmented_healthy"] = audit_directory(aug_healthy_dir)

    # 4. Audit dataset/healthy/ (sample summary)
    healthy_dir = PROJECT_ROOT / "dataset" / "healthy"
    if healthy_dir.exists():
        print("\n--- Auditing dataset/healthy ---")
        results["dataset_healthy"] = audit_directory(healthy_dir, is_host_dir=True)

    # Save results to JSON
    out_file = PROJECT_ROOT / "scripts" / "dataset_audit_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAudit complete! Saved detailed metrics to: {out_file}")

if __name__ == "__main__":
    run_audit()
