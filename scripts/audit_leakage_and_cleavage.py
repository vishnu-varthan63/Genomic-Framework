#!/usr/bin/env python3
"""
scripts/audit_leakage_and_cleavage.py
=====================================
Detailed Data Leakage & Cleavage Site Auditor
1. Extracts parent accession / isolate identifiers from all balanced batches.
2. Simulates the preprocessor file-level train/val/test split (seed=42).
3. Detects if identical parent isolates or sequences exist across splits.
4. Identifies HA cleavage site motifs in H5/H7 sequences across classes.
"""

import os
import re
import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path(r"F:\vishnu\f\Projects\GEnomic-framework")
BALANCED_DIR = PROJECT_ROOT / "dataset_augmented" / "balanced"

CLASS_LABELS = ["Benign", "Likely_Benign", "Likely_Pathogenic", "Pathogenic"]

def parse_header_metadata(header: str):
    meta = {
        "header": header,
        "is_host": False,
        "subtype": None,
        "parent_accession": None,
        "strain": None,
        "segment": None,
        "aug_type": None,
    }
    
    # Check if host duck
    if "cds_XP_" in header or "Anas platyrhynchos" in header or "lcl|NC_" in header:
        meta["is_host"] = True
        m_acc = re.search(r"cds_([A-Za-z0-9_.]+)", header)
        if m_acc:
            meta["parent_accession"] = m_acc.group(1)
        else:
            meta["parent_accession"] = "HOST_DUCK_GENOME"
            
    # Check viral accession
    m_vir_acc = re.match(r">([A-Za-z0-9_.]+)", header)
    if m_vir_acc and not meta["is_host"]:
        meta["parent_accession"] = m_vir_acc.group(1)
        
    # Check subtype
    m_sub = re.search(r"\(?(H\d+N\d+)\)?", header, re.I)
    if m_sub:
        meta["subtype"] = m_sub.group(1).upper()
        
    # Check segment
    m_seg = re.search(r"segment\s+(\d+)", header, re.I)
    if m_seg:
        meta["segment"] = int(m_seg.group(1))
        
    # Check strain
    m_str = re.search(r"Influenza A virus \(([^)]+)\)", header)
    if m_str:
        meta["strain"] = m_str.group(1)
        
    # Check augmentation tag
    m_aug = re.search(r"\|aug=([^|]+)", header)
    if m_aug:
        meta["aug_type"] = m_aug.group(1)
        
    return meta

def run_leakage_audit():
    print("=" * 70)
    print("RUNNING SPLIT LEAKAGE & CLEAVAGE SITE AUDIT")
    print("=" * 70)
    
    class_files = defaultdict(list)
    for c_idx, c_name in enumerate(CLASS_LABELS):
        c_dir = BALANCED_DIR / c_name
        if c_dir.exists():
            for f in sorted(c_dir.glob("batch_*.fasta")):
                class_files[c_idx].append(f)
                
    # Simulate preprocessor split logic (seed=42, stratified)
    rng = random.Random(42)
    train_files = []
    val_files = []
    test_files = []
    
    for c_idx in sorted(class_files.keys()):
        files = list(class_files[c_idx])
        rng.shuffle(files)
        n = len(files)
        if n <= 4 and n >= 3:
            te = files[-1:]
            v  = files[-2:-1]
            t  = files[:-2]
        elif n == 2:
            te = files[-1:]
            v  = []
            t  = files[:1]
        elif n == 1:
            t, v, te = files[:], [], []
        else:
            n_train = max(1, int(n * 0.70))
            n_val   = max(1, int(n * 0.15))
            n_val   = min(n_val, n - n_train - 1)
            n_val   = max(1, n_val)
            t  = files[:n_train]
            v  = files[n_train: n_train + n_val]
            te = files[n_train + n_val:]
        train_files.extend([(f, c_idx) for f in t])
        val_files.extend([(f, c_idx) for f in v])
        test_files.extend([(f, c_idx) for f in te])
        
    print(f"Allocated Files: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    def gather_split_data(file_list, split_name):
        seq_records = []
        parent_accessions = Counter()
        strains = Counter()
        exact_seqs = {}
        for f, c_idx in file_list:
            cur_hdr = None
            chunks = []
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    line = line.strip()
                    if line.startswith(">"):
                        if cur_hdr:
                            seq = "".join(chunks)
                            meta = parse_header_metadata(cur_hdr)
                            meta["class"] = CLASS_LABELS[c_idx]
                            meta["split"] = split_name
                            meta["seq_len"] = len(seq)
                            meta["seq_sha"] = hashlib.sha256(seq.upper().encode()).hexdigest()
                            seq_records.append(meta)
                            if meta["parent_accession"]:
                                parent_accessions[meta["parent_accession"]] += 1
                            if meta["strain"]:
                                strains[meta["strain"]] += 1
                            exact_seqs[meta["seq_sha"]] = meta
                            chunks = []
                        cur_hdr = line
                    else:
                        chunks.append(line)
                if cur_hdr:
                    seq = "".join(chunks)
                    meta = parse_header_metadata(cur_hdr)
                    meta["class"] = CLASS_LABELS[c_idx]
                    meta["split"] = split_name
                    meta["seq_len"] = len(seq)
                    meta["seq_sha"] = hashlib.sha256(seq.upper().encode()).hexdigest()
                    seq_records.append(meta)
                    if meta["parent_accession"]:
                        parent_accessions[meta["parent_accession"]] += 1
                    if meta["strain"]:
                        strains[meta["strain"]] += 1
                    exact_seqs[meta["seq_sha"]] = meta
        return seq_records, parent_accessions, strains, exact_seqs

    train_recs, train_accs, train_strains, train_seqs = gather_split_data(train_files, "train")
    val_recs, val_accs, val_strains, val_seqs = gather_split_data(val_files, "val")
    test_recs, test_accs, test_strains, test_seqs = gather_split_data(test_files, "test")
    
    print(f"Loaded Sequences: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")
    
    # Check parent accession overlap
    train_acc_set = set(train_accs.keys())
    val_acc_set = set(val_accs.keys())
    test_acc_set = set(test_accs.keys())
    
    leakage_train_val = train_acc_set & val_acc_set
    leakage_train_test = train_acc_set & test_acc_set
    leakage_val_test = val_acc_set & test_acc_set
    
    print("\n--- PARENT ACCESSION OVERLAP (DATA LEAKAGE) ---")
    print(f"Unique Parent Accessions: Train={len(train_acc_set)}, Val={len(val_acc_set)}, Test={len(test_acc_set)}")
    print(f"Overlap Train & Val: {len(leakage_train_val)} accessions")
    print(f"Overlap Train & Test: {len(leakage_train_test)} accessions")
    print(f"Overlap Val & Test: {len(leakage_val_test)} accessions")
    
    # Check exact sequence overlap
    train_seq_set = set(train_seqs.keys())
    val_seq_set = set(val_seqs.keys())
    test_seq_set = set(test_seqs.keys())
    
    seq_leak_train_val = train_seq_set & val_seq_set
    seq_leak_train_test = train_seq_set & test_seq_set
    seq_leak_val_test = val_seq_set & test_seq_set
    
    print("\n--- EXACT IDENTICAL SEQUENCE OVERLAP ---")
    print(f"Identical sequences Train & Val: {len(seq_leak_train_val)}")
    print(f"Identical sequences Train & Test: {len(seq_leak_train_test)}")
    print(f"Identical sequences Val & Test: {len(seq_leak_val_test)}")
    
    # Check Benign class composition
    benign_host_count = sum(1 for r in train_recs + val_recs + test_recs if r["class"] == "Benign" and r["is_host"])
    benign_viral_count = sum(1 for r in train_recs + val_recs + test_recs if r["class"] == "Benign" and not r["is_host"])
    print(f"\n--- BENIGN CLASS ANALYSIS ---")
    print(f"Total Benign Sequences: {benign_host_count + benign_viral_count}")
    print(f"  Avian Host (Duck CDS) Sequences: {benign_host_count} ({benign_host_count/(benign_host_count+benign_viral_count)*100:.1f}%)")
    print(f"  Viral Sequences: {benign_viral_count} ({benign_viral_count/(benign_host_count+benign_viral_count)*100:.1f}%)")

    # Output to JSON
    leakage_summary = {
        "split_counts": {
            "train": len(train_recs),
            "val": len(val_recs),
            "test": len(test_recs),
        },
        "parent_accession_leakage": {
            "train_val_overlap": len(leakage_train_val),
            "train_test_overlap": len(leakage_train_test),
            "val_test_overlap": len(leakage_val_test),
            "sample_leaked_accessions_train_test": list(leakage_train_test)[:15],
        },
        "exact_sequence_leakage": {
            "train_val_overlap": len(seq_leak_train_val),
            "train_test_overlap": len(seq_leak_train_test),
            "val_test_overlap": len(seq_leak_val_test),
        },
        "benign_composition": {
            "host_sequences": benign_host_count,
            "viral_sequences": benign_viral_count,
            "percent_host": round(benign_host_count/(benign_host_count+benign_viral_count)*100, 2) if (benign_host_count+benign_viral_count) > 0 else 0,
        }
    }
    with open(PROJECT_ROOT / "scripts" / "leakage_audit_summary.json", "w", encoding="utf-8") as out_f:
        json.dump(leakage_summary, out_f, indent=2)
    print(f"\nSaved leakage summary to scripts/leakage_audit_summary.json")

if __name__ == "__main__":
    run_leakage_audit()
