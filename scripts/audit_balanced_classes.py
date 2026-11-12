#!/usr/bin/env python3
"""
scripts/audit_balanced_classes.py
=================================
Audits dataset_augmented/balanced/ across all 4 classes:
  - Benign
  - Likely_Benign
  - Likely_Pathogenic
  - Pathogenic

Measures exact counts, unique sequences, exact duplicates, bases, lengths,
host vs viral records, and subtype distributions.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(r"F:\vishnu\f\Projects\GEnomic-framework")
BALANCED_DIR = PROJECT_ROOT / "dataset_augmented" / "balanced"

CLASSES = ["Benign", "Likely_Benign", "Likely_Pathogenic", "Pathogenic"]

def audit_balanced():
    class_results = {}
    total_balanced_seqs = 0
    total_balanced_bases = 0
    all_seq_hashes = set()
    all_accessions = set()
    
    for c_idx, c_name in enumerate(CLASSES):
        c_dir = BALANCED_DIR / c_name
        files = sorted(list(c_dir.glob("batch_*.fasta")))
        
        c_seqs = 0
        c_bases = 0
        c_hashes = set()
        c_dups = 0
        c_accessions = Counter()
        c_subtypes = Counter()
        c_hosts = 0
        c_viral = 0
        lengths = []
        iupac = Counter()
        
        for f in files:
            cur_hdr = None
            chunks = []
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        if cur_hdr:
                            seq = "".join(chunks)
                            c_seqs += 1
                            total_balanced_seqs += 1
                            slen = len(seq)
                            c_bases += slen
                            total_balanced_bases += slen
                            lengths.append(slen)
                            
                            # check if host
                            if "cds_XP_" in cur_hdr or "Anas platyrhynchos" in cur_hdr or "lcl|NC_" in cur_hdr:
                                c_hosts += 1
                            else:
                                c_viral += 1
                                
                            # accession
                            m_acc = re.match(r">([A-Za-z0-9_.]+)", cur_hdr)
                            acc = m_acc.group(1) if m_acc else "UNKNOWN"
                            c_accessions[acc] += 1
                            all_accessions.add(acc)
                            
                            # hash
                            h = hashlib.sha256(seq.upper().encode()).hexdigest()
                            if h in c_hashes:
                                c_dups += 1
                            else:
                                c_hashes.add(h)
                            all_seq_hashes.add(h)
                            
                            # subtype
                            m_sub = re.search(r"\(?(H\d+N\d+)\)?", cur_hdr, re.I)
                            if m_sub:
                                c_subtypes[m_sub.group(1).upper()] += 1
                            elif "cds_XP_" in cur_hdr:
                                c_subtypes["HOST_DUCK"] += 1
                            else:
                                c_subtypes["UNKNOWN"] += 1
                                
                            seq_up = seq.upper()
                            for ch in "ACGTN":
                                cnt = seq_up.count(ch)
                                if cnt:
                                    iupac[ch] += cnt
                            oth = slen - sum(seq_up.count(ch) for ch in "ACGTN")
                            if oth:
                                iupac["OTHER"] += oth
                            chunks = []
                        cur_hdr = line
                    else:
                        chunks.append(line)
                if cur_hdr:
                    seq = "".join(chunks)
                    c_seqs += 1
                    total_balanced_seqs += 1
                    slen = len(seq)
                    c_bases += slen
                    total_balanced_bases += slen
                    lengths.append(slen)
                    
                    if "cds_XP_" in cur_hdr or "Anas platyrhynchos" in cur_hdr or "lcl|NC_" in cur_hdr:
                        c_hosts += 1
                    else:
                        c_viral += 1
                        
                    m_acc = re.match(r">([A-Za-z0-9_.]+)", cur_hdr)
                    acc = m_acc.group(1) if m_acc else "UNKNOWN"
                    c_accessions[acc] += 1
                    all_accessions.add(acc)
                    
                    h = hashlib.sha256(seq.upper().encode()).hexdigest()
                    if h in c_hashes:
                        c_dups += 1
                    else:
                        c_hashes.add(h)
                    all_seq_hashes.add(h)
                    
                    m_sub = re.search(r"\(?(H\d+N\d+)\)?", cur_hdr, re.I)
                    if m_sub:
                        c_subtypes[m_sub.group(1).upper()] += 1
                    elif "cds_XP_" in cur_hdr:
                        c_subtypes["HOST_DUCK"] += 1
                    else:
                        c_subtypes["UNKNOWN"] += 1
                        
                    seq_up = seq.upper()
                    for ch in "ACGTN":
                        cnt = seq_up.count(ch)
                        if cnt:
                            iupac[ch] += cnt
                    oth = slen - sum(seq_up.count(ch) for ch in "ACGTN")
                    if oth:
                        iupac["OTHER"] += oth

        lengths.sort()
        n = len(lengths)
        class_results[c_name] = {
            "class_index": c_idx,
            "num_files": len(files),
            "num_sequences": c_seqs,
            "unique_sequences": len(c_hashes),
            "exact_duplicates": c_dups,
            "unique_accessions": len(c_accessions),
            "total_bases": c_bases,
            "mean_length": round(c_bases / n, 2) if n > 0 else 0,
            "median_length": lengths[n // 2] if n > 0 else 0,
            "min_length": lengths[0] if n > 0 else 0,
            "max_length": lengths[-1] if n > 0 else 0,
            "p25_length": lengths[int(n * 0.25)] if n > 0 else 0,
            "p75_length": lengths[int(n * 0.75)] if n > 0 else 0,
            "host_sequences": c_hosts,
            "viral_sequences": c_viral,
            "subtypes": dict(c_subtypes),
            "iupac_composition": dict(iupac),
        }
        
    summary = {
        "total_balanced_files": sum(len(list((BALANCED_DIR / c).glob("batch_*.fasta"))) for c in CLASSES),
        "total_balanced_sequences": total_balanced_seqs,
        "unique_balanced_sequences": len(all_seq_hashes),
        "total_balanced_bases": total_balanced_bases,
        "unique_accessions": len(all_accessions),
        "classes": class_results,
    }
    
    out_file = PROJECT_ROOT / "scripts" / "balanced_classes_audit_summary.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"Audited balanced pool: {total_balanced_seqs:,} seqs, {total_balanced_bases:,} bases across {summary['total_balanced_files']} files.")
    print(f"Saved to: {out_file}")

if __name__ == "__main__":
    audit_balanced()
