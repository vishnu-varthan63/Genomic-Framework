#!/usr/bin/env python3
"""
scripts/audit_raw_viral.py
==========================
Audits dataset/raw/ specifically (all 134 viral FASTA files across 14 subtypes).
Calculates exact counts, lengths, segments, subtypes, completeness, and sequence uniqueness.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(r"F:\vishnu\f\Projects\GEnomic-framework")
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"

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
        return m.group(1)
    return "UNKNOWN"

def run_raw_audit():
    fasta_files = sorted([f for f in RAW_DIR.rglob("*") if f.suffix.lower() in (".fasta", ".fa", ".fna")])
    
    total_seqs = 0
    total_bases = 0
    lengths = []
    subtypes = defaultdict(lambda: {"files": 0, "seqs": 0, "bases": 0, "complete": 0, "partial": 0})
    segments = defaultdict(lambda: {"seqs": 0, "bases": 0, "complete": 0, "partial": 0})
    accessions = set()
    seq_hashes = set()
    duplicate_seqs = 0
    iupac_counts = Counter()
    
    for fp in fasta_files:
        subtype_folder = fp.parent.name
        subtypes[subtype_folder]["files"] += 1
        
        cur_hdr = None
        chunks = []
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if cur_hdr:
                        seq = "".join(chunks)
                        total_seqs += 1
                        seq_len = len(seq)
                        total_bases += seq_len
                        lengths.append(seq_len)
                        
                        acc = extract_accession(cur_hdr)
                        accessions.add(acc)
                        
                        h = hashlib.sha256(seq.upper().encode()).hexdigest()
                        if h in seq_hashes:
                            duplicate_seqs += 1
                        else:
                            seq_hashes.add(h)
                            
                        seg = identify_segment(cur_hdr)
                        
                        # Completeness
                        exp = EXPECTED_SEGMENT_LENGTHS.get(seg, 1500)
                        is_comp = (seq_len >= exp * 0.90)
                        
                        subtypes[subtype_folder]["seqs"] += 1
                        subtypes[subtype_folder]["bases"] += seq_len
                        if is_comp:
                            subtypes[subtype_folder]["complete"] += 1
                        else:
                            subtypes[subtype_folder]["partial"] += 1
                            
                        segments[seg]["seqs"] += 1
                        segments[seg]["bases"] += seq_len
                        if is_comp:
                            segments[seg]["complete"] += 1
                        else:
                            segments[seg]["partial"] += 1
                            
                        seq_up = seq.upper()
                        for ch in "ACGTN":
                            c = seq_up.count(ch)
                            if c:
                                iupac_counts[ch] += c
                        oth = seq_len - sum(seq_up.count(ch) for ch in "ACGTN")
                        if oth:
                            iupac_counts["OTHER"] += oth
                        chunks = []
                    cur_hdr = line
                else:
                    chunks.append(line)
            if cur_hdr:
                seq = "".join(chunks)
                total_seqs += 1
                seq_len = len(seq)
                total_bases += seq_len
                lengths.append(seq_len)
                
                acc = extract_accession(cur_hdr)
                accessions.add(acc)
                
                h = hashlib.sha256(seq.upper().encode()).hexdigest()
                if h in seq_hashes:
                    duplicate_seqs += 1
                else:
                    seq_hashes.add(h)
                    
                seg = identify_segment(cur_hdr)
                exp = EXPECTED_SEGMENT_LENGTHS.get(seg, 1500)
                is_comp = (seq_len >= exp * 0.90)
                
                subtypes[subtype_folder]["seqs"] += 1
                subtypes[subtype_folder]["bases"] += seq_len
                if is_comp:
                    subtypes[subtype_folder]["complete"] += 1
                else:
                    subtypes[subtype_folder]["partial"] += 1
                    
                segments[seg]["seqs"] += 1
                segments[seg]["bases"] += seq_len
                if is_comp:
                    segments[seg]["complete"] += 1
                else:
                    segments[seg]["partial"] += 1
                    
                seq_up = seq.upper()
                for ch in "ACGTN":
                    c = seq_up.count(ch)
                    if c:
                        iupac_counts[ch] += c
                oth = seq_len - sum(seq_up.count(ch) for ch in "ACGTN")
                if oth:
                    iupac_counts["OTHER"] += oth

    lengths.sort()
    n = len(lengths)
    
    summary = {
        "total_files": len(fasta_files),
        "total_sequences": total_seqs,
        "unique_sequences": len(seq_hashes),
        "exact_duplicates": duplicate_seqs,
        "unique_accessions": len(accessions),
        "total_bases": total_bases,
        "mean_length": round(total_bases / n, 2) if n > 0 else 0,
        "median_length": lengths[n // 2] if n > 0 else 0,
        "min_length": lengths[0] if n > 0 else 0,
        "max_length": lengths[-1] if n > 0 else 0,
        "p25_length": lengths[int(n * 0.25)] if n > 0 else 0,
        "p75_length": lengths[int(n * 0.75)] if n > 0 else 0,
        "iupac_composition": dict(iupac_counts),
        "subtypes": dict(subtypes),
        "segments": dict(segments),
    }
    
    out_file = PROJECT_ROOT / "scripts" / "raw_viral_audit_summary.json"
    with open(out_file, "w", encoding="utf-8") as out_f:
        json.dump(summary, out_f, indent=2)
    print(f"Audited {len(fasta_files)} raw viral files: {total_seqs:,} seqs, {total_bases:,} bases.")
    print(f"Saved summary to: {out_file}")

if __name__ == "__main__":
    run_raw_audit()
