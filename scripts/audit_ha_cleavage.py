#!/usr/bin/env python3
"""
scripts/audit_ha_cleavage.py
============================
Examines HA (Segment 4) sequences in raw and balanced datasets.
Translates 6 frames (or finds the HA ORF) and searches for canonical
proteolytic cleavage site motifs:
  - Monobasic (LPAI): [P][QEKNR][R][E][TR][RG]... with single basic residue before Gly
  - Polybasic (HPAI): R-X-K/R-R*G or contains R-R-R-K, R-K-K-R, R-E-R-R-R-K-R, etc.
"""

import os
import re
import json
from collections import Counter, defaultdict
from pathlib import Path
from Bio.Seq import Seq

PROJECT_ROOT = Path(r"F:\vishnu\f\Projects\GEnomic-framework")
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
BALANCED_DIR = PROJECT_ROOT / "dataset_augmented" / "balanced"

# Amino acid motifs for HA cleavage site
# Cleavage site is followed by GLF (Gly-Leu-Phe) at the start of HA2: G-L-F-G-A-I-A-G-F-I
# Monobasic motif: [P][QEKTR][REK][TR][RG] * GLF
# Polybasic motif: [R][X][KR][R] * GLF or >= 2 basic residues (R/K) immediately preceding GLF

GLF_REGEX = re.compile(r"([A-Z]{4,15})(GLFGAIAGF)", re.I)

def analyze_ha_sequence(dna_seq: str):
    """
    Translates sequence in all 3 forward frames, finds the HA1/HA2 cleavage junction (GLFGAIAGF).
    """
    for frame in range(3):
        sub_seq = dna_seq[frame:]
        trim_len = (len(sub_seq) // 3) * 3
        if trim_len < 60:
            continue
        try:
            protein = str(Seq(sub_seq[:trim_len]).translate())
        except Exception:
            continue
            
        m = GLF_REGEX.search(protein)
        if m:
            cleavage_upstream = m.group(1) # up to 15 AA before GLF
            fusion_peptide = m.group(2)
            
            # Check basic residues (R = Arginine, K = Lysine)
            # Count basic residues in the last 6 amino acids before GLF
            tail = cleavage_upstream[-6:]
            basic_count = tail.count("R") + tail.count("K")
            
            # Polybasic criterion: contains R-X-K/R-R or >= 3 basic residues in the insertion loop
            # Or ending in RRRKR, RERRRKR, etc.
            is_polybasic = False
            if basic_count >= 3 or re.search(r"R.[KR]R", tail) or re.search(r"[KR]{2,}R", tail):
                is_polybasic = True
                
            return {
                "detected": True,
                "motif": cleavage_upstream[-10:],
                "junction": f"{cleavage_upstream[-8:]}*{fusion_peptide[:4]}",
                "basic_count": basic_count,
                "is_polybasic": is_polybasic,
                "pathogenicity_call": "HPAI" if is_polybasic else "LPAI"
            }
            
    return {"detected": False}

def run_cleavage_audit():
    print("=" * 70)
    print("RUNNING HA CLEAVAGE MOTIF AUDIT (HPAI vs LPAI)")
    print("=" * 70)
    
    # 1. Audit HA in dataset/raw
    raw_ha_files = []
    for f in RAW_DIR.rglob("*.fasta"):
        raw_ha_files.append(f)
        
    raw_calls = Counter()
    raw_subtype_calls = defaultdict(lambda: Counter())
    
    total_raw_ha = 0
    detected_raw = 0
    
    for f in raw_ha_files:
        st = f.parent.name
        cur_hdr = None
        chunks = []
        with open(f, "r", encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                line = line.strip()
                if line.startswith(">"):
                    if cur_hdr:
                        if ("segment 4" in cur_hdr.lower() or "hemagglutinin" in cur_hdr.lower() or "ha gene" in cur_hdr.lower() or "ha (" in cur_hdr.lower() or "ha)" in cur_hdr.lower()):
                            total_raw_ha += 1
                            res = analyze_ha_sequence("".join(chunks))
                            if res["detected"]:
                                detected_raw += 1
                                raw_calls[res["pathogenicity_call"]] += 1
                                raw_subtype_calls[st][res["pathogenicity_call"]] += 1
                            else:
                                raw_calls["UNDETECTED_OR_TRUNCATED"] += 1
                                raw_subtype_calls[st]["UNDETECTED_OR_TRUNCATED"] += 1
                        chunks = []
                    cur_hdr = line
                else:
                    chunks.append(line)
            if cur_hdr and ("segment 4" in cur_hdr.lower() or "hemagglutinin" in cur_hdr.lower() or "ha gene" in cur_hdr.lower()):
                total_raw_ha += 1
                res = analyze_ha_sequence("".join(chunks))
                if res["detected"]:
                    detected_raw += 1
                    raw_calls[res["pathogenicity_call"]] += 1
                    raw_subtype_calls[st][res["pathogenicity_call"]] += 1
                else:
                    raw_calls["UNDETECTED_OR_TRUNCATED"] += 1
                    raw_subtype_calls[st]["UNDETECTED_OR_TRUNCATED"] += 1

    print(f"Total Raw HA Sequences Examined: {total_raw_ha}")
    print(f"Junctions Successfully Detected: {detected_raw}")
    print(f"Overall Calls: {dict(raw_calls)}")
    print("\nBreakdown by Subtype in Raw Dataset:")
    for st, c in sorted(raw_subtype_calls.items()):
        print(f"  {st:8s}: HPAI={c['HPAI']}, LPAI={c['LPAI']}, Undetected/Partial={c['UNDETECTED_OR_TRUNCATED']}")

    # Output to JSON
    summary = {
        "total_raw_ha_examined": total_raw_ha,
        "junctions_detected": detected_raw,
        "overall_calls": dict(raw_calls),
        "subtype_calls": {st: dict(c) for st, c in raw_subtype_calls.items()}
    }
    with open(PROJECT_ROOT / "scripts" / "ha_cleavage_audit_summary.json", "w", encoding="utf-8") as out_f:
        json.dump(summary, out_f, indent=2)
    print("\nSaved cleavage audit to scripts/ha_cleavage_audit_summary.json")

if __name__ == "__main__":
    run_cleavage_audit()
