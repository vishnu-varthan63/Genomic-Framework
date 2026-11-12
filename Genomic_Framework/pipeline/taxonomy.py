"""Shared avian influenza taxonomy: subtypes, segments, pathotypes."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

SEGMENT_LABELS: List[str] = ["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"]
SEGMENT_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(SEGMENT_LABELS)}
SEGMENT_ALIASES: Dict[str, str] = {
    "PB2": "PB2", "SEGMENT 1": "PB2", "SEG1": "PB2", "1": "PB2",
    "PB1": "PB1", "SEGMENT 2": "PB1", "SEG2": "PB1", "2": "PB1",
    "PA": "PA", "SEGMENT 3": "PA", "SEG3": "PA", "3": "PA",
    "HA": "HA", "HEMAGGLUTININ": "HA", "SEGMENT 4": "HA", "SEG4": "HA", "4": "HA",
    "NP": "NP", "NUCLEOPROTEIN": "NP", "SEGMENT 5": "NP", "SEG5": "NP", "5": "NP",
    "NA": "NA", "NEURAMINIDASE": "NA", "SEGMENT 6": "NA", "SEG6": "NA", "6": "NA",
    "M": "M", "MP": "M", "M1": "M", "M2": "M", "MATRIX": "M", "SEGMENT 7": "M", "SEG7": "M", "7": "M",
    "NS": "NS", "NS1": "NS", "NEP": "NS", "SEGMENT 8": "NS", "SEG8": "NS", "8": "NS",
}

PATHOTYPE_LABELS: List[str] = ["LPAI", "HPAI"]
PATHOTYPE_TO_INDEX: Dict[str, int] = {"LPAI": 0, "HPAI": 1}
INDETERMINATE_PATHOTYPE = "INDETERMINATE"
IGNORE_INDEX = -1

DEFAULT_SUBTYPES: List[str] = [
    "H5N1", "H5N2", "H5N3", "H5N5", "H5N6", "H5N8", "H5N9",
    "H7N1", "H7N2", "H7N3", "H7N7", "H7N8", "H7N9",
    "H9N2", "H10N7", "H10N8",
]

_SUBTYPE_RE = re.compile(r"\bH([1-9]|1[0-6])N([1-9]|1[0-2])\b", re.IGNORECASE)
_STRAIN_RE = re.compile(r"\(([^)]+)\)")
_HOST_POULTRY = ("chicken", "gallus", "turkey", "meleagris", "quail", "coturnix", "poultry", "duck farm")
_HOST_WILD = ("mallard", "anas", "wild bird", "swan", "goose", "anser", "teal", "wigeon", "gull", "shorebird")

CONTAMINANT_MARKERS = (
    "batillaria", "acinetobacter", "rattus", "mus musculus", "patent",
    "synthetic construct", "vaccine construct", "plasmid", "vector",
    "homo sapiens", "escherichia", "bacteriophage", "scaffold",
)


def normalize_subtype(text: str) -> Optional[str]:
    if not text:
        return None
    match = _SUBTYPE_RE.search(text.replace(" ", ""))
    if not match:
        match = _SUBTYPE_RE.search(text)
    if not match:
        return None
    return f"H{int(match.group(1))}N{int(match.group(2))}"


def parse_segment(text: str) -> Optional[str]:
    if not text:
        return None
    upper = text.upper()
    for key, canon in SEGMENT_ALIASES.items():
        if re.search(rf"\b{re.escape(key)}\b", upper):
            return canon
    return None


def isolate_id_from_header(header: str, accession: str) -> str:
    match = _STRAIN_RE.search(header or "")
    if match:
        return re.sub(r"\s+", "_", match.group(1).strip())
    return accession.split(".")[0]


def host_ecology(host: str) -> str:
    raw = (host or "").lower()
    if any(tok in raw for tok in _HOST_POULTRY):
        return "poultry"
    if any(tok in raw for tok in _HOST_WILD):
        return "wild-bird"
    return "unknown"


def looks_contaminated(header: str, sequence: str) -> Optional[str]:
    blob = f"{header} {sequence[:80]}".lower()
    for marker in CONTAMINANT_MARKERS:
        if marker in blob:
            return marker
    if len(sequence) > 8000:
        return "excessive_length"
    return None


def translate_frame(seq: str, frame: int = 0) -> str:
    table = {
        "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
        "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
        "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
        "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
        "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
        "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
        "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
        "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    }
    seq = seq.upper().replace("U", "T")
    aa = []
    for i in range(frame, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        aa.append(table.get(codon, "X"))
    return "".join(aa)


def classify_ha_pathotype(sequence: str) -> Tuple[str, str]:
    """Return (LPAI|HPAI|INDETERMINATE, motif_or_reason) from HA nucleotide sequence."""
    seq = re.sub(r"[^ACGTU]", "", sequence.upper().replace("U", "T"))
    if len(seq) < 200:
        return INDETERMINATE_PATHOTYPE, "too_short"
    best = INDETERMINATE_PATHOTYPE
    motif = "no_cleavage_motif"
    for frame in range(3):
        protein = translate_frame(seq, frame)
        for m in re.finditer(r"[KR]{2,6}.{0,3}GLF", protein):
            window = m.group(0)
            basics = sum(ch in "KR" for ch in window[:6])
            if basics >= 3 or re.search(r"R.K?RGLF|R.RRGLF", window):
                return "HPAI", window
            best = "LPAI"
            motif = window
        if re.search(r"QRETRGLF|PQIETRGLF|PEKTRGLF|PENPKTRGLF", protein):
            best = "LPAI"
            motif = "monobasic"
    return best, motif
