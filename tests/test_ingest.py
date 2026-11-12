"""
tests/test_ingest.py
====================
Unit tests for scripts/ingest_ncbi_ena.py parsing, header extraction,
and checkpointing.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.ingest_ncbi_ena import (
    ALL_AVIAN_SUBTYPES,
    VIRAL_SUBSTRANDS,
    IngestCheckpoint,
    _looks_synthetic,
    _normalise_seq,
    _parse_fasta_text,
    _parse_influenza_header,
    _sha256,
)


class TestIngestModules(unittest.TestCase):

    def test_catalogues_presence(self):
        """Verify avian subtypes and segments are fully specified."""
        self.assertGreaterEqual(len(ALL_AVIAN_SUBTYPES), 30)
        self.assertIn("H5N1", ALL_AVIAN_SUBTYPES)
        self.assertIn("H7N9", ALL_AVIAN_SUBTYPES)
        self.assertIn("H9N2", ALL_AVIAN_SUBTYPES)
        self.assertIn("H10N8", ALL_AVIAN_SUBTYPES)

        segment_names = [s[0] for s in VIRAL_SUBSTRANDS]
        self.assertEqual(len(segment_names), 8)
        self.assertEqual(segment_names, ["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"])

    def test_normalise_seq(self):
        """Verify normalisation maps valid ACGTN and replaces invalid chars."""
        raw = "acgtuRYKWSMBDHVN- "
        norm = _normalise_seq(raw)
        self.assertTrue(all(c in "ACGTN" for c in norm))
        self.assertTrue(norm.startswith("ACGT"))

    def test_sha256(self):
        """Verify deterministic SHA-256 computation."""
        seq = "ATGC"
        sha = _sha256(seq)
        self.assertEqual(len(sha), 64)
        self.assertEqual(sha, _sha256(seq))

    def test_parse_fasta_text(self):
        """Verify multi-record FASTA parsing."""
        fasta = ">acc1 First sequence\nACGT\nTGCA\n>acc2 Second sequence\nGGCC\n"
        records = _parse_fasta_text(fasta)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0][0], "acc1")
        self.assertEqual(records[0][2], "ACGTTGCA")
        self.assertEqual(records[1][0], "acc2")
        self.assertEqual(records[1][2], "GGCC")

    def test_header_metadata_parsing(self):
        """Verify influenza header extraction across different subtypes and hosts."""
        h1 = (
            "QB294628.1 Influenza A virus "
            "(A/Mallard Duck/UT/24-025488-075-original/2024(H5N6)) "
            "segment 8 nuclear export protein (NEP) and nonstructural protein 1 (NS1) genes, complete cds"
        )
        meta1 = _parse_influenza_header(h1)
        self.assertEqual(meta1["subtype"], "H5N6")
        self.assertEqual(meta1["segment"], "NS")
        self.assertEqual(meta1["host"], "Mallard Duck")
        self.assertEqual(meta1["country"], "UT")
        self.assertEqual(meta1["collection_date"], "2024")

        h2 = (
            "QB050203.1 Influenza A virus "
            "(A/Green-winged Teal/NC/24-036121-001-original/2024(H7N9)) "
            "segment 4 hemagglutinin (HA) gene, partial cds"
        )
        meta2 = _parse_influenza_header(h2)
        self.assertEqual(meta2["subtype"], "H7N9")
        self.assertEqual(meta2["segment"], "HA")
        self.assertEqual(meta2["host"], "Green-winged Teal")

    def test_looks_synthetic(self):
        """Verify synthetic / patent construct detection."""
        self.assertTrue(_looks_synthetic("Patent WO2020123456 recombinant expression"))
        self.assertTrue(_looks_synthetic("Synthetic construct clone pUC19"))
        self.assertFalse(_looks_synthetic("Influenza A virus (A/duck/Vietnam/2024(H5N1))"))

    def test_checkpoint(self):
        """Verify ingest checkpoint save and resume."""
        with tempfile.TemporaryDirectory() as tmp:
            cp_path = Path(tmp) / "cp.json"
            cp = IngestCheckpoint(cp_path)
            self.assertFalse(cp.is_completed("subtype_H5N1"))
            cp.record_completed("subtype_H5N1", 50)
            self.assertTrue(cp.is_completed("subtype_H5N1"))

            # Reload from disk
            cp2 = IngestCheckpoint(cp_path)
            self.assertTrue(cp2.is_completed("subtype_H5N1"))
            self.assertEqual(cp2.data["stats"]["subtype_H5N1"], 50)


if __name__ == "__main__":
    unittest.main()
