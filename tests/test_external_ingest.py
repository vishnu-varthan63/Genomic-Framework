"""
tests/test_external_ingest.py
=============================
Unit tests for Option A (BV-BRC) and Option B (Nextstrain) external ingestion:
- GenBank flat-file parsing (pure Python)
- Subtype and segment extraction
- Checkpoint persistence
- Data hygiene & normalization
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.ingest_external_sources import (
    ALL_AVIAN_SUBTYPES,
    NEXTSTRAIN_LINEAGES,
    NEXTSTRAIN_SEGMENTS,
    ExternalIngestCheckpoint,
    _looks_synthetic,
    _normalise_seq,
    _parse_genbank_record,
    _parse_segment_from_text,
    _sha256,
)


class TestExternalIngest(unittest.TestCase):

    def test_catalogues(self):
        """Verify Nextstrain lineages, segments, and BV-BRC subtypes catalogues."""
        self.assertIn("h5n1", NEXTSTRAIN_LINEAGES)
        self.assertIn("h7n9", NEXTSTRAIN_LINEAGES)
        self.assertIn("ha", NEXTSTRAIN_SEGMENTS)
        self.assertIn("pb2", NEXTSTRAIN_SEGMENTS)
        self.assertGreaterEqual(len(ALL_AVIAN_SUBTYPES), 30)

    def test_parse_genbank_record(self):
        """Verify pure Python GenBank flat file parser extracts accession, definition, and sequence."""
        gb_sample = """LOCUS       AF144305                1760 bp    RNA     linear   VRL 31-AUG-1999
DEFINITION  Influenza A virus (A/Goose/Guangdong/1/96(H5N1)) hemagglutinin (HA)
            gene, complete cds.
ACCESSION   AF144305
VERSION     AF144305.1
ORIGIN      
        1 gcaggggtat aatctgtcaa aatggagaaa atagtgcttc ttcttgcaat agtcagtctt
       61 gttaaaagtg atcagatttg cattggttac catgcaaaca actcgacaga gcaggttgac
//
"""
        parsed = _parse_genbank_record(gb_sample, default_sub="h5n1", default_seg="ha")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["accession"], "AF144305")
        self.assertEqual(parsed["segment"], "HA")
        self.assertEqual(parsed["subtype"], "H5N1")
        self.assertEqual(parsed["host"], "Goose")
        self.assertTrue(parsed["sequence"].startswith("GCAGGGGTATAATCTG"))
        self.assertEqual(len(parsed["sequence"]), 120)

    def test_segment_from_text(self):
        """Verify segment parsing across names and numbers."""
        self.assertEqual(_parse_segment_from_text("polymerase basic 2 (PB2)"), "PB2")
        self.assertEqual(_parse_segment_from_text("segment 4 hemagglutinin"), "HA")
        self.assertEqual(_parse_segment_from_text("neuraminidase (NA) gene"), "NA")
        self.assertEqual(_parse_segment_from_text("segment 7 matrix protein"), "M")
        self.assertEqual(_parse_segment_from_text("nonstructural protein 1 (NS1)"), "NS")

    def test_normalise_and_sha256(self):
        """Verify nucleotide normalization and deterministic SHA-256."""
        raw = "acgtur ykws-n"
        norm = _normalise_seq(raw)
        self.assertTrue(all(c in "ACGTN" for c in norm))
        self.assertEqual(_sha256(norm), _sha256(norm))

    def test_synthetic_detection(self):
        """Verify patent and synthetic construct detection."""
        self.assertTrue(_looks_synthetic("Patent US987654 recombinant influenza"))
        self.assertTrue(_looks_synthetic("Cloning vector pCDNA3.1"))
        self.assertFalse(_looks_synthetic("Influenza A virus (A/Goose/Guangdong/1/96(H5N1))"))

    def test_checkpoint(self):
        """Verify checkpoint recording and persistence."""
        with tempfile.TemporaryDirectory() as tmp:
            cp_path = Path(tmp) / "ext_cp.json"
            cp = ExternalIngestCheckpoint(cp_path)
            self.assertFalse(cp.is_completed("bvbrc_subtype_H5N1"))
            cp.record_completed("bvbrc_subtype_H5N1", 50)
            self.assertTrue(cp.is_completed("bvbrc_subtype_H5N1"))

            cp_reloaded = ExternalIngestCheckpoint(cp_path)
            self.assertTrue(cp_reloaded.is_completed("bvbrc_subtype_H5N1"))
            self.assertEqual(cp_reloaded.data["stats"]["bvbrc_subtype_H5N1"], 50)


if __name__ == "__main__":
    unittest.main()

