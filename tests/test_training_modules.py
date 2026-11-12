"""
tests/test_training_modules.py
==============================
Unit tests for Genomic Framework: Tokeniser, QC, Leakage, Multi-Task Engine, Trainer.
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile
import unittest

# Ensure project root is in sys.path
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Compatibility guard for Windows AppLocker/WDAC policy blocking optree._C.pyd
if "torch._dynamo" not in sys.modules:
    try:
        import optree  # noqa: F401
    except Exception:
        import types
        class _DummyDynamo(types.ModuleType):
            __file__ = "dummy.py"
            def disable(self, *args, **kwargs):
                if args and callable(args[0]):
                    return args[0]
                return lambda f: f
            def graph_break(self, *args, **kwargs):
                pass
            def __getattr__(self, name):
                if name.startswith("__"):
                    raise AttributeError(name)
                return lambda *args, **kwargs: None
        sys.modules["torch._dynamo"] = _DummyDynamo("torch._dynamo")

import torch
from torch.utils.data import DataLoader, TensorDataset

import Genomic_Framework
from Genomic_Framework.engine import (
    GenomicClassificationEngine,
    CLASS_LABELS,
    PATHOTYPE_LABELS,
    SEGMENT_LABELS,
)
from Genomic_Framework.pipeline.input_qc import GenomicSequenceIngester
from Genomic_Framework.pipeline.preprocessor import (
    GenomicDataset,
    tokenise_sequence,
)
from Genomic_Framework.training.trainer import MetricsTracker, GenomicTrainer
from scripts.accession_split import _stratified_group_split


class TestTrainingModules(unittest.TestCase):
    """Core verification test suite for data pipeline, QC, model heads, and trainer."""

    def test_tokeniser_deterministic(self):
        t1 = tokenise_sequence("ACGTACGTACGT", seq_len=16, vocab_size=4096, k=6)
        t2 = tokenise_sequence("ACGTACGTACGT", seq_len=16, vocab_size=4096, k=6)
        self.assertEqual(t1.shape, (16,))
        self.assertEqual(t1.dtype, torch.long)
        self.assertEqual(t1.tolist(), t2.tolist())
        # First valid positions must be non-zero
        self.assertTrue((t1[:7] > 0).all())

    def test_taxonomy_labels(self):
        self.assertEqual(len(CLASS_LABELS), 4)
        self.assertIn("Avirulent", CLASS_LABELS)
        self.assertIn("Low_Pathogenicity", CLASS_LABELS)
        self.assertIn("Moderate_High_Pathogenicity", CLASS_LABELS)
        self.assertIn("Highly_Pathogenic", CLASS_LABELS)
        self.assertEqual(len(PATHOTYPE_LABELS), 3)
        self.assertEqual(len(SEGMENT_LABELS), 8)

    def test_qc_pipeline_ingestion(self):
        ingester = GenomicSequenceIngester(max_seq_len=64, strict_validation=True)
        raw_seqs = [
            ">test_clean\nACGTACGTACGTACGT",
            ">test_with_n\nACGTNNNNACGT",
            ">test_invalid\nXYZ123!@#",
        ]
        cleaned, log = ingester.clean_and_validate(raw_seqs)
        self.assertGreaterEqual(len(cleaned), 2)
        for seq in cleaned:
            self.assertTrue(set(seq).issubset(set("ACGTN")))

    def test_zero_accession_leakage(self):
        groups = {
            f"isolate_{i}": [f"acc_{i}_seg{s}" for s in range(1, 9)]
            for i in range(20)
        }
        strata = {
            f"acc_{i}_seg{s}": f"H5N1_HPAI" if i % 2 == 0 else "H9N2_LPAI"
            for i in range(20)
            for s in range(1, 9)
        }
        train_accs, val_accs, test_accs = _stratified_group_split(
            groups, strata, train_ratio=0.70, val_ratio=0.15, seed=42
        )
        set_train = set(train_accs)
        set_val = set(val_accs)
        set_test = set(test_accs)
        self.assertEqual(len(set_train & set_val), 0, "Train and Val must have 0 overlap")
        self.assertEqual(len(set_train & set_test), 0, "Train and Test must have 0 overlap")
        self.assertEqual(len(set_val & set_test), 0, "Val and Test must have 0 overlap")
        self.assertEqual(len(set_train | set_val | set_test), 160)

    def test_multitask_engine_forward(self):
        engine = GenomicClassificationEngine(
            seq_len=32,
            vocab_size=256,
            hyena_dim=64,
            hidden_dim=128,
            num_classes=4,
            num_subtypes=10,
            num_segments=8,
            num_pathotypes=3,
            max_mutations=8,
            mutation_dim=64,
        )
        engine.eval()
        tok = torch.randint(1, 200, (2, 32))
        mut = torch.randn(2, 8, 64)

        # Standard forward compatibility
        logits, fused = engine(tok, mut)
        self.assertEqual(logits.shape, (2, 4))
        self.assertEqual(fused.shape, (2, 128))

        # Multi-task forward
        mt_out = engine.forward_multitask(tok, mut)
        self.assertIn("logits", mt_out)
        self.assertIn("subtype_logits", mt_out)
        self.assertIn("segment_logits", mt_out)
        self.assertIn("pathotype_logits", mt_out)
        self.assertEqual(mt_out["subtype_logits"].shape, (2, 10))
        self.assertEqual(mt_out["segment_logits"].shape, (2, 8))
        self.assertEqual(mt_out["pathotype_logits"].shape, (2, 3))

    def test_metrics_tracker(self):
        tracker = MetricsTracker(num_classes=4, class_labels=CLASS_LABELS)
        logits = torch.tensor([[2.0, 0.5, 0.1, 0.0], [0.0, 0.1, 2.0, 0.0]])
        labels = torch.tensor([0, 2])
        tracker.update(logits, labels, loss=0.5)
        m = tracker.compute()
        self.assertEqual(m["accuracy"], 1.0)
        self.assertIn("macro_f1", m)
        self.assertIn("confusion_matrix", m)

    def test_genomic_dataset_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            fasta_file = tmp_path / "test.fasta"
            fasta_file.write_text(
                ">seq1\nACGTACGTACGTACGTACGTACGTACGTACGT\n"
                ">seq2\nTTTTAAAACCCCGGGGTTTTAAAACCCCGGGG\n",
                encoding="utf-8",
            )
            ds = GenomicDataset(
                fasta_paths=[(fasta_file, 0)],
                seq_len=16,
                vocab_size=256,
                max_mutations=8,
                mutation_dim=64,
                max_seq_len_qc=512,
                cache_tokens=True,
            )
            self.assertEqual(len(ds), 2)
            tok, mut, lbl = ds[0]
            self.assertEqual(tok.shape, (16,))
            self.assertEqual(mut.shape, (8, 64))
            self.assertEqual(lbl.item(), 0)

    def test_trainer_cycle(self):
        def _dummy_loader():
            toks = torch.randint(1, 100, (4, 16))
            muts = torch.zeros(4, 8, 64)
            lbls = torch.randint(0, 4, (4,))
            return DataLoader(TensorDataset(toks, muts, lbls), batch_size=2)

        engine = GenomicClassificationEngine(
            seq_len=16, vocab_size=256, hyena_dim=64, hidden_dim=128,
            num_classes=4, max_mutations=8, mutation_dim=64,
        )
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_tmp = pathlib.Path(tmp) / "checkpoints"
            trainer = GenomicTrainer(
                model=engine,
                train_loader=_dummy_loader(),
                val_loader=_dummy_loader(),
                test_loader=_dummy_loader(),
                num_epochs=1,
                patience=2,
                checkpoint_dir=ckpt_tmp,
            )
            trainer._current_epoch = 1
            train_m = trainer.train_epoch()
            val_m = trainer.validate_epoch(trainer.val_loader)
            self.assertIn("loss", train_m)
            self.assertIn("accuracy", val_m)
            trainer.save_checkpoint(epoch=1, val_loss=val_m["loss"], is_best=True)
            ckpt_path = ckpt_tmp / "best_model.pt"
            self.assertTrue(ckpt_path.exists())
            loaded = trainer.load_checkpoint(ckpt_path)
            self.assertEqual(loaded, 1)


if __name__ == "__main__":
    unittest.main()
