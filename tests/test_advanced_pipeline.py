"""
tests/test_advanced_pipeline.py
================================
Unit Tests for Advanced Training, Optimization, and Evaluation Pipeline
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Genomic_Framework.engine import (
    GenomicClassificationEngine,
    SEQ_LEN,
    VOCAB_SIZE,
    HYENA_DIM,
    HIDDEN_DIM,
    MAX_MUTATIONS,
    MUTATION_DIM,
    NUM_CLASSES,
)
from Genomic_Framework.training.advanced_trainer import (
    AdvancedMetricsTracker,
    AdvancedTrainer,
    detect_device,
    set_seed,
)
from Genomic_Framework.inference.advanced_evaluator import AdvancedEvaluator


class TestAdvancedPipeline(unittest.TestCase):

    def setUp(self):
        set_seed(42, deterministic=True)
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = pathlib.Path(self.tmp_dir)

        # Mock Model & Data
        self.model = GenomicClassificationEngine(
            seq_len=SEQ_LEN,
            vocab_size=VOCAB_SIZE,
            hyena_dim=HYENA_DIM,
            hidden_dim=HIDDEN_DIM,
            num_classes=NUM_CLASSES,
            max_mutations=MAX_MUTATIONS,
            mutation_dim=MUTATION_DIM,
        )

        # Create dummy tensors for 16 samples
        tokens = torch.randint(1, 100, (16, SEQ_LEN))
        muts = torch.randn(16, MAX_MUTATIONS, MUTATION_DIM)
        labels = torch.randint(0, NUM_CLASSES, (16,))

        ds = TensorDataset(tokens, muts, labels)
        self.loader = DataLoader(ds, batch_size=4)

        # Config Dict
        self.config = {
            "system": {"seed": 42, "deterministic": True, "device": "cpu", "multi_gpu": False},
            "paths": {
                "dataset_root": "dataset_augmented",
                "output_dir": str(self.tmp_path / "output_test"),
                "checkpoint_dir": "checkpoints",
                "log_dir": "logs",
                "plot_dir": "plots",
                "metric_dir": "metrics",
                "tensorboard_dir": "tensorboard",
                "report_dir": "reports",
            },
            "model": {
                "seq_len": SEQ_LEN,
                "vocab_size": VOCAB_SIZE,
                "hyena_dim": HYENA_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_classes": NUM_CLASSES,
                "max_mutations": MAX_MUTATIONS,
                "mutation_dim": MUTATION_DIM,
            },
            "data": {"batch_size": 4, "num_workers": 0, "class_balance": False},
            "optimization": {
                "optimizer": "adamw",
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "gradient_clipping": 1.0,
                "gradient_accumulation_steps": 1,
                "use_amp": False,
            },
            "scheduler": {"name": "cosine", "min_lr": 1e-6},
            "training": {
                "max_epochs": 2,
                "min_epochs": 1,
                "early_stopping": {"enabled": False},
                "model_selection_metric": "val_f1",
                "checkpoint_interval": 1,
            },
            "evaluation": {"compute_auc": True, "formats": ["csv", "json"]},
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_seed_and_device(self):
        set_seed(123)
        dev = detect_device("cpu")
        self.assertEqual(dev.type, "cpu")

    def test_metrics_tracker(self):
        tracker = AdvancedMetricsTracker(num_classes=4)
        logits = torch.tensor([
            [5.0, 1.0, 0.0, 0.0],  # pred 0
            [0.0, 4.0, 0.0, 0.0],  # pred 1
            [0.0, 0.0, 6.0, 0.0],  # pred 2
            [0.0, 0.0, 0.0, 3.0],  # pred 3
        ])
        labels = torch.tensor([0, 1, 2, 3])
        tracker.update(logits, labels, loss=0.2)
        m = tracker.compute()

        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["macro_f1"], 1.0)
        self.assertAlmostEqual(m["mcc"], 1.0)
        self.assertEqual(m["confusion_matrix"], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def test_trainer_and_evaluator_flow(self):
        trainer = AdvancedTrainer(
            model=self.model,
            train_loader=self.loader,
            val_loader=self.loader,
            test_loader=self.loader,
            config=self.config,
        )

        history = trainer.train()
        self.assertIn("train_loss", history)
        self.assertIn("val_f1", history)
        self.assertEqual(len(history["train_loss"]), 2)

        # Verify best model pt exists
        best_pt = trainer.checkpoint_dir / "best_model.pt"
        self.assertTrue(best_pt.exists())

        # Test Evaluator
        evaluator = AdvancedEvaluator(
            model=self.model,
            checkpoint_path=best_pt,
            output_dir=self.config["paths"]["output_dir"],
        )
        res = evaluator.evaluate(self.loader, save_outputs=True, formats=["csv", "json"])
        self.assertIn("accuracy", res)
        self.assertIn("macro_f1", res)

        # Check export files
        out_path = pathlib.Path(self.config["paths"]["output_dir"])
        self.assertTrue((out_path / "predictions" / "predictions.csv").exists())
        self.assertTrue((out_path / "predictions" / "predictions.json").exists())
        self.assertTrue((out_path / "metrics" / "evaluation_metrics.json").exists())
        self.assertTrue((out_path / "plots" / "confusion_matrix.png").exists())
        self.assertTrue((out_path / "reports" / "evaluation_report.html").exists())


if __name__ == "__main__":
    unittest.main()
