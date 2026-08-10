"""
evaluate_advanced.py
====================
Advanced Production Entry Point for Testing, Evaluation, and Batch Inference

Usage:
    python evaluate_advanced.py --config configs/train_config.yaml
    python evaluate_advanced.py --checkpoint output_advanced/checkpoints/best_model.pt --dataset-root dataset_augmented
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
from Genomic_Framework.pipeline.preprocessor import GenomicDatasetBuilder
from Genomic_Framework.inference.advanced_evaluator import AdvancedEvaluator

def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))

    root.addHandler(ch)
    root.addHandler(fh)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Advanced Evaluation & Testing Pipeline")
    p.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"), help="Path to YAML configuration file")
    p.add_argument("--checkpoint", type=Path, default=None, help="Path to model checkpoint .pt file")
    p.add_argument("--dataset-root", type=Path, default=None, help="Override dataset root path")
    p.add_argument("--output-dir", type=Path, default=None, help="Override output directory path")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size for testing")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.config.exists():
        with args.config.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "paths": {"dataset_root": "dataset_augmented", "output_dir": "output_advanced"},
            "model": {"seq_len": 128, "vocab_size": 4096, "hyena_dim": 256, "hidden_dim": 512, "num_classes": 4, "max_mutations": 32, "mutation_dim": 64},
            "data": {"batch_size": 16, "num_workers": 0, "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
            "system": {"seed": 42},
            "evaluation": {"formats": ["csv", "json", "excel"]},
        }

    # CLI Overrides
    if args.dataset_root:
        config["paths"]["dataset_root"] = str(args.dataset_root)
    if args.output_dir:
        config["paths"]["output_dir"] = str(args.output_dir)
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size

    output_dir = Path(config["paths"].get("output_dir", "output_advanced"))
    log_dir = output_dir / "logs"
    configure_logging(log_dir / "inference.log")

    checkpoint_path = args.checkpoint or (output_dir / "checkpoints" / "best_model.pt")

    if not checkpoint_path.exists():
        logger.error("Checkpoint file not found at: %s", checkpoint_path)
        return 1

    logger.info("Starting Advanced Testing & Evaluation Pipeline using Checkpoint: %s", checkpoint_path)

    dataset_root = Path(config["paths"]["dataset_root"])
    builder = GenomicDatasetBuilder(
        dataset_root=dataset_root,
        seq_len=config["model"]["seq_len"],
        batch_size=config["data"]["batch_size"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        num_workers=config["data"]["num_workers"],
        seed=config["system"]["seed"],
        stratified=config["data"].get("stratified", True),
        k_mer=config["model"].get("k_mer_size", 6),
    )

    try:
        train_loader, val_loader, test_loader = builder.build_dataloaders()
        # class_weights is stored as builder.class_weights property after build_dataloaders()
        _class_weights = builder.class_weights
    except Exception as exc:
        logger.error("Failed to build test DataLoader: %s", exc)
        return 1

    engine = GenomicClassificationEngine(
        seq_len=config["model"]["seq_len"],
        vocab_size=config["model"]["vocab_size"],
        hyena_dim=config["model"]["hyena_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
        max_mutations=config["model"]["max_mutations"],
        mutation_dim=config["model"]["mutation_dim"],
    )

    evaluator = AdvancedEvaluator(
        model=engine,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )

    formats = config.get("evaluation", {}).get("formats", ["csv", "json", "excel"])
    metrics = evaluator.evaluate(test_loader, save_outputs=True, formats=formats)

    logger.info("=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("  Test Accuracy      : %.4f", metrics["accuracy"])
    logger.info("  Balanced Accuracy  : %.4f", metrics.get("balanced_accuracy", 0.0))
    logger.info("  Macro F1           : %.4f", metrics["macro_f1"])
    logger.info("  Weighted F1        : %.4f", metrics.get("weighted_f1", 0.0))
    roc = metrics["roc_auc_macro"]
    roc_str = f"{roc:.4f}" if (isinstance(roc, float) and not __import__('math').isnan(roc)) else "N/A"
    logger.info("  ROC-AUC (Macro)    : %s", roc_str)
    logger.info("  PR-AUC (Macro)     : %.4f", metrics["pr_auc_macro"])
    logger.info("  MCC                : %.4f", metrics["mcc"])
    logger.info("  Cohen Kappa        : %.4f", metrics["cohen_kappa"])
    if metrics.get("classes_missing_from_test"):
        logger.warning("  Classes missing from test set: %s", metrics["classes_missing_from_test"])
    logger.info("  Full report saved to: %s", (output_dir / "reports").resolve())
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
