"""
train_advanced.py
=================
Advanced Production Entry Point for Model Training & Optimization

Usage:
    python train_advanced.py --config configs/train_config.yaml
    python train_advanced.py --config configs/train_config.yaml --epochs 30 --batch-size 32
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch

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
from Genomic_Framework.training.advanced_trainer import AdvancedTrainer, set_seed

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
    p = argparse.ArgumentParser(description="Advanced Training & Optimization Pipeline")
    p.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"), help="Path to YAML configuration file")
    p.add_argument("--dataset-root", type=Path, default=None, help="Override dataset root path")
    p.add_argument("--output-dir", type=Path, default=None, help="Override output directory path")
    p.add_argument("--epochs", type=int, default=None, help="Override maximum epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    p.add_argument("--seed", type=int, default=None, help="Override seed")
    p.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.config.exists():
        print(f"Error: Configuration file not found at {args.config}")
        return 1

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # CLI Overrides
    if args.dataset_root:
        config["paths"]["dataset_root"] = str(args.dataset_root)
    if args.output_dir:
        config["paths"]["output_dir"] = str(args.output_dir)
    if args.epochs:
        config["training"]["max_epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.lr:
        config["optimization"]["learning_rate"] = args.lr
    if args.seed:
        config["system"]["seed"] = args.seed

    output_dir = Path(config["paths"].get("output_dir", "output_advanced"))
    log_dir = output_dir / config["paths"].get("log_dir", "logs")
    configure_logging(log_dir / "training.log")

    num_threads = int(config.get("system", {}).get("num_threads", 6))
    torch.set_num_threads(num_threads)
    logger.info("Starting Advanced Training Script with Config: %s (torch threads=%d)", args.config, num_threads)

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
        class_balance=config["data"]["class_balance"],
        samples_per_epoch=config["data"].get("samples_per_epoch", None),
    )

    try:
        train_loader, val_loader, test_loader = builder.build_dataloaders()
    except Exception as exc:
        logger.error("Failed to build DataLoaders: %s", exc)
        return 1

    class_weights = builder.class_weights if config["data"].get("class_weighting", True) else None

    engine = GenomicClassificationEngine(
        seq_len=config["model"]["seq_len"],
        vocab_size=config["model"]["vocab_size"],
        hyena_dim=config["model"]["hyena_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
        max_mutations=config["model"]["max_mutations"],
        mutation_dim=config["model"]["mutation_dim"],
        dropout=config["model"].get("dropout", 0.2),
    )

    trainer = AdvancedTrainer(
        model=engine,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        class_weights=class_weights,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    history = trainer.train()
    logger.info("Advanced Training Completed Successfully. Best Epoch: %d", history.get("best_epoch", 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
