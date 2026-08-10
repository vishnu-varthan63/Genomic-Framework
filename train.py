"""
train.py
========
Project-root entry point for the Genomic Sequence Classification Framework
training pipeline.

Usage
-----
    # Full training run (50 epochs, batch size 16)
    python train.py

    # Quick test run
    python train.py --epochs 2 --batch-size 4

    # Dry-run: scan dataset + one forward pass, no training
    python train.py --dry-run

    # Resume from an existing checkpoint
    python train.py --resume checkpoints/best_model.pt --epochs 30

    # Download-only species
    python train.py --dataset-root dataset/ --batch-size 8 --epochs 20 --lr 5e-5

CLI Arguments
-------------
    --seq-len        INT    Token sequence length          (default: 128)
    --batch-size     INT    Training batch size            (default: 16)
    --epochs         INT    Max training epochs            (default: 50)
    --lr             FLOAT  Initial learning rate          (default: 1e-4)
    --patience       INT    Early-stopping patience        (default: 10)
    --checkpoint-dir PATH   Checkpoint output directory    (default: checkpoints/)
    --dataset-root   PATH   Root of dataset directory      (default: dataset/)
    --num-workers    INT    DataLoader workers             (default: 0)
    --seed           INT    Global random seed             (default: 42)
    --resume         PATH   Resume from checkpoint path
    --dry-run               Build loaders + 1 forward pass, do not train

Output Files
------------
    checkpoints/
        checkpoint_epoch_NNN.pt
        best_model.pt
        training_curves.png
    training.log
    training_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Ensure project root is importable when executed directly
# ---------------------------------------------------------------------------
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
)
from Genomic_Framework.pipeline.preprocessor import (
    CLASS_LABELS,
    NUM_CLASSES,
    GenomicDatasetBuilder,
)
from Genomic_Framework.training.trainer import GenomicTrainer

# ===========================================================================
# LOGGING SETUP
# ===========================================================================

_LOG_FILE = _PROJECT_ROOT / "training.log"


def _configure_logging(log_file: Path = _LOG_FILE) -> None:
    """
    Attach an INFO console handler and a DEBUG file handler to the root logger.

    Parameters
    ----------
    log_file : Path — Path to the output log file.
    """
    fmt      = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    # Avoid adding duplicate handlers on repeated calls
    if root.handlers:
        return

    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))

    root.addHandler(ch)
    root.addHandler(fh)


logger = logging.getLogger(__name__)

# ===========================================================================
# ARGUMENT PARSER
# ===========================================================================


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for train.py."""
    p = argparse.ArgumentParser(
        prog="train.py",
        description="Train the Genomic Sequence Classification Framework.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Architecture
    p.add_argument("--seq-len",    type=int,   default=SEQ_LEN,
                   help="Token sequence length (must match engine SEQ_LEN).")
    # DataLoader
    p.add_argument("--batch-size", type=int,   default=16,
                   help="Mini-batch size.")
    p.add_argument("--num-workers",type=int,   default=0,
                   help="DataLoader worker processes.")
    # Optimisation
    p.add_argument("--epochs",     type=int,   default=50,
                   help="Maximum training epochs.")
    p.add_argument("--min-epochs", type=int,   default=25,
                   help="Minimum training epochs before early stopping can trigger.")
    p.add_argument("--lr",         type=float, default=1e-4,
                   help="Initial AdamW learning rate.")
    p.add_argument("--patience",   type=int,   default=25,
                   help="Early-stopping patience (epochs without improvement).")
    # Paths
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"),
                   help="Directory to save checkpoint files.")
    p.add_argument("--dataset-root",   type=Path, default=Path("dataset"),
                   help="Root directory containing raw/ and healthy/ subdirs.")
    # Misc
    p.add_argument("--seed",   type=int,  default=42,
                   help="Global random seed for reproducibility.")
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to a checkpoint .pt file to resume from.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build DataLoaders, run one forward pass, then exit.")

    return p


# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================


def _set_seeds(seed: int) -> None:
    """Set Python, NumPy (if available), and PyTorch random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def _dry_run(
    engine: GenomicClassificationEngine,
    builder: GenomicDatasetBuilder,
    device: torch.device,
    args: argparse.Namespace,
) -> int:
    """
    Build DataLoaders, print the class distribution, run one forward pass,
    and validate tensor shapes.

    Parameters
    ----------
    engine  : GenomicClassificationEngine — Assembled model.
    builder : GenomicDatasetBuilder       — Dataset builder (already built loaders).
    device  : torch.device
    args    : argparse.Namespace

    Returns
    -------
    int — Exit code (0 = success).
    """
    train_loader, val_loader, test_loader = builder.build_dataloaders()

    # ---- Class distribution table ----
    print()
    print("  Class Distribution (training files):")
    print(f"  {'Class':<22}  {'Files':>6}  {'Label':>5}")
    print(f"  {'-'*40}")
    for i, name in enumerate(CLASS_LABELS):
        cnt = builder.class_distribution.get(i, 0)
        print(f"  {name:<22}  {cnt:>6}  {i:>5}")
    print()

    # ---- One forward pass ----
    engine.to(device)
    engine.eval()

    batch = next(iter(train_loader))
    token_ids, mut_tensor, labels = batch
    token_ids  = token_ids.to(device)
    mut_tensor = mut_tensor.to(device)
    labels     = labels.to(device)

    print("  One-batch forward-pass verification:")
    print(f"    token_ids   shape : {list(token_ids.shape)}")
    print(f"    mut_tensor  shape : {list(mut_tensor.shape)}")
    print(f"    labels      shape : {list(labels.shape)}")

    with torch.no_grad():
        logits, fused = engine(token_ids, mut_tensor)

    print(f"    logits      shape : {list(logits.shape)}  (expected [B, {NUM_CLASSES}])")
    print(f"    fused       shape : {list(fused.shape)}")

    # Assertions
    B = token_ids.size(0)
    assert logits.shape == (B, NUM_CLASSES), (
        f"Logits shape mismatch: got {list(logits.shape)}, "
        f"expected [{B}, {NUM_CLASSES}]"
    )
    assert fused.shape == (B, HIDDEN_DIM), (
        f"Fused shape mismatch: got {list(fused.shape)}, "
        f"expected [{B}, {HIDDEN_DIM}]"
    )

    print()
    print("  [OK] All tensor shapes verified. Dry-run complete.")
    print()

    logger.info(
        "Dry-run complete — "
        "train_batches=%d  val_batches=%d  test_batches=%d",
        len(train_loader), len(val_loader), len(test_loader),
    )
    return 0


def _save_summary(
    summary: dict, path: Path
) -> None:
    """Write *summary* as formatted JSON to *path*."""
    path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Training summary saved: %s", path)


def _print_summary_table(history: dict, args: argparse.Namespace) -> None:
    """Print a human-readable summary to stdout."""
    test = history.get("test_metrics", {})
    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best epoch    : {history.get('best_epoch', '?')}")
    print(f"  Best val loss : {history.get('best_val_loss', 0.0):.6f}")
    print(f"  Test accuracy : {test.get('accuracy', 0.0):.4f}")
    print(f"  Test macro F1 : {test.get('macro_f1', 0.0):.4f}")
    print()
    print("  Per-class F1 on test set:")
    for name, f1 in test.get("per_class_f1", {}).items():
        print(f"    {name:<22} : {f1:.4f}")
    print()
    print(f"  Checkpoints   : {args.checkpoint_dir.resolve()}")
    print(f"  Log file      : {_LOG_FILE.resolve()}")
    print(f"  Summary JSON  : {(_PROJECT_ROOT / 'training_summary.json').resolve()}")
    print("=" * 60)
    print()


# ===========================================================================
# MAIN
# ===========================================================================


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point.

    Parameters
    ----------
    argv : list[str] | None — Override ``sys.argv`` (useful for testing).

    Returns
    -------
    int — Exit code: 0 = success, 1 = failure.
    """
    _configure_logging()
    args = _build_parser().parse_args(argv)
    _set_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)
    logger.info("Genomic Sequence Classification Framework — Training")
    logger.info("Device       : %s", device)
    logger.info("Dataset root : %s", args.dataset_root.resolve())
    logger.info("Checkpoint   : %s", args.checkpoint_dir.resolve())
    logger.info("Epochs       : %d  |  Batch: %d  |  LR: %.2e",
                args.epochs, args.batch_size, args.lr)
    logger.info("=" * 60)

    # ---- Dataset builder ----
    builder = GenomicDatasetBuilder(
        dataset_root=args.dataset_root,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        class_balance=True,
    )

    # ---- Engine ----
    engine = GenomicClassificationEngine(
        seq_len=args.seq_len,
        vocab_size=VOCAB_SIZE,
        hyena_dim=HYENA_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        max_mutations=MAX_MUTATIONS,
        mutation_dim=MUTATION_DIM,
    )

    # ---- Dry-run ----
    if args.dry_run:
        return _dry_run(engine, builder, device, args)

    # ---- Build DataLoaders ----
    try:
        train_loader, val_loader, test_loader = builder.build_dataloaders()
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Dataset error: %s", exc)
        return 1

    if len(train_loader.dataset) == 0:
        logger.error(
            "Training dataset is empty. "
            "Ensure FASTA files exist in dataset/raw/ and dataset/healthy/."
        )
        return 1

    class_weights = builder.class_weights

    # ---- Trainer ----
    trainer = GenomicTrainer(
        model=engine,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_weights=class_weights,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
    )

    # ---- Resume ----
    start_epoch = 0
    if args.resume is not None:
        try:
            start_epoch = trainer.resume_from(args.resume)
        except FileNotFoundError as exc:
            logger.error("Resume checkpoint not found: %s", exc)
            return 1

    # ---- Train (with KeyboardInterrupt guard) ----
    history: dict = {}
    run_date = datetime.now(timezone.utc).isoformat()
    try:
        history = trainer.train()
    except KeyboardInterrupt:
        logger.warning(
            "KeyboardInterrupt — saving current checkpoint and exiting …"
        )
        trainer.save_checkpoint(
            epoch=trainer._current_epoch,
            val_loss=float("inf"),
            is_best=False,
        )
        return 1
    except Exception:
        logger.error("Unhandled training error:\n%s", traceback.format_exc())
        return 1

    # ---- Dataset statistics ----
    train_n = len(train_loader.dataset)
    val_n   = len(val_loader.dataset)
    test_n  = len(test_loader.dataset)

    class_dist = {
        CLASS_LABELS[i]: builder.class_distribution.get(i, 0)
        for i in range(NUM_CLASSES)
    }

    # ---- Build summary ----
    summary = {
        "run_date": run_date,
        "config": {
            "seq_len":       args.seq_len,
            "batch_size":    args.batch_size,
            "epochs_run":    len(history.get("train_loss", [])),
            "best_epoch":    history.get("best_epoch", 0),
            "learning_rate": args.lr,
            "patience":      args.patience,
            "optimizer":     "AdamW",
            "scheduler":     "ReduceLROnPlateau",
            "device":        str(device),
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        },
        "dataset": {
            "root":              str(args.dataset_root.resolve()),
            "total_sequences":   train_n + val_n + test_n,
            "train_sequences":   train_n,
            "val_sequences":     val_n,
            "test_sequences":    test_n,
            "class_distribution": class_dist,
        },
        "best_val_metrics": {
            "best_epoch": history.get("best_epoch", 0),
            "val_loss":   history.get("best_val_loss", None),
            "val_accuracy": (
                history["val_accuracy"][history["best_epoch"] - 1]
                if history.get("val_accuracy") and history.get("best_epoch")
                else None
            ),
            "val_f1": (
                history["val_f1"][history["best_epoch"] - 1]
                if history.get("val_f1") and history.get("best_epoch")
                else None
            ),
        },
        "test_metrics": history.get("test_metrics", {}),
    }

    summary_path = args.checkpoint_dir / "training_summary.json"
    _save_summary(summary, summary_path)
    _save_summary(summary, _PROJECT_ROOT / "training_summary.json")
    _print_summary_table(history, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
