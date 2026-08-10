"""
Genomic_Framework/training/trainer.py
======================================
Full training engine for the Genomic Sequence Classification Framework.

Components
----------
* :class:`MetricsTracker`  — Accumulates TP/FP/FN per class and computes
  Accuracy, Precision, Recall, F1, Macro-F1 and Confusion Matrix
  **from first principles** (no sklearn dependency).

* :class:`GenomicTrainer`  — Complete training loop around
  ``GenomicClassificationEngine`` including AdamW optimiser,
  ReduceLROnPlateau scheduler, gradient clipping, early stopping,
  checkpoint management, and final test-set evaluation.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Genomic_Framework.pipeline.preprocessor import CLASS_LABELS, NUM_CLASSES

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ===========================================================================
# METRICS TRACKER
# ===========================================================================


class MetricsTracker:
    """
    Epoch-level metrics accumulator.

    Tracks true positives, false positives, and false negatives per class
    across an entire epoch without storing predictions in memory.  All
    metrics are computed from first principles with no external dependencies.

    Parameters
    ----------
    num_classes  : int       — Number of target classes.
    class_labels : list[str] — Human-readable class names.

    Examples
    --------
    >>> tracker = MetricsTracker(num_classes=4, class_labels=CLASS_LABELS)
    >>> tracker.update(logits, labels, loss=0.45)
    >>> metrics = tracker.compute()
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        class_labels: Optional[List[str]] = None,
    ) -> None:
        self.num_classes  = num_classes
        self.class_labels = class_labels or [str(i) for i in range(num_classes)]
        self.reset()

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: float,
    ) -> None:
        """
        Accumulate predictions from one batch.

        Parameters
        ----------
        logits : torch.Tensor — Raw logits ``[B, num_classes]``.
        labels : torch.Tensor — Ground-truth integer labels ``[B]``.
        loss   : float        — Scalar loss value for this batch.
        """
        preds = logits.argmax(dim=-1).cpu()
        lbls  = labels.cpu()

        batch_size      = lbls.size(0)
        self._n_batches += 1
        self._total_loss    += loss * batch_size
        self._n_samples     += batch_size
        self._n_correct     += (preds == lbls).sum().item()

        for pred, true in zip(preds.tolist(), lbls.tolist()):
            self._cm[true][pred] += 1

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics for the current epoch.

        Returns
        -------
        dict with keys:
          ``loss``, ``accuracy``, ``macro_f1``, ``macro_precision``,
          ``macro_recall``, ``per_class_f1``, ``per_class_precision``,
          ``per_class_recall``, ``confusion_matrix``.
        """
        eps = 1e-8
        n   = self._n_samples or 1

        accuracy = self._n_correct / n
        avg_loss = self._total_loss / n

        per_f1, per_prec, per_rec = {}, {}, {}

        for c in range(self.num_classes):
            tp = self._cm[c][c]
            fp = sum(self._cm[r][c] for r in range(self.num_classes) if r != c)
            fn = sum(self._cm[c][d] for d in range(self.num_classes) if d != c)

            prec = tp / (tp + fp + eps)
            rec  = tp / (tp + fn + eps)
            f1   = 2 * prec * rec / (prec + rec + eps)

            name              = self.class_labels[c]
            per_prec[name]    = round(prec, 6)
            per_rec[name]     = round(rec,  6)
            per_f1[name]      = round(f1,   6)

        macro_f1   = sum(per_f1.values())   / self.num_classes
        macro_prec = sum(per_prec.values()) / self.num_classes
        macro_rec  = sum(per_rec.values())  / self.num_classes

        return {
            "loss":                round(avg_loss,  6),
            "accuracy":            round(accuracy,  6),
            "macro_f1":            round(macro_f1,  6),
            "macro_precision":     round(macro_prec, 6),
            "macro_recall":        round(macro_rec,  6),
            "per_class_f1":        per_f1,
            "per_class_precision": per_prec,
            "per_class_recall":    per_rec,
            "confusion_matrix":    [list(row) for row in self._cm],
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all accumulators for a new epoch."""
        self._n_batches  = 0
        self._n_samples  = 0
        self._n_correct  = 0
        self._total_loss = 0.0
        # Confusion matrix: _cm[true][pred]
        self._cm: List[List[int]] = [
            [0] * self.num_classes for _ in range(self.num_classes)
        ]

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"MetricsTracker(num_classes={self.num_classes}, "
            f"samples_seen={self._n_samples})"
        )


# ===========================================================================
# GENOMIC TRAINER
# ===========================================================================


class GenomicTrainer:
    """
    Full training engine for :class:`GenomicClassificationEngine`.

    Implements AdamW + ReduceLROnPlateau, gradient clipping, early stopping,
    best-checkpoint tracking, and final test-set evaluation.

    Parameters
    ----------
    model           : nn.Module    — The assembled ``GenomicClassificationEngine``.
    train_loader    : DataLoader   — Training DataLoader.
    val_loader      : DataLoader   — Validation DataLoader.
    test_loader     : DataLoader   — Test DataLoader.
    class_weights   : FloatTensor | None — Shape ``[num_classes]``. If given,
                      passed to ``CrossEntropyLoss`` to handle imbalance.
    learning_rate   : float  — Initial AdamW learning rate.
    weight_decay    : float  — L2 regularisation coefficient.
    num_epochs      : int    — Maximum training epochs.
    patience        : int    — Early-stopping: stop after this many epochs
                               with no val_loss improvement.
    grad_clip       : float  — Maximum gradient norm for clipping.
    checkpoint_dir  : Path   — Where to save ``.pt`` checkpoint files.
    device          : torch.device | None — Inferred automatically if None.
    log_every_n_steps : int  — Log step-level metrics every N batches.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        class_weights: Optional[torch.FloatTensor] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        num_epochs: int = 50,
        min_epochs: int = 25,
        patience: int = 25,
        grad_clip: float = 1.0,
        checkpoint_dir: Path = Path("checkpoints"),
        device: Optional[torch.device] = None,
        log_every_n_steps: int = 10,
    ) -> None:
        self.model           = model
        self.train_loader    = train_loader
        self.val_loader      = val_loader
        self.test_loader     = test_loader
        self.num_epochs      = num_epochs
        self.min_epochs      = min_epochs
        self.patience        = patience
        self.grad_clip       = grad_clip
        self.checkpoint_dir  = Path(checkpoint_dir)
        self.log_every_n_steps = log_every_n_steps

        # Device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        logger.info("Training device: %s", self.device)

        # Loss
        if class_weights is not None:
            cw = class_weights.to(self.device)
        else:
            cw = None
        self.criterion = nn.CrossEntropyLoss(weight=cw)

        # Optimiser
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler — reduce LR on plateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        # Metrics trackers
        self._train_tracker = MetricsTracker(NUM_CLASSES, CLASS_LABELS)
        self._val_tracker   = MetricsTracker(NUM_CLASSES, CLASS_LABELS)

        # State
        self._best_val_loss = float("inf")
        self._best_epoch    = 0
        self._patience_ctr  = 0
        self._current_epoch = 0

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable    = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            "GenomicTrainer initialised — params=%.2fM (trainable=%.2fM), "
            "epochs=%d, patience=%d, lr=%.2e",
            total_params / 1e6, trainable / 1e6,
            num_epochs, patience, learning_rate,
        )

    # ------------------------------------------------------------------
    # Public training API
    # ------------------------------------------------------------------

    def train(self) -> Dict:
        """
        Execute the full training loop.

        Steps per epoch
        ---------------
        1. ``train_epoch()``    — forward + backward + clip + step
        2. ``validate_epoch()`` — evaluate on validation set
        3. Scheduler step on val_loss
        4. Early-stopping check
        5. Save best checkpoint when val_loss improves

        After the loop the best checkpoint is loaded and the model is
        evaluated on the held-out test set.

        Returns
        -------
        dict — Full training history (see module docstring for schema).
        """
        history: Dict = {
            "train_loss":     [],
            "train_accuracy": [],
            "val_loss":       [],
            "val_accuracy":   [],
            "val_f1":         [],
            "learning_rates": [],
            "best_epoch":     0,
            "best_val_loss":  float("inf"),
            "test_metrics":   {},
        }

        logger.info("=" * 68)
        logger.info(
            "Starting training — %d epochs, device=%s",
            self.num_epochs, self.device,
        )
        logger.info("=" * 68)

        for epoch in range(1, self.num_epochs + 1):
            self._current_epoch = epoch

            # ---- Train ----
            train_metrics = self.train_epoch()

            # ---- Validate ----
            val_metrics   = self.validate_epoch(self.val_loader)
            val_loss      = val_metrics["loss"]
            val_f1        = val_metrics["macro_f1"]

            # ---- Scheduler ----
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ---- History ----
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_f1)
            history["learning_rates"].append(current_lr)

            logger.info(
                "Epoch %3d/%d  "
                "train_loss=%.4f  train_acc=%.4f  "
                "val_loss=%.4f  val_acc=%.4f  val_f1=%.4f  lr=%.2e",
                epoch, self.num_epochs,
                train_metrics["loss"], train_metrics["accuracy"],
                val_loss,              val_metrics["accuracy"],
                val_f1,                current_lr,
            )

            # ---- Checkpoint / Early stopping ----
            is_best = val_loss < self._best_val_loss
            if is_best:
                self._best_val_loss = val_loss
                self._best_epoch    = epoch
                self._patience_ctr  = 0
                history["best_epoch"]    = epoch
                history["best_val_loss"] = val_loss
                logger.info(
                    "  ✓ New best val_loss=%.6f at epoch %d",
                    val_loss, epoch,
                )
            else:
                self._patience_ctr += 1

            self.save_checkpoint(epoch, val_loss, is_best)

            if epoch >= self.min_epochs and self._patience_ctr >= self.patience:
                logger.info(
                    "Early stopping triggered at epoch %d after %d epochs without "
                    "improvement (patience=%d, min_epochs=%d).",
                    epoch, self._patience_ctr, self.patience, self.min_epochs,
                )
                break

        # ---- Load best and evaluate on test set ----
        best_ckpt = self.checkpoint_dir / "best_model.pt"
        if best_ckpt.exists():
            self.load_checkpoint(best_ckpt)
            logger.info("Loaded best checkpoint from epoch %d.", self._best_epoch)

        logger.info("Evaluating on test set …")
        test_metrics = self.evaluate(self.test_loader)
        history["test_metrics"] = test_metrics

        logger.info(
            "Test — accuracy=%.4f  macro_f1=%.4f",
            test_metrics["accuracy"], test_metrics["macro_f1"],
        )

        # ---- Training curves ----
        self.plot_training_curves(history)

        logger.info("Training complete.")
        return history

    # ------------------------------------------------------------------
    # Train / validate epoch
    # ------------------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        """
        One full pass over the training DataLoader.

        Returns
        -------
        dict with keys ``loss`` and ``accuracy``.
        """
        self.model.train()
        self._train_tracker.reset()
        step = 0

        with tqdm(
            self.train_loader,
            desc=f"  Epoch {self._current_epoch:3d} [train]",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        ) as pbar:
            for token_ids, mut_tensor, labels in pbar:
                token_ids  = token_ids.to(self.device)
                mut_tensor = mut_tensor.to(self.device)
                labels     = labels.to(self.device)

                # Forward
                logits, _ = self.model(token_ids, mut_tensor)
                loss       = self.criterion(logits, labels)

                # Backward
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.optimizer.step()

                # Accumulate metrics
                self._train_tracker.update(logits.detach(), labels, loss.item())
                step += 1

                if step % self.log_every_n_steps == 0:
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        acc=f"{self._train_tracker._n_correct / max(self._train_tracker._n_samples, 1):.4f}",
                    )

        metrics = self._train_tracker.compute()
        return {"loss": metrics["loss"], "accuracy": metrics["accuracy"]}

    def validate_epoch(
        self, loader: DataLoader
    ) -> Dict[str, float]:
        """
        One full pass over *loader* without gradient computation.

        Parameters
        ----------
        loader : DataLoader — Validation or test DataLoader.

        Returns
        -------
        dict with keys ``loss``, ``accuracy``, ``macro_f1``.
        """
        self.model.eval()
        self._val_tracker.reset()

        with torch.no_grad(), tqdm(
            loader,
            desc=f"  Epoch {self._current_epoch:3d} [val  ]",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        ) as pbar:
            for token_ids, mut_tensor, labels in pbar:
                token_ids  = token_ids.to(self.device)
                mut_tensor = mut_tensor.to(self.device)
                labels     = labels.to(self.device)

                logits, _ = self.model(token_ids, mut_tensor)
                loss       = self.criterion(logits, labels)

                self._val_tracker.update(logits, labels, loss.item())

        metrics = self._val_tracker.compute()
        return {
            "loss":     metrics["loss"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save_checkpoint(
        self, epoch: int, val_loss: float, is_best: bool
    ) -> None:
        """
        Persist model weights, optimiser state, and training metadata.

        Saves to ``checkpoints/checkpoint_epoch_{epoch:03d}.pt`` and,
        when *is_best* is ``True``, also to ``checkpoints/best_model.pt``.

        Parameters
        ----------
        epoch    : int   — Completed epoch number.
        val_loss : float — Validation loss at this epoch.
        is_best  : bool  — Whether this is the new best checkpoint.
        """
        payload = {
            "epoch":          epoch,
            "val_loss":       val_loss,
            "best_val_loss":  self._best_val_loss,
            "best_epoch":     self._best_epoch,
            "model_state":    self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }

        epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(payload, epoch_path)
        logger.debug("Checkpoint saved: %s", epoch_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy2(epoch_path, best_path)
            logger.info("Best checkpoint updated: %s", best_path)

    def load_checkpoint(self, path: Path) -> int:
        """
        Restore model weights and optimiser state from a checkpoint file.

        Parameters
        ----------
        path : Path — Path to a ``.pt`` checkpoint file.

        Returns
        -------
        int — The epoch number stored in the checkpoint.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])

        if "optimizer_state" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        if "scheduler_state" in payload:
            self.scheduler.load_state_dict(payload["scheduler_state"])

        epoch            = payload.get("epoch", 0)
        self._best_val_loss = payload.get("best_val_loss", float("inf"))
        self._best_epoch    = payload.get("best_epoch", epoch)

        logger.info(
            "Checkpoint loaded: epoch=%d  best_val_loss=%.6f  [%s]",
            epoch, self._best_val_loss, path,
        )
        return epoch

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self, loader: DataLoader) -> Dict:
        """
        Comprehensive evaluation on *loader* returning all metrics.

        All metrics are computed from first principles using
        :class:`MetricsTracker`.

        Parameters
        ----------
        loader : DataLoader — DataLoader to evaluate on.

        Returns
        -------
        dict with keys:
          ``accuracy``, ``macro_f1``, ``per_class_f1``,
          ``per_class_precision``, ``per_class_recall``,
          ``confusion_matrix``, ``loss``.
        """
        self.model.eval()
        tracker = MetricsTracker(NUM_CLASSES, CLASS_LABELS)

        with torch.no_grad(), tqdm(
            loader,
            desc="  Evaluating",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        ) as pbar:
            for token_ids, mut_tensor, labels in pbar:
                token_ids  = token_ids.to(self.device)
                mut_tensor = mut_tensor.to(self.device)
                labels     = labels.to(self.device)

                logits, _ = self.model(token_ids, mut_tensor)
                loss       = self.criterion(logits, labels)
                tracker.update(logits, labels, loss.item())

        metrics = tracker.compute()

        # Log per-class results
        logger.info("  %-22s  %6s  %6s  %6s", "Class", "Prec", "Rec", "F1")
        logger.info("  %s", "-" * 46)
        for name in CLASS_LABELS:
            logger.info(
                "  %-22s  %6.4f  %6.4f  %6.4f",
                name,
                metrics["per_class_precision"].get(name, 0.0),
                metrics["per_class_recall"].get(name, 0.0),
                metrics["per_class_f1"].get(name, 0.0),
            )
        logger.info(
            "  %-22s  %6.4f  %6.4f  %6.4f",
            "MACRO",
            metrics["macro_precision"],
            metrics["macro_recall"],
            metrics["macro_f1"],
        )

        return {
            "accuracy":            metrics["accuracy"],
            "loss":                metrics["loss"],
            "macro_f1":            metrics["macro_f1"],
            "per_class_f1":        metrics["per_class_f1"],
            "per_class_precision": metrics["per_class_precision"],
            "per_class_recall":    metrics["per_class_recall"],
            "confusion_matrix":    metrics["confusion_matrix"],
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_training_curves(self, history: Dict) -> None:
        """
        Save loss and accuracy training curves to
        ``checkpoints/training_curves.png``.

        Uses ``matplotlib`` when available; logs a warning and skips
        gracefully if it is not installed.

        Parameters
        ----------
        history : dict — Training history returned by :meth:`train`.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")   # non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning(
                "matplotlib not installed — training curves will not be saved. "
                "Install with: pip install matplotlib"
            )
            return

        epochs = list(range(1, len(history["train_loss"]) + 1))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Genomic Classifier — Training Curves", fontsize=14)

        # ---- Loss ----
        ax = axes[0]
        ax.plot(epochs, history["train_loss"], label="Train Loss",  color="#3B82F6")
        ax.plot(epochs, history["val_loss"],   label="Val Loss",    color="#EF4444")
        if history.get("best_epoch"):
            ax.axvline(
                x=history["best_epoch"], color="green",
                linestyle="--", alpha=0.7, label=f"Best (epoch {history['best_epoch']})",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ---- Accuracy ----
        ax = axes[1]
        ax.plot(epochs, history["train_accuracy"], label="Train Acc", color="#3B82F6")
        ax.plot(epochs, history["val_accuracy"],   label="Val Acc",   color="#EF4444")
        if history.get("val_f1"):
            ax.plot(epochs, history["val_f1"], label="Val F1",
                    color="#F59E0B", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Accuracy / F1")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        out_path = self.checkpoint_dir / "training_curves.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Training curves saved: %s", out_path)

    # ------------------------------------------------------------------
    # Resume helper (used by train.py)
    # ------------------------------------------------------------------

    def resume_from(self, checkpoint_path: Path) -> int:
        """
        Load a checkpoint and return the epoch to resume from.

        Parameters
        ----------
        checkpoint_path : Path — Path to a ``.pt`` checkpoint file.

        Returns
        -------
        int — The epoch stored in the checkpoint (training resumes from
              epoch + 1).
        """
        epoch = self.load_checkpoint(checkpoint_path)
        self._current_epoch = epoch
        logger.info("Resuming training from epoch %d.", epoch + 1)
        return epoch

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"GenomicTrainer("
            f"epochs={self.num_epochs}, "
            f"patience={self.patience}, "
            f"device={self.device})"
        )
