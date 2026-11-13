"""
Genomic_Framework/training/advanced_trainer.py
===============================================
Research-Grade Advanced Training, Optimization, and Validation Engine

Features:
  - Device Management: CUDA (Multi-GPU), Apple MPS, CPU auto-detection.
  - Reproducibility: Deterministic seeds across Python, NumPy, PyTorch, CUDNN.
  - AMP (Automatic Mixed Precision): Native PyTorch AMP scaling.
  - Optimization: AdamW, Adam, SGD with configurable weight decay & gradient clipping.
  - Schedulers: CosineAnnealing, ReduceLROnPlateau, OneCycleLR, Linear Warmup.
  - Comprehensive Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, Cohen's Kappa.
  - Automatic Model Selection: Best checkpoint selection by customizable metric.
  - TensorBoard & File Logging: Epoch and step-level logging to TensorBoard and disk.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Scikit-learn metrics for gold-standard research evaluation
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    SummaryWriter = None
    HAS_TENSORBOARD = False

from Genomic_Framework.pipeline.preprocessor import CLASS_LABELS, NUM_CLASSES

logger = logging.getLogger(__name__)


# ==============================================================================
# REPRODUCIBILITY UTILITY
# ==============================================================================

def set_seed(seed: int = 42, deterministic: bool = True, benchmark: bool = False) -> None:
    """Set random seeds across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = benchmark

    logger.info("Random seed locked to %d (deterministic=%s)", seed, deterministic)


# ==============================================================================
# DEVICE SELECTION UTILITY
# ==============================================================================

def detect_device(preferred: str = "auto") -> torch.device:
    """Auto-detect optimal compute device (CUDA, Apple MPS, CPU)."""
    pref = preferred.lower()
    if pref == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif pref == "cpu":
        device = torch.device("cpu")
    else:
        # Auto detection
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    logger.info("Target compute device detected: %s", device)
    if device.type == "cuda":
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("CUDA GPU Available: %s (%d GPU device(s))", gpu_name, gpu_count)

    return device


# ==============================================================================
# COMPREHENSIVE METRICS CALCULATOR
# ==============================================================================

class AdvancedMetricsTracker:
    """
    Computes research-grade evaluation metrics including:
    Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, and Cohen's Kappa.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, class_labels: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_labels = class_labels or CLASS_LABELS
        self.reset()

    def reset(self) -> None:
        self.y_true: List[int] = []
        self.y_pred: List[int] = []
        self.y_probs: List[List[float]] = []
        self.total_loss: float = 0.0
        self.total_samples: int = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: float) -> None:
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        lbls = labels.detach().cpu().numpy()

        self.y_true.extend(lbls.tolist())
        self.y_pred.extend(preds.tolist())
        self.y_probs.extend(probs.tolist())
        self.total_loss += loss * len(lbls)
        self.total_samples += len(lbls)

    def compute(self, compute_auc: bool = True) -> Dict[str, Any]:
        if self.total_samples == 0:
            return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0}

        avg_loss = self.total_loss / self.total_samples
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        y_probs = np.array(self.y_probs)

        acc = accuracy_score(y_true, y_pred)
        macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        # FIX: Add weighted F1 and micro F1 (critical for imbalanced multi-class)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        micro_f1    = f1_score(y_true, y_pred, average="micro",    zero_division=0)
        # FIX: Add balanced accuracy (accounts for class imbalance)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        # Per-class metrics
        per_class_p = precision_score(y_true, y_pred, average=None, labels=list(range(self.num_classes)), zero_division=0)
        per_class_r = recall_score(y_true, y_pred, average=None, labels=list(range(self.num_classes)), zero_division=0)
        per_class_f = f1_score(y_true, y_pred, average=None, labels=list(range(self.num_classes)), zero_division=0)

        per_p_dict = {self.class_labels[i]: float(per_class_p[i]) for i in range(self.num_classes)}
        per_r_dict = {self.class_labels[i]: float(per_class_r[i]) for i in range(self.num_classes)}
        per_f_dict = {self.class_labels[i]: float(per_class_f[i]) for i in range(self.num_classes)}

        mcc = matthews_corrcoef(y_true, y_pred)
        # FIX: Guard cohen_kappa for single-class edge case (crashes during early training)
        try:
            kappa = cohen_kappa_score(y_true, y_pred)
        except ValueError:
            kappa = 0.0
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes))).tolist()

        # ROC-AUC & PR-AUC computation
        roc_auc_macro = 0.0
        pr_auc_macro = 0.0
        roc_auc_per_class: Dict[str, float] = {}
        pr_auc_per_class: Dict[str, float] = {}

        if compute_auc and len(np.unique(y_true)) > 1:
            try:
                # One-vs-Rest binarized targets
                y_true_onehot = np.eye(self.num_classes)[y_true]
                
                # Multi-class ROC-AUC
                roc_auc_macro = float(roc_auc_score(y_true_onehot, y_probs, average="macro", multi_class="ovr"))
                roc_auc_class = roc_auc_score(y_true_onehot, y_probs, average=None, multi_class="ovr")
                for i in range(self.num_classes):
                    roc_auc_per_class[self.class_labels[i]] = float(roc_auc_class[i])
                
                # Multi-class PR-AUC
                pr_aucs = []
                for i in range(self.num_classes):
                    p_curve, r_curve, _ = precision_recall_curve(y_true_onehot[:, i], y_probs[:, i])
                    pr_auc_val = float(auc(r_curve, p_curve))
                    pr_auc_per_class[self.class_labels[i]] = pr_auc_val
                    pr_aucs.append(pr_auc_val)
                pr_auc_macro = float(np.mean(pr_aucs))
            except Exception as exc:
                logger.warning("AUC computation encountered numerical edge-case: %s", exc)

        return {
            "loss": float(avg_loss),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),   # FIX: added
            "macro_precision": float(macro_prec),
            "macro_recall": float(macro_rec),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),      # FIX: added
            "micro_f1": float(micro_f1),            # FIX: added
            "mcc": float(mcc),
            "kappa": float(kappa),
            "roc_auc_macro": float(roc_auc_macro),
            "pr_auc_macro": float(pr_auc_macro),
            "per_class_precision": per_p_dict,
            "per_class_recall": per_r_dict,
            "per_class_f1": per_f_dict,
            "per_class_roc_auc": roc_auc_per_class,
            "per_class_pr_auc": pr_auc_per_class,
            "confusion_matrix": cm,
        }


# ==============================================================================
# FOCAL LOSS FOR CLASS IMBALANCE MITIGATION
# ==============================================================================

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with dynamic alpha weighting and focusing gamma.
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Suppresses easy majority-class gradients and forces high gradient updates
    on hard/minority classes (such as Avirulent/Benign host genomes).
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        unweighted_ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-unweighted_ce)
        focal_term = ((1.0 - pt) ** self.gamma) * unweighted_ce
        if self.weight is not None:
            alpha_t = self.weight[targets]
            focal_loss = alpha_t * focal_term
        else:
            focal_loss = focal_term

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ==============================================================================
# ADVANCED RESEARCH TRAINER
# ==============================================================================

class AdvancedTrainer:
    """
    Research-Grade Training & Optimization Engine.

    Integrates:
      - Optimizer (AdamW/Adam/SGD)
      - Schedulers (CosineAnnealing, ReduceLROnPlateau, OneCycleLR, Warmup)
      - Automatic Mixed Precision (AMP) via GradScaler
      - Gradient Clipping & Norm Monitoring
      - Gradient Accumulation
      - TensorBoard Event Logging
      - Best Checkpoint Selection by Configurable Metric
      - Early Stopping & Min-Epoch Controls
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        class_weights: Optional[torch.FloatTensor] = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # System & Device
        sys_cfg = config.get("system", {})
        set_seed(
            seed=sys_cfg.get("seed", 42),
            deterministic=sys_cfg.get("deterministic", True),
            benchmark=sys_cfg.get("benchmark", False),
        )
        self.device = detect_device(sys_cfg.get("device", "auto"))

        # Multi-GPU Wrap
        self.raw_model = model
        if self.device.type == "cuda" and sys_cfg.get("multi_gpu", True) and torch.cuda.device_count() > 1:
            logger.info("Multi-GPU enabled: wrapping model in DataParallel across %d GPUs", torch.cuda.device_count())
            self.model = nn.DataParallel(model).to(self.device)
        else:
            self.model = model.to(self.device)

        # Loss Function (Supports Focal Loss for high-consequence class imbalance mitigation)
        cw = class_weights.to(self.device) if class_weights is not None else None
        loss_type = config.get("optimization", {}).get("loss_type", "focal").lower()
        if loss_type == "focal":
            focal_gamma = float(config.get("optimization", {}).get("focal_gamma", 2.0))
            self.criterion = FocalLoss(weight=cw, gamma=focal_gamma)
            logger.info("Using Focal Loss with gamma=%.1f and dynamic class weights for imbalance mitigation", focal_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=cw)
            logger.info("Using Weighted Cross Entropy Loss")

        # Optimization & AMP Parameters
        opt_cfg = config.get("optimization", {})
        self.lr = float(opt_cfg.get("learning_rate", 1e-4))
        self.weight_decay = float(opt_cfg.get("weight_decay", 0.01))
        self.grad_clip = float(opt_cfg.get("gradient_clipping", 1.0))
        self.accum_steps = int(opt_cfg.get("gradient_accumulation_steps", 1))
        self.use_amp = bool(opt_cfg.get("use_amp", True)) and (self.device.type == "cuda")
        # FIX: store device type string for autocast (was hardcoded 'cuda')
        self._amp_device_type = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"

        # PyTorch AMP Scaler — use device type dynamically
        self.scaler = torch.amp.GradScaler(self._amp_device_type, enabled=self.use_amp)
        if self.use_amp:
            logger.info("Automatic Mixed Precision (AMP) enabled for CUDA training")

        # Build Optimizer
        opt_name = opt_cfg.get("optimizer", "adamw").lower()
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        eps = float(opt_cfg.get("eps", 1e-8))

        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=betas, eps=eps
            )
        elif opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=betas, eps=eps
            )
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        # Epoch & Early Stopping Controls
        tr_cfg = config.get("training", {})
        self.max_epochs = int(tr_cfg.get("max_epochs", 50))
        self.min_epochs = int(tr_cfg.get("min_epochs", 25))
        self.selection_metric = tr_cfg.get("model_selection_metric", "val_f1")
        self.log_every_n_steps = int(tr_cfg.get("log_every_n_steps", 10))

        es_cfg = tr_cfg.get("early_stopping", {})
        self.es_enabled = bool(es_cfg.get("enabled", True))
        self.es_patience = int(es_cfg.get("patience", 15))
        self.es_min_delta = float(es_cfg.get("min_delta", 1e-4))

        # Scheduler Setup
        sched_cfg = config.get("scheduler", {})
        sched_name = sched_cfg.get("name", "cosine").lower()
        self.sched_type = sched_name
        # FIX: Linear warmup epochs (prevents cold-start LR spike)
        self.warmup_epochs = int(sched_cfg.get("warmup_epochs", 3))

        if sched_name == "cosine":
            # FIX: Use CosineAnnealingWarmRestarts for multi-cycle annealing
            #      or wrap with SequentialLR for warmup + cosine
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.max_epochs - self.warmup_epochs),
                eta_min=float(sched_cfg.get("min_lr", 1e-6)),
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[self.warmup_epochs],
            )
            logger.info("Scheduler: Linear Warmup (%d epochs) + CosineAnnealing", self.warmup_epochs)
        elif sched_name == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min" if "loss" in self.selection_metric else "max",
                factor=float(sched_cfg.get("plateau_factor", 0.5)),
                patience=int(sched_cfg.get("plateau_patience", 5)),
            )
        elif sched_name == "onecycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.lr * 10,
                total_steps=self.max_epochs * len(self.train_loader),
            )
        else:
            self.scheduler = None

        # Output Directories
        paths = config.get("paths", {})
        self.output_dir = Path(paths.get("output_dir", "output_advanced"))
        self.checkpoint_dir = self.output_dir / paths.get("checkpoint_dir", "checkpoints")
        self.log_dir = self.output_dir / paths.get("log_dir", "logs")
        self.plot_dir = self.output_dir / paths.get("plot_dir", "plots")
        self.metric_dir = self.output_dir / paths.get("metric_dir", "metrics")
        self.tb_dir = self.output_dir / paths.get("tensorboard_dir", "tensorboard")

        for d in [self.checkpoint_dir, self.log_dir, self.plot_dir, self.metric_dir, self.tb_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # TensorBoard SummaryWriter
        self.tb_writer = SummaryWriter(log_dir=str(self.tb_dir)) if HAS_TENSORBOARD else None

        # State Variables
        self.best_metric_value = -float("inf") if "loss" not in self.selection_metric else float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.current_epoch = 0

        # Metrics trackers
        self.train_tracker = AdvancedMetricsTracker(NUM_CLASSES, CLASS_LABELS)
        self.val_tracker = AdvancedMetricsTracker(NUM_CLASSES, CLASS_LABELS)

    # --------------------------------------------------------------------------
    # Main Training Loop
    # --------------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Execute complete multi-epoch training and validation pipeline."""
        start_time = time.time()
        logger.info("=" * 72)
        logger.info("Starting Advanced Training Loop — Max Epochs: %d | Selection Metric: %s", self.max_epochs, self.selection_metric)
        logger.info("=" * 72)

        history: Dict[str, List[Any]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_mcc": [],
            "val_roc_auc": [],
            "learning_rate": [],
            "grad_norm": [],
            "epoch_duration": [],
        }

        for epoch in range(1, self.max_epochs + 1):
            self.current_epoch = epoch
            ep_start = time.time()

            # Train epoch
            train_loss, train_acc, grad_norm = self._train_one_epoch()
            
            # Validate epoch
            val_metrics = self._validate_one_epoch(self.val_loader)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["macro_f1"]
            val_mcc = val_metrics["mcc"]
            val_roc_auc = val_metrics["roc_auc_macro"]

            ep_duration = time.time() - ep_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Scheduler step
            if self.scheduler:
                if self.sched_type == "plateau":
                    self.scheduler.step(val_loss if "loss" in self.selection_metric else val_f1)
                elif self.sched_type != "onecycle":
                    self.scheduler.step()

            # Record metrics history
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            history["val_f1"].append(val_f1)
            history["val_mcc"].append(val_mcc)
            history["val_roc_auc"].append(val_roc_auc)
            history["learning_rate"].append(current_lr)
            history["grad_norm"].append(grad_norm)
            history["epoch_duration"].append(ep_duration)

            # TensorBoard logging
            if self.tb_writer:
                self.tb_writer.add_scalar("Loss/Train", train_loss, epoch)
                self.tb_writer.add_scalar("Loss/Val", val_loss, epoch)
                self.tb_writer.add_scalar("Accuracy/Train", train_acc, epoch)
                self.tb_writer.add_scalar("Accuracy/Val", val_acc, epoch)
                self.tb_writer.add_scalar("F1/Val_Macro", val_f1, epoch)
                self.tb_writer.add_scalar("MCC/Val", val_mcc, epoch)
                self.tb_writer.add_scalar("LearningRate", current_lr, epoch)

            val_bal_acc = val_metrics.get("balanced_accuracy", 0.0)
            val_wf1     = val_metrics.get("weighted_f1", 0.0)
            logger.info(
                "Epoch %3d/%d [%.1fs] | Train Loss: %.4f | Train Acc: %.4f | "
                "Val Loss: %.4f | Val Acc: %.4f | Val BalAcc: %.4f | "
                "Val MacF1: %.4f | Val WtdF1: %.4f | LR: %.2e",
                epoch, self.max_epochs, ep_duration,
                train_loss, train_acc,
                val_loss, val_acc, val_bal_acc,
                val_f1, val_wf1, current_lr,
            )

            # Model Selection check
            current_metric_val = self._extract_selection_metric(val_metrics)
            is_best = self._is_better(current_metric_val)

            if is_best:
                self.best_metric_value = current_metric_val
                self.best_epoch = epoch
                self.patience_counter = 0
                logger.info("  ✓ New Best Model Identified (Epoch %d | %s = %.4f)", epoch, self.selection_metric, current_metric_val)
            else:
                self.patience_counter += 1

            # Save checkpoints
            self._save_checkpoint(epoch, val_metrics, is_best)

            # Early stopping check
            if self.es_enabled and epoch >= self.min_epochs and self.patience_counter >= self.es_patience:
                logger.info("Early stopping triggered at Epoch %d (Patience of %d epochs reached)", epoch, self.es_patience)
                break

        total_duration = time.time() - start_time
        logger.info("Training finished in %.2f seconds. Best Epoch: %d (%s = %.4f)", total_duration, self.best_epoch, self.selection_metric, self.best_metric_value)

        # Plot curves
        self._plot_training_curves(history)

        # Load best model for evaluation
        best_pt = self.checkpoint_dir / "best_model.pt"
        if best_pt.exists():
            self.load_checkpoint(best_pt)

        # Final Test Evaluation
        logger.info("Evaluating Best Model on Held-Out Test Set ...")
        test_metrics = self._validate_one_epoch(self.test_loader)
        history["test_metrics"] = test_metrics
        history["best_epoch"] = self.best_epoch
        history["total_duration"] = total_duration

        if self.tb_writer:
            self.tb_writer.close()

        return history

    # --------------------------------------------------------------------------
    # Single Epoch Handlers
    # --------------------------------------------------------------------------

    def _train_one_epoch(self) -> Tuple[float, float, float]:
        self.model.train()
        self.train_tracker.reset()
        grad_norms = []

        self.optimizer.zero_grad(set_to_none=True)

        for step, (token_ids, mut_tensor, labels) in enumerate(self.train_loader, 1):
            token_ids = token_ids.to(self.device, non_blocking=True)
            mut_tensor = mut_tensor.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # FIX: Use dynamic device type string — was hardcoded 'cuda'
            with torch.amp.autocast(self._amp_device_type, enabled=self.use_amp):
                logits, _ = self.model(token_ids, mut_tensor)
                loss = self.criterion(logits, labels) / self.accum_steps

            # FIX: NaN loss detection — skip step to prevent gradient corruption
            if not torch.isfinite(loss):
                logger.critical(
                    "[ERROR] NaN/Inf loss detected at step %d epoch %d — skipping step!",
                    step, self.current_epoch,
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # AMP Scaled Backward
            self.scaler.scale(loss).backward()

            if step % self.accum_steps == 0 or step == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip))
                grad_norms.append(norm)

                # FIX: Gradient explosion alarm
                if norm > 100.0:
                    logger.critical(
                        "[ERROR] Gradient explosion detected! grad_norm=%.2f at step %d epoch %d.",
                        norm, step, self.current_epoch,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler and self.sched_type == "onecycle":
                    self.scheduler.step()

            # Record batch metrics
            self.train_tracker.update(logits.detach(), labels, loss.item() * self.accum_steps)

            if step % self.log_every_n_steps == 0 or step == len(self.train_loader):
                cur_norm = grad_norms[-1] if grad_norms else 0.0
                logger.info(
                    "  [Epoch %d/%d | Step %d/%d] Batch Loss: %.4f | Grad Norm: %.3f",
                    self.current_epoch, self.max_epochs, step, len(self.train_loader),
                    loss.item() * self.accum_steps, cur_norm,
                )

        metrics = self.train_tracker.compute(compute_auc=False)
        avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
        return metrics["loss"], metrics["accuracy"], avg_grad_norm

    def _validate_one_epoch(self, loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        self.val_tracker.reset()

        with torch.no_grad():
            for token_ids, mut_tensor, labels in loader:
                token_ids = token_ids.to(self.device, non_blocking=True)
                mut_tensor = mut_tensor.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # FIX: Use dynamic device type string — was hardcoded 'cuda'
                with torch.amp.autocast(self._amp_device_type, enabled=self.use_amp):
                    logits, _ = self.model(token_ids, mut_tensor)
                    loss = self.criterion(logits, labels)

                self.val_tracker.update(logits, labels, loss.item())

        return self.val_tracker.compute(compute_auc=self.config.get("evaluation", {}).get("compute_auc", True))

    # --------------------------------------------------------------------------
    # Checkpoint Management
    # --------------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_metrics: Dict[str, Any], is_best: bool) -> None:
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        state = {
            "epoch": epoch,
            "model_state": model_to_save.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict() if self.use_amp else None,
            "val_metrics": val_metrics,
            "config": self.config,
        }

        # Latest model
        latest_pt = self.checkpoint_dir / "latest_model.pt"
        torch.save(state, latest_pt)

        # Periodic checkpoint
        interval = int(self.config.get("training", {}).get("checkpoint_interval", 5))
        if epoch % interval == 0:
            torch.save(state, self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt")

        # Best model
        if is_best:
            best_pt = self.checkpoint_dir / "best_model.pt"
            shutil.copy2(latest_pt, best_pt)

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        logger.info("Loading checkpoint from %s ...", checkpoint_path)
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_to_load = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model_to_load.load_state_dict(state["model_state"])

        if "optimizer_state" in state and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(state["optimizer_state"])
        if "scheduler_state" in state and self.scheduler and state["scheduler_state"]:
            self.scheduler.load_state_dict(state["scheduler_state"])
        if "scaler_state" in state and self.use_amp and state["scaler_state"]:
            self.scaler.load_state_dict(state["scaler_state"])

        epoch = state.get("epoch", 0)
        logger.info("Successfully loaded checkpoint (Epoch %d)", epoch)
        return epoch

    # --------------------------------------------------------------------------
    # Helper Utilities
    # --------------------------------------------------------------------------

    def _extract_selection_metric(self, metrics: Dict[str, Any]) -> float:
        if self.selection_metric == "val_loss":
            return float(metrics["loss"])
        elif self.selection_metric == "val_accuracy":
            return float(metrics["accuracy"])
        elif self.selection_metric == "val_f1":
            return float(metrics["macro_f1"])
        elif self.selection_metric == "val_mcc":
            return float(metrics["mcc"])
        return float(metrics.get(self.selection_metric, metrics["macro_f1"]))

    def _is_better(self, current: float) -> bool:
        if "loss" in self.selection_metric:
            return current < (self.best_metric_value - self.es_min_delta)
        else:
            return current > (self.best_metric_value + self.es_min_delta)

    def _plot_training_curves(self, history: Dict[str, Any]) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs = list(range(1, len(history["train_loss"]) + 1))
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle("Advanced Training & Optimization Curves", fontsize=16, fontweight="bold")

            # Loss Curve
            ax = axes[0, 0]
            ax.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
            ax.plot(epochs, history["val_loss"], label="Val Loss", color="#d62728", lw=2)
            if self.best_epoch > 0:
                ax.axvline(self.best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best Epoch ({self.best_epoch})")
            ax.set_title("Loss Trajectory")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Accuracy & F1 Curve
            ax = axes[0, 1]
            ax.plot(epochs, history["train_accuracy"], label="Train Acc", color="#1f77b4", lw=2)
            ax.plot(epochs, history["val_accuracy"], label="Val Acc", color="#2ca02c", lw=2)
            ax.plot(epochs, history["val_f1"], label="Val Macro F1", color="#ff7f0e", lw=2, linestyle="--")
            ax.set_title("Accuracy & F1 Trajectory")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Learning Rate Curve
            ax = axes[1, 0]
            ax.plot(epochs, history["learning_rate"], label="Learning Rate", color="#9467bd", lw=2)
            ax.set_title("Learning Rate Schedule")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("LR")
            ax.set_yscale("log")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Gradient Norm Curve
            ax = axes[1, 1]
            ax.plot(epochs, history["grad_norm"], label="Grad Norm", color="#8c564b", lw=2)
            ax.set_title("Gradient Norm Trajectory")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Norm")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = self.plot_dir / "training_optimization_curves.png"
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved training optimization curves plot to %s", plot_path)
        except Exception as exc:
            logger.warning("Failed to render training curves plot: %s", exc)
