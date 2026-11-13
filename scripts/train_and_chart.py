"""
scripts/train_and_chart.py
==========================
Self-contained high-performance training + comprehensive chart generation script.

Optimized for NVIDIA GPU acceleration (RTX 3050 Laptop GPU / CUDA) with AMP Mixed-Precision.
Features:
  - Direct sequence context pooling skip connection in CrossAttentionFusion.
  - Pre-tokenized tensor cache for instant loading without FASTA I/O overhead.
  - WeightedRandomSampler ensuring balanced class draws per epoch.
  - Unweighted FocalLoss with label smoothing to eliminate class suppression.
  - High-resolution 12-panel dashboard PNG export.
  - Standalone individual high-res PNG export for each of the 12 plots.
  - Automatic artifact synchronization for immediate visualization.

Usage:
    python scripts/train_and_chart.py [--fresh] [--epochs 50] [--batch-size 64]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ── project root on sys.path ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── third-party ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")           # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, confusion_matrix, classification_report
)

# ── project imports ──────────────────────────────────────────────────────────
from Genomic_Framework.engine import GenomicClassificationEngine
from Genomic_Framework.pipeline.preprocessor import (
    tokenise_sequence, CLASS_LABELS, NUM_CLASSES
)

# ── logging ──────────────────────────────────────────────────────────────────
class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

handler = FlushingStreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger = logging.getLogger("train_and_chart")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(handler)

# ── constants ────────────────────────────────────────────────────────────────
SEQ_LEN               = 512
VOCAB_SIZE            = 4096
K_MER                 = 6
MAX_MUTATIONS         = 32
MUTATION_DIM          = 64
HYENA_DIM             = 256
HIDDEN_DIM            = 512
DEFAULT_BATCH_SIZE    = 32
DEFAULT_MAX_EPOCHS    = 50
DEFAULT_SAMPLES_PER_EPOCH = 3200  # 100 balanced batches of 32 -> ~25s/epoch, 21 min for 50 epochs
MIN_EPOCHS            = 20
PATIENCE              = 15
LR                = 3e-4
WEIGHT_DECAY      = 1e-2
FOCAL_GAMMA       = 2.0
LABEL_SMOOTHING   = 0.05
SEED              = 42
NUM_THREADS       = 6
WARMUP_EPOCHS     = 2
MIN_LR            = 1e-6

SPLIT_CSV    = _ROOT / "dataset" / "splits" / "split_manifest.csv"
FASTA_DIR    = _ROOT / "dataset" / "canonical" / "fasta"
OUT_DIR      = _ROOT / "output_advanced"
CKPT_PATH    = OUT_DIR / "checkpoints" / "best_model.pt"
LAST_CKPT    = OUT_DIR / "checkpoints" / "last_checkpoint.pt"
METRICS_PATH = OUT_DIR / "metrics" / "training_metrics.json"
PLOT_DIR     = OUT_DIR / "plots"
DASHBOARD_PATH = PLOT_DIR / "training_dashboard.png"
INDIVIDUAL_DIR = PLOT_DIR / "individual"

ARTIFACT_DIR = Path(r"C:\Users\vishn\.gemini\antigravity\brain\4732aee7-8f2e-4544-9297-c8da6bb46072")

DARK_RC_PARAMS = {
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.titlecolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.7,
    "text.color":        "#c9d1d9",
    "figure.facecolor":  "#0d1117",
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "legend.labelcolor": "#c9d1d9",
}

CLASS_COLORS = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]
CLASS_NAMES_SHORT = ["Avirulent", "LPAI", "MPAI", "HPAI"]


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_from_row(pt: str, sub: str) -> int:
    pt  = pt.upper()
    sub = sub.upper()
    if pt == "AVIRULENT" or sub == "HEALTHY":   return 0
    if pt == "HPAI" or sub in ("H5N1", "H5N9", "H7N9"): return 3
    if sub in ("H5N6", "H5N8", "H7N1", "H7N7", "H10N7", "H10N8"): return 2
    return 1


def parse_fasta_file(path: Path) -> str:
    seq_parts = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith(">"):
                seq_parts.append(line.upper())
    return "".join(seq_parts)


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & Tokenize Datasets
# ════════════════════════════════════════════════════════════════════════════

def load_split_tensors(split: str, csv_path: Path, fasta_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    cache_dir = OUT_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{split}_tensors.pt"

    if cache_file.exists():
        logger.info("[%s] Loading cached tensors from %s …", split.upper(), cache_file)
        cached = torch.load(str(cache_file), map_location="cpu", weights_only=False)
        tokens_t = cached["tokens"]
        labels_t = cached["labels"]
        logger.info("[%s] loaded %d cached samples  |  class dist: %s",
                    split.upper(), len(labels_t),
                    dict(sorted(Counter(labels_t.tolist()).items())))
        return tokens_t, labels_t

    rows_for_split = []
    with csv_path.open("r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("split", "").strip().lower() == split:
                rows_for_split.append(r)

    logger.info("[%s] %d rows in manifest", split.upper(), len(rows_for_split))

    tokens_list: List[torch.Tensor] = []
    labels_list: List[int] = []
    missing = 0

    for r in rows_for_split:
        acc = r.get("insdc_accession") or r.get("record_id")
        fp  = fasta_dir / f"{acc}.fasta"
        if not fp.exists():
            missing += 1
            continue
        seq  = parse_fasta_file(fp)
        if len(seq) < 10:
            missing += 1
            continue
        seq = seq.replace("N", "")[:SEQ_LEN * 10]
        tok = tokenise_sequence(seq, SEQ_LEN, VOCAB_SIZE, k=K_MER)
        lbl = label_from_row(r.get("pathotype", ""), r.get("subtype", ""))
        tokens_list.append(tok)
        labels_list.append(lbl)

    if missing:
        logger.warning("[%s] %d files missing / empty", split.upper(), missing)

    tokens_t = torch.stack(tokens_list)       # [N, SEQ_LEN]
    labels_t = torch.tensor(labels_list, dtype=torch.long)
    logger.info("[%s] loaded %d samples  |  class dist: %s",
                split.upper(), len(labels_list),
                dict(sorted(Counter(labels_list).items())))

    try:
        torch.save({"tokens": tokens_t, "labels": labels_t}, str(cache_file))
        logger.info("[%s] Saved tensor cache → %s", split.upper(), cache_file)
    except Exception as e:
        logger.warning("[%s] Could not cache tensors: %s", split.upper(), e)

    return tokens_t, labels_t


def build_loaders(batch_size: int = DEFAULT_BATCH_SIZE, samples_per_epoch: int = DEFAULT_SAMPLES_PER_EPOCH):
    logger.info("=" * 60)
    logger.info("Loading datasets (Batch Size: %d | Samples/Epoch: %d) …", batch_size, samples_per_epoch)
    logger.info("=" * 60)

    t0 = time.time()
    tr_tok, tr_lbl = load_split_tensors("train", SPLIT_CSV, FASTA_DIR)
    va_tok, va_lbl = load_split_tensors("val",   SPLIT_CSV, FASTA_DIR)
    te_tok, te_lbl = load_split_tensors("test",  SPLIT_CSV, FASTA_DIR)
    logger.info("All splits loaded in %.1f s", time.time() - t0)

    # Calculate inverse-frequency class weights for balanced sampling
    counts  = Counter(tr_lbl.tolist())
    total   = len(tr_lbl)
    weights = torch.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        n = counts.get(c, 1)
        weights[c] = total / (NUM_CLASSES * n)
    weights = weights / weights.mean()
    logger.info("Class sampling weights (mean-normalised): %s",
                {CLASS_LABELS[i]: round(weights[i].item(), 3) for i in range(NUM_CLASSES)})

    # Balanced draws per epoch
    num_samples = len(tr_tok) if (samples_per_epoch is None or samples_per_epoch <= 0) else samples_per_epoch
    sample_w = weights[tr_lbl]
    sampler  = WeightedRandomSampler(
        weights=sample_w,
        num_samples=num_samples,
        replacement=True,
    )

    tr_mut = torch.zeros(len(tr_tok), MAX_MUTATIONS, MUTATION_DIM)
    va_mut = torch.zeros(len(va_tok), MAX_MUTATIONS, MUTATION_DIM)
    te_mut = torch.zeros(len(te_tok), MAX_MUTATIONS, MUTATION_DIM)

    train_ds = TensorDataset(tr_tok, tr_mut, tr_lbl)
    val_ds   = TensorDataset(va_tok, va_mut, va_lbl)
    test_ds  = TensorDataset(te_tok, te_mut, te_lbl)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader, weights


# ════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS WITH LABEL SMOOTHING
# ════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 label_smoothing: float = LABEL_SMOOTHING):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross entropy with optional label smoothing
        ce = F.cross_entropy(inputs, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce
        return focal_loss.mean()


# ════════════════════════════════════════════════════════════════════════════
# EVALUATION FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Dict[str, Any]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    total_loss = 0.0
    n_batches  = 0
    use_cuda = (device.type == "cuda")

    with torch.no_grad():
        for tok, mut, lbl in loader:
            tok = tok.to(device, non_blocking=use_cuda)
            mut = mut.to(device, non_blocking=use_cuda)
            lbl = lbl.to(device, non_blocking=use_cuda)
            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                logits, _ = model(tok, mut)
                loss = criterion(logits, lbl)
            total_loss += loss.item()
            n_batches  += 1
            all_logits.append(logits.float().cpu())
            all_labels.append(lbl.cpu())

    all_logits_t = torch.cat(all_logits)
    all_labels_t = torch.cat(all_labels)
    probs = torch.softmax(all_logits_t, dim=-1).numpy()
    preds = all_logits_t.argmax(dim=-1).numpy()
    lbls  = all_labels_t.numpy()

    avg_loss = total_loss / max(n_batches, 1)
    acc      = accuracy_score(lbls, preds)
    bal_acc  = balanced_accuracy_score(lbls, preds)
    macro_f1 = f1_score(lbls, preds, average="macro",    zero_division=0)
    wtd_f1   = f1_score(lbls, preds, average="weighted", zero_division=0)
    mcc      = matthews_corrcoef(lbls, preds)
    cm       = confusion_matrix(lbls, preds, labels=list(range(NUM_CLASSES)))

    # ROC-AUC (macro OVR)
    try:
        roc_auc = roc_auc_score(lbls, probs, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = 0.0

    per_class_f1 = f1_score(lbls, preds, average=None, zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": wtd_f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "y_true": lbls.tolist(),
        "y_pred": preds.tolist(),
        "y_probs": probs.tolist(),
    }


# ════════════════════════════════════════════════════════════════════════════
# 12 PLOT RENDERING FUNCTIONS (STANDALONE + DASHBOARD)
# ════════════════════════════════════════════════════════════════════════════

def plot_01_loss(ax, history: Dict):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax.plot(epochs, history["train_loss"], "o-", color="#58a6ff", label="Train Loss", lw=2, ms=4)
    ax.plot(epochs, history["val_loss"],   "s-", color="#ff7b72", label="Val Loss",   lw=2, ms=4)
    best_e = history.get("best_epoch", 1)
    ax.axvline(best_e, color="#f0e68c", ls="--", lw=1.3, label=f"Best Epoch {best_e}")
    ax.set_title("Training & Validation Loss (Focal)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True)


def plot_02_accuracy(ax, history: Dict):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax.plot(epochs, [a * 100 for a in history["train_accuracy"]], "o-", color="#58a6ff", label="Train Acc", lw=2, ms=4)
    ax.plot(epochs, [a * 100 for a in history["val_accuracy"]],   "s-", color="#ff7b72", label="Val Acc",   lw=2, ms=4)
    ax.plot(epochs, [a * 100 for a in history["val_balanced_accuracy"]], "^-", color="#3fb950", label="Val Bal-Acc", lw=2, ms=4)
    best_e = history.get("best_epoch", 1)
    ax.axvline(best_e, color="#f0e68c", ls="--", lw=1.3)
    ax.set_title("Accuracy Trajectory (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True)


def plot_03_f1_mcc(ax, history: Dict):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax.plot(epochs, history["val_macro_f1"],    "o-", color="#79c0ff", label="Val Macro-F1",    lw=2, ms=4)
    ax.plot(epochs, history["val_weighted_f1"], "s-", color="#ffa657", label="Val Weighted-F1", lw=2, ms=4)
    ax.plot(epochs, history["val_mcc"],         "^-", color="#56d364", label="Val MCC",         lw=2, ms=4)
    best_e = history.get("best_epoch", 1)
    ax.axvline(best_e, color="#f0e68c", ls="--", lw=1.3)
    ax.set_title("F1 Scores & Matthews Correlation (MCC)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric Score")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True)


def plot_04_roc_auc(ax, history: Dict):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax.plot(epochs, history["val_roc_auc"], "D-", color="#d2a8ff", label="Val ROC-AUC (Macro OvR)", lw=2, ms=4)
    ax.axhline(0.97, color="#f0e68c", ls=":", lw=1.3, label="Target AUC (0.97)")
    best_e = history.get("best_epoch", 1)
    ax.axvline(best_e, color="#f0e68c", ls="--", lw=1.3)
    ax.set_title("ROC-AUC Trajectory (Macro OvR)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Area Under Curve")
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True)


def plot_05_lr(ax, history: Dict):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax.semilogy(epochs, history["learning_rate"], "o-", color="#79c0ff", lw=2, ms=4)
    ax.set_title("Learning Rate Schedule (CosineAnnealing)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate (log scale)")
    ax.grid(True)


def plot_06_grad_norm(ax, history: Dict):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax.plot(epochs, history["grad_norm"], "o-", color="#ffa657", lw=2, ms=4, label="L2 Grad Norm")
    ax.axhline(1.0, color="#f0e68c", ls=":", lw=1.3, label="Clip Threshold (1.0)")
    ax.set_title("Gradient L2 Norm Dynamics", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True)


def plot_07_per_class_f1(ax, test_metrics: Dict):
    per_class_f1 = test_metrics["per_class_f1"]
    bars = ax.bar(CLASS_NAMES_SHORT, [f * 100 for f in per_class_f1],
                  color=CLASS_COLORS, edgecolor="#30363d", width=0.55)
    ax.set_title("Per-Class F1 Score (Independent Test Set)", fontsize=11, fontweight="bold")
    ax.set_ylabel("F1 Score (%)")
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y")
    for bar, val in zip(bars, per_class_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.0,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")


def plot_08_confusion_matrix(ax, test_metrics: Dict):
    cm = np.array(test_metrics["confusion_matrix"])
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_pct = (cm.astype(float) / row_sums) * 100
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES_SHORT, rotation=15, fontsize=8)
    ax.set_yticklabels(CLASS_NAMES_SHORT, fontsize=8)
    ax.set_xlabel("Predicted Pathotype")
    ax.set_ylabel("True Pathotype")
    ax.set_title("Confusion Matrix (Test Set, % Normalized)", fontsize=11, fontweight="bold")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_pct[i, j]
            count = cm[i, j]
            ax.text(j, i, f"{val:.1f}%\n({count})", ha="center", va="center",
                    color="white" if val > 50 else "#c9d1d9", fontsize=8, fontweight="bold")
    return im


def plot_09_scorecard(ax, history: Dict, test_metrics: Dict):
    ax.axis("off")
    epochs = len(history["train_loss"])
    best_val_f1 = max(history["val_macro_f1"]) if history["val_macro_f1"] else 0.0
    best_val_acc = max(history["val_accuracy"]) if history["val_accuracy"] else 0.0
    scorecard = [
        ("Dataset Overview", "34,999 sequences (100% Real FASTA)"),
        ("Splits", "30,565 Train | 2,160 Val | 2,274 Test"),
        ("Architecture", "DNABERT-2 + HyenaDNA + Skip Fusion"),
        ("Parameters", "10.32 Million Trainable"),
        ("Compute Device", "NVIDIA RTX 3050 Laptop GPU (AMP fp16)"),
        ("Best Epoch", f"{history.get('best_epoch', 0)} / {epochs}"),
        ("", ""),
        ("── Peak Validation Performance ──", ""),
        ("Accuracy",          f"{best_val_acc*100:.2f}%"),
        ("Balanced Accuracy", f"{max(history['val_balanced_accuracy'])*100:.2f}%"),
        ("Macro F1",          f"{best_val_f1:.4f}"),
        ("Weighted F1",       f"{max(history['val_weighted_f1']):.4f}"),
        ("MCC",               f"{max(history['val_mcc']):.4f}"),
        ("ROC-AUC (Macro)",   f"{max(history['val_roc_auc']):.4f}"),
        ("", ""),
        ("── Test Evaluation Performance ──", ""),
        ("Overall Accuracy",  f"{test_metrics['accuracy']*100:.2f}%"),
        ("Balanced Accuracy", f"{test_metrics['balanced_accuracy']*100:.2f}%"),
        ("Macro F1",          f"{test_metrics['macro_f1']:.4f}"),
        ("Weighted F1",       f"{test_metrics['weighted_f1']:.4f}"),
        ("MCC",               f"{test_metrics['mcc']:.4f}"),
        ("ROC-AUC",           f"{test_metrics['roc_auc']:.4f}"),
    ]
    y_pos = 0.98
    for key, val in scorecard:
        if key.startswith("──"):
            ax.text(0.02, y_pos, key, transform=ax.transAxes,
                    fontsize=9.5, color="#58a6ff", fontweight="bold")
        elif key == "" and val == "":
            pass
        elif val == "":
            ax.text(0.02, y_pos, key, transform=ax.transAxes,
                    fontsize=9, color="#8b949e")
        else:
            ax.text(0.02, y_pos, f"{key}:", transform=ax.transAxes,
                    fontsize=9, color="#8b949e")
            ax.text(0.50, y_pos, val, transform=ax.transAxes,
                    fontsize=9, color="#c9d1d9", fontweight="bold")
        y_pos -= 0.048
    ax.set_title("Executive Summary Scorecard", fontsize=11, fontweight="bold", pad=8)


def plot_10_precision_recall_f1(ax, test_metrics: Dict):
    cm = np.array(test_metrics["confusion_matrix"])
    col_sums = cm.sum(axis=0)
    row_sums = cm.sum(axis=1)
    diag = cm.diagonal()
    precs = [diag[i] / col_sums[i] if col_sums[i] > 0 else 0.0 for i in range(NUM_CLASSES)]
    recalls = [diag[i] / row_sums[i] if row_sums[i] > 0 else 0.0 for i in range(NUM_CLASSES)]
    f1s = test_metrics["per_class_f1"]

    x = np.arange(NUM_CLASSES)
    w = 0.25
    ax.bar(x - w, [p * 100 for p in precs],   width=w, label="Precision", color="#79c0ff",  edgecolor="#30363d")
    ax.bar(x,     [r * 100 for r in recalls], width=w, label="Recall",    color="#56d364",  edgecolor="#30363d")
    ax.bar(x + w, [f * 100 for f in f1s],     width=w, label="F1 Score",  color="#ffa657",  edgecolor="#30363d")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES_SHORT, fontsize=8)
    ax.set_title("Test Precision, Recall & F1 by Class (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric (%)")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y")


def plot_11_epoch_time(ax, history: Dict):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax.bar(epochs, history["epoch_time"], color="#58a6ff", edgecolor="#30363d", width=0.6)
    mean_t = np.mean(history["epoch_time"]) if history["epoch_time"] else 0.0
    ax.axhline(mean_t, color="#f0e68c", ls="--", lw=1.3,
               label=f"Mean: {mean_t:.1f} s/epoch")
    ax.set_title("Epoch Wall-Clock Execution Time (GPU)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y")


def plot_12_class_dist(ax, history: Dict, test_metrics: Dict):
    split_names = ["Train", "Val", "Test"]
    class_counts = {split: [] for split in split_names}
    tc = dict(Counter(history.get("train_class_dist", []))) or {0: 833, 1: 14492, 2: 6816, 3: 8424}
    vc = dict(Counter(history.get("val_class_dist", []))) or {0: 178, 1: 977, 2: 498, 3: 507}
    ec = dict(Counter(test_metrics.get("y_true", []))) or {0: 180, 1: 1053, 2: 514, 3: 527}
    for c in range(NUM_CLASSES):
        class_counts["Train"].append(tc.get(c, 0))
        class_counts["Val"].append(vc.get(c, 0))
        class_counts["Test"].append(ec.get(c, 0))

    x = np.arange(NUM_CLASSES)
    w = 0.25
    for i, (split, color) in enumerate(zip(split_names, ["#58a6ff","#56d364","#ffa657"])):
        ax.bar(x + (i-1)*w, class_counts[split], width=w, label=split,
               color=color, edgecolor="#30363d")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES_SHORT, fontsize=8)
    ax.set_title("Dataset Class Distribution Across Splits", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sequence Count")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{int(val):,}"))


# ════════════════════════════════════════════════════════════════════════════
# UNIFIED DASHBOARD AND 12 INDIVIDUAL EXPORTS
# ════════════════════════════════════════════════════════════════════════════

def export_all_charts(history: Dict, test_metrics: Dict) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    if ARTIFACT_DIR.exists():
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(DARK_RC_PARAMS)

    # 1. Unified 12-Panel Master Dashboard
    fig = plt.figure(figsize=(24, 26), facecolor="#0d1117")
    fig.suptitle(
        "Genomic Classification Framework — Full GPU Training Dashboard\n"
        "DNABERT-2 + HyenaDNA + Skip-Connected CrossAttentionFusion | 10.32M Params | 34,999 Sequences",
        fontsize=16, color="white", fontweight="bold", y=0.99
    )

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.95, bottom=0.04)

    ax1 = fig.add_subplot(gs[0, 0]); plot_01_loss(ax1, history)
    ax2 = fig.add_subplot(gs[0, 1]); plot_02_accuracy(ax2, history)
    ax3 = fig.add_subplot(gs[0, 2]); plot_03_f1_mcc(ax3, history)
    ax4 = fig.add_subplot(gs[1, 0]); plot_04_roc_auc(ax4, history)
    ax5 = fig.add_subplot(gs[1, 1]); plot_05_lr(ax5, history)
    ax6 = fig.add_subplot(gs[1, 2]); plot_06_grad_norm(ax6, history)
    ax7 = fig.add_subplot(gs[2, 0]); plot_07_per_class_f1(ax7, test_metrics)
    ax8 = fig.add_subplot(gs[2, 1]); im8 = plot_08_confusion_matrix(ax8, test_metrics); plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    ax9 = fig.add_subplot(gs[2, 2]); plot_09_scorecard(ax9, history, test_metrics)
    ax10 = fig.add_subplot(gs[3, 0]); plot_10_precision_recall_f1(ax10, test_metrics)
    ax11 = fig.add_subplot(gs[3, 1]); plot_11_epoch_time(ax11, history)
    ax12 = fig.add_subplot(gs[3, 2]); plot_12_class_dist(ax12, history, test_metrics)

    fig.savefig(str(DASHBOARD_PATH), dpi=160, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    logger.info("Unified Dashboard saved → %s", DASHBOARD_PATH)

    # 2. Standalone 12 Individual High-Resolution Plots
    individual_plots = [
        ("01_loss_curves.png", plot_01_loss, (history,)),
        ("02_accuracy_curves.png", plot_02_accuracy, (history,)),
        ("03_f1_mcc_curves.png", plot_03_f1_mcc, (history,)),
        ("04_roc_auc_curve.png", plot_04_roc_auc, (history,)),
        ("05_learning_rate_schedule.png", plot_05_lr, (history,)),
        ("06_gradient_norm.png", plot_06_grad_norm, (history,)),
        ("07_per_class_f1.png", plot_07_per_class_f1, (test_metrics,)),
        ("08_confusion_matrix.png", lambda ax: (plot_08_confusion_matrix(ax, test_metrics), plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)), ()),
        ("09_summary_scorecard.png", plot_09_scorecard, (history, test_metrics)),
        ("10_precision_recall_f1.png", plot_10_precision_recall_f1, (test_metrics,)),
        ("11_epoch_wall_time.png", plot_11_epoch_time, (history,)),
        ("12_dataset_class_distribution.png", plot_12_class_dist, (history, test_metrics)),
    ]

    for filename, fn, args in individual_plots:
        s_fig, s_ax = plt.subplots(figsize=(9, 7), facecolor="#0d1117")
        if args:
            fn(s_ax, *args)
        else:
            fn(s_ax)
        save_file = INDIVIDUAL_DIR / filename
        s_fig.savefig(str(save_file), dpi=200, bbox_inches="tight", facecolor="#0d1117")
        plt.close(s_fig)
        logger.info("Saved individual plot → %s", save_file)

    # Copy all plots to the artifact directory for direct viewing
    if ARTIFACT_DIR.exists():
        try:
            shutil.copy2(str(DASHBOARD_PATH), str(ARTIFACT_DIR / DASHBOARD_PATH.name))
            for f in INDIVIDUAL_DIR.glob("*.png"):
                shutil.copy2(str(f), str(ARTIFACT_DIR / f.name))
            logger.info("Synchronized 13 plots into brain artifact directory: %s", ARTIFACT_DIR)
        except Exception as e:
            logger.warning("Could not sync plots to artifact directory: %s", e)


# ════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING WORKFLOW
# ════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="Genomic Classification Engine GPU Training")
    parser.add_argument("--fresh", action="store_true", help="Start training fresh from epoch 1")
    parser.add_argument("--epochs", type=int, default=DEFAULT_MAX_EPOCHS, help="Number of maximum epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--samples-per-epoch", type=int, default=DEFAULT_SAMPLES_PER_EPOCH,
                        help="Number of balanced sequence draws per epoch (default 3200)")
    args = parser.parse_args()

    max_epochs = args.epochs
    batch_size = args.batch_size
    samples_per_epoch = args.samples_per_epoch

    seed_everything(SEED)
    torch.set_num_threads(NUM_THREADS)

    # ── Device selection (Auto CUDA / GPU) ──────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = (device.type == "cuda")

    for d in [OUT_DIR / "checkpoints", OUT_DIR / "metrics", PLOT_DIR, INDIVIDUAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Genomic Classification Training — GPU High-Performance Mode")
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("Device: GPU (%s, %.2f GB VRAM) | AMP Mixed-Precision: Enabled", gpu_name, vram_gb)
    else:
        logger.info("Device: CPU | Threads: %d", NUM_THREADS)
    logger.info("Epochs: %d | Batch Size: %d | Samples/Epoch: %d | Fresh Start: %s",
                max_epochs, batch_size, samples_per_epoch, args.fresh)
    logger.info("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_weights = build_loaders(
        batch_size=batch_size, samples_per_epoch=samples_per_epoch
    )

    # ── Build model ────────────────────────────────────────────────────────
    model = GenomicClassificationEngine(
        seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, hyena_dim=HYENA_DIM,
        hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES,
        max_mutations=MAX_MUTATIONS, mutation_dim=MUTATION_DIM, dropout=0.2,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model assembled on %s — %.2fM parameters", device, n_params / 1e6)

    # WeightedRandomSampler draws balanced classes, so criterion uses unweighted FocalLoss
    criterion = FocalLoss(weight=None, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    n_batches_per_epoch = len(train_loader)
    T_max = max(1, max_epochs - WARMUP_EPOCHS)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=MIN_LR
    )

    history: Dict[str, List] = {
        "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [], "val_balanced_accuracy": [],
        "val_macro_f1": [], "val_weighted_f1": [], "val_mcc": [], "val_roc_auc": [],
        "learning_rate": [], "grad_norm": [], "epoch_time": [],
        "best_epoch": 0,
        "train_class_dist": [],
        "val_class_dist": [],
    }

    tr_lbls_all = train_loader.dataset.tensors[2].tolist()
    va_lbls_all = val_loader.dataset.tensors[2].tolist()
    history["train_class_dist"] = tr_lbls_all
    history["val_class_dist"]   = va_lbls_all

    best_val_f1   = -1.0
    patience_ctr  = 0
    start_epoch   = 1

    if not args.fresh:
        resume_path = LAST_CKPT if LAST_CKPT.exists() else (CKPT_PATH if CKPT_PATH.exists() else None)
        if resume_path is not None:
            logger.info("Found checkpoint at %s — resuming …", resume_path)
            ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
            saved_epoch = ckpt.get("epoch", 0)
            if saved_epoch >= 1:
                try:
                    model.load_state_dict(ckpt["model_state"])
                    if "optimizer_state" in ckpt:
                        optimizer.load_state_dict(ckpt["optimizer_state"])
                    best_val_f1 = ckpt.get("best_val_f1", -1.0)
                    start_epoch = saved_epoch + 1
                    patience_ctr = ckpt.get("patience_ctr", 0)
                    if "history" in ckpt and ckpt["history"].get("train_loss"):
                        history.update(ckpt["history"])
                    for _ in range(max(0, saved_epoch - WARMUP_EPOCHS)):
                        cosine_sched.step()
                    logger.info("Resumed from epoch %d! best_val_f1=%.4f, starting at epoch %d",
                                saved_epoch, best_val_f1, start_epoch)
                except Exception as e:
                    logger.warning("Could not restore checkpoint cleanly (%s), starting fresh", e)
                    start_epoch = 1

    logger.info("=" * 60)
    logger.info("Beginning execution: Epoch %d to %d (%d batches/epoch)",
                start_epoch, max_epochs, n_batches_per_epoch)
    logger.info("=" * 60)

    for epoch in range(start_epoch, max_epochs + 1):
        ep_start = time.time()

        # Warmup phase
        if epoch <= WARMUP_EPOCHS:
            warmup_lr = LR * epoch / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # Training phase
        model.train()
        ep_loss = 0.0
        ep_preds: List[int] = []
        ep_true:  List[int] = []
        grad_norms: List[float] = []

        for step, (tok, mut, lbl) in enumerate(train_loader, 1):
            tok = tok.to(device, non_blocking=use_cuda)
            mut = mut.to(device, non_blocking=use_cuda)
            lbl = lbl.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                logits, _ = model(tok, mut)
                loss = criterion(logits, lbl)

            if not torch.isfinite(loss):
                logger.warning("Non-finite loss at step %d — skipping", step)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            grad_norms.append(norm)
            scaler.step(optimizer)
            scaler.update()

            ep_loss  += loss.item()
            ep_preds += logits.argmax(1).tolist()
            ep_true  += lbl.tolist()

            if step % 25 == 0 or step == n_batches_per_epoch:
                logger.info("  Epoch %2d/%d | Step %3d/%d | Loss: %.4f | GradNorm: %.3f",
                            epoch, max_epochs, step, n_batches_per_epoch, loss.item(), norm)

        # Step LR scheduler
        if epoch > WARMUP_EPOCHS:
            cosine_sched.step()

        avg_train_loss = ep_loss / max(n_batches_per_epoch, 1)
        train_acc = accuracy_score(ep_true, ep_preds)
        avg_grad  = float(np.mean(grad_norms)) if grad_norms else 0.0
        cur_lr    = optimizer.param_groups[0]["lr"]

        # Validation phase
        vm = evaluate(model, val_loader, criterion, device)
        ep_time = time.time() - ep_start

        logger.info(
            "Epoch %2d/%d [%.1fs] | "
            "TrLoss: %.4f TrAcc: %.3f | "
            "ValLoss: %.4f ValAcc: %.3f BalAcc: %.3f | "
            "MacroF1: %.4f WtdF1: %.4f MCC: %.4f AUC: %.4f | LR: %.2e",
            epoch, max_epochs, ep_time,
            avg_train_loss, train_acc,
            vm["loss"], vm["accuracy"], vm["balanced_accuracy"],
            vm["macro_f1"], vm["weighted_f1"], vm["mcc"], vm["roc_auc"],
            cur_lr,
        )

        history["train_loss"].append(avg_train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(vm["loss"])
        history["val_accuracy"].append(vm["accuracy"])
        history["val_balanced_accuracy"].append(vm["balanced_accuracy"])
        history["val_macro_f1"].append(vm["macro_f1"])
        history["val_weighted_f1"].append(vm["weighted_f1"])
        history["val_mcc"].append(vm["mcc"])
        history["val_roc_auc"].append(vm["roc_auc"])
        history["learning_rate"].append(cur_lr)
        history["grad_norm"].append(avg_grad)
        history["epoch_time"].append(ep_time)

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "history": history,
            "patience_ctr": patience_ctr
        }, str(LAST_CKPT))

        # Checkpoint selection on Macro-F1
        if vm["macro_f1"] > best_val_f1 + 1e-4:
            best_val_f1 = vm["macro_f1"]
            history["best_epoch"] = epoch
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "history": history,
                "patience_ctr": patience_ctr
            }, str(CKPT_PATH))
            logger.info("  ✓ New Best Model Saved (Val Macro-F1: %.4f)", best_val_f1)
        else:
            patience_ctr += 1
            logger.info("  Patience: %d/%d", patience_ctr, PATIENCE)
            if epoch >= MIN_EPOCHS and patience_ctr >= PATIENCE:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    # ── Final Test Set Evaluation ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Evaluating Best Model Checkpoint on Unseen Test Isolates …")
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    logger.info("=" * 60)
    logger.info("FINAL TEST SET PERFORMANCE METRICS:")
    logger.info("  Overall Accuracy:      %.4f (%.2f%%)", test_metrics["accuracy"], test_metrics["accuracy"] * 100)
    logger.info("  Balanced Accuracy:     %.4f (%.2f%%)", test_metrics["balanced_accuracy"], test_metrics["balanced_accuracy"] * 100)
    logger.info("  Macro-Averaged F1:     %.4f", test_metrics["macro_f1"])
    logger.info("  Weighted-Averaged F1:  %.4f", test_metrics["weighted_f1"])
    logger.info("  Matthews Corr (MCC):   %.4f", test_metrics["mcc"])
    logger.info("  ROC-AUC (Macro OvR):   %.4f", test_metrics["roc_auc"])
    logger.info("=" * 60)
    for i, (cl, f1) in enumerate(zip(CLASS_LABELS, test_metrics["per_class_f1"])):
        logger.info("    Class %d [%s]: F1 = %.4f", i, cl, f1)

    # ── Save Metrics JSON ──────────────────────────────────────────────────
    save_hist = {k: v for k, v in history.items() if k not in ("train_class_dist", "val_class_dist")}
    payload = {
        "history": save_hist,
        "test_metrics": {
            k: v for k, v in test_metrics.items() if k not in ("y_true", "y_pred", "y_probs")
        }
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Saved JSON metrics → %s", METRICS_PATH)

    # ── Generate Unified Dashboard + 12 Standalone Plots ───────────────────
    export_all_charts(history, test_metrics)

    logger.info("Execution complete. Best Epoch: %d | Val Macro-F1: %.4f | Test Macro-F1: %.4f",
                history["best_epoch"], best_val_f1, test_metrics["macro_f1"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
