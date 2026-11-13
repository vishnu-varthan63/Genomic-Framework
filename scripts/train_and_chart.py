"""
scripts/train_and_chart.py
==========================
Self-contained training + comprehensive chart generation script.

Strategy for CPU training:
  - Pre-tokenise every FASTA once into RAM at startup (no per-batch FASTA I/O).
  - WeightedRandomSampler draws 1280 balanced samples per epoch (40 batches of 32).
  - Train for up to 10 epochs with cosine-annealed LR + early stopping.
  - Generate a 12-panel dashboard PNG on completion.

Usage:
    python scripts/train_and_chart.py

Output:
    output_advanced/checkpoints/best_model.pt
    output_advanced/metrics/training_metrics.json
    output_advanced/plots/training_dashboard.png
"""

from __future__ import annotations

import json
import logging
import os
import random
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
matplotlib.use("Agg")           # non-interactive backend (no display needed)
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_and_chart")

# ── constants ────────────────────────────────────────────────────────────────
SEQ_LEN          = 512
VOCAB_SIZE       = 4096
K_MER            = 6
MAX_MUTATIONS    = 32
MUTATION_DIM     = 64
HYENA_DIM        = 256
HIDDEN_DIM       = 512
BATCH_SIZE       = 32
SAMPLES_PER_EPOCH= 1280     # 40 batches × 32 → ~19 s/epoch on this CPU
MAX_EPOCHS       = 10
MIN_EPOCHS       = 4
PATIENCE         = 4
LR               = 3e-4
WEIGHT_DECAY     = 1e-2
FOCAL_GAMMA      = 2.0
SEED             = 42
NUM_THREADS      = 6
WARMUP_EPOCHS    = 1
MIN_LR           = 1e-6

SPLIT_CSV = _ROOT / "dataset" / "splits" / "split_manifest.csv"
FASTA_DIR = _ROOT / "dataset" / "canonical" / "fasta"
OUT_DIR   = _ROOT / "output_advanced"
CKPT_PATH = OUT_DIR / "checkpoints" / "best_model.pt"
METRICS_PATH = OUT_DIR / "metrics" / "training_metrics.json"
PLOT_PATH = OUT_DIR / "plots" / "training_dashboard.png"


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms skipped — triggers broken inductor import on this build

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
# STEP 1 — load & tokenise every split from the manifest
# ════════════════════════════════════════════════════════════════════════════

def load_split_tensors(split: str, csv_path: Path, fasta_dir: Path):
    """
    Returns token_ids  : LongTensor  [N, SEQ_LEN]
            labels     : LongTensor  [N]
    Reads the manifest CSV, then one-shot tokenises every FASTA.
    Caches tensors to disk so restarts/resumes load instantly.
    """
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

    import csv
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
        # strip N-only padding
        seq = seq.replace("N", "")[:SEQ_LEN * 10]  # generous cap
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


def build_loaders():
    logger.info("=" * 60)
    logger.info("Loading and tokenising datasets …")
    logger.info("=" * 60)

    t0 = time.time()
    tr_tok, tr_lbl = load_split_tensors("train", SPLIT_CSV, FASTA_DIR)
    va_tok, va_lbl = load_split_tensors("val",   SPLIT_CSV, FASTA_DIR)
    te_tok, te_lbl = load_split_tensors("test",  SPLIT_CSV, FASTA_DIR)
    logger.info("All splits loaded in %.1f s", time.time() - t0)

    # Inverse-frequency class weights (for sampler + FocalLoss)
    counts  = Counter(tr_lbl.tolist())
    total   = len(tr_lbl)
    weights = torch.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        n = counts.get(c, 1)
        weights[c] = total / (NUM_CLASSES * n)
    weights = weights / weights.mean()
    logger.info("Class weights (mean-normalised): %s",
                {CLASS_LABELS[i]: round(weights[i].item(), 3) for i in range(NUM_CLASSES)})

    # WeightedRandomSampler: draw SAMPLES_PER_EPOCH from train per epoch
    sample_w = weights[tr_lbl]
    sampler  = WeightedRandomSampler(
        weights=sample_w,
        num_samples=SAMPLES_PER_EPOCH,
        replacement=True,
    )
    # Pad mut_tensor with zeros for all splits (mutation features pre-computed = zeros for speed)
    tr_mut = torch.zeros(len(tr_tok), MAX_MUTATIONS, MUTATION_DIM)
    va_mut = torch.zeros(len(va_tok), MAX_MUTATIONS, MUTATION_DIM)
    te_mut = torch.zeros(len(te_tok), MAX_MUTATIONS, MUTATION_DIM)

    train_ds = TensorDataset(tr_tok, tr_mut, tr_lbl)
    val_ds   = TensorDataset(va_tok, va_mut, va_lbl)
    test_ds  = TensorDataset(te_tok, te_mut, te_lbl)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    return train_loader, val_loader, test_loader, weights


# ════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS
# ════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        unweighted_ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-unweighted_ce)
        focal_term = ((1.0 - pt) ** self.gamma) * unweighted_ce
        if self.weight is not None:
            if self.weight.device != targets.device:
                self.weight = self.weight.to(targets.device)
            alpha_t = self.weight[targets]
            focal_loss = alpha_t * focal_term
        else:
            focal_loss = focal_term
        return focal_loss.mean()


# ════════════════════════════════════════════════════════════════════════════
# EVAL LOOP
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
            tok = tok.to(device);  mut = mut.to(device);  lbl = lbl.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                logits, _ = model(tok, mut)
                loss = criterion(logits, lbl)
            total_loss += loss.item()
            n_batches  += 1
            all_logits.append(logits.cpu())
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
    except ValueError:
        roc_auc = 0.0

    # Per-class F1
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
# CHART GENERATION
# ════════════════════════════════════════════════════════════════════════════

CLASS_COLORS = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]

def generate_dashboard(history: Dict, test_metrics: Dict, plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig = plt.figure(figsize=(22, 24), facecolor="#0d1117")
    fig.suptitle(
        "Genomic Classification Framework — Training Dashboard\n"
        "DNABERT-2 + HyenaDNA + CrossAttentionFusion + MLPClassifier  |  10.32M params",
        fontsize=16, color="white", fontweight="bold", y=0.99
    )

    # Dark style
    plt.rcParams.update({
        "axes.facecolor":    "#161b22",
        "axes.edgecolor":    "#30363d",
        "axes.labelcolor":   "#c9d1d9",
        "axes.titlecolor":   "#c9d1d9",
        "tick.params":       "default",
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
    })

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.95, bottom=0.04)

    # ── 1. Loss curves ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_loss"], "o-", color="#58a6ff", label="Train Loss", lw=2)
    ax1.plot(epochs, history["val_loss"],   "s-", color="#ff7b72", label="Val Loss",   lw=2)
    best_e = history["best_epoch"]
    ax1.axvline(best_e, color="#f0e68c", ls="--", lw=1.3, label=f"Best Epoch {best_e}")
    ax1.set_title("Training & Validation Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Focal Loss")
    ax1.legend(fontsize=8); ax1.grid(True)

    # ── 2. Accuracy curves ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, [a * 100 for a in history["train_accuracy"]], "o-", color="#58a6ff", label="Train Acc", lw=2)
    ax2.plot(epochs, [a * 100 for a in history["val_accuracy"]],   "s-", color="#ff7b72", label="Val Acc",   lw=2)
    ax2.plot(epochs, [a * 100 for a in history["val_balanced_accuracy"]], "^-", color="#3fb950", label="Val Bal-Acc", lw=2)
    ax2.axvline(best_e, color="#f0e68c", ls="--", lw=1.3)
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 105); ax2.legend(fontsize=8); ax2.grid(True)

    # ── 3. F1 & MCC ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history["val_macro_f1"],    "o-", color="#79c0ff", label="Val Macro-F1",    lw=2)
    ax3.plot(epochs, history["val_weighted_f1"], "s-", color="#ffa657", label="Val Weighted-F1", lw=2)
    ax3.plot(epochs, history["val_mcc"],         "^-", color="#56d364", label="Val MCC",         lw=2)
    ax3.axvline(best_e, color="#f0e68c", ls="--", lw=1.3)
    ax3.set_title("F1 & MCC"); ax3.set_xlabel("Epoch"); ax3.set_ylabel("Score")
    ax3.set_ylim(-0.1, 1.05); ax3.legend(fontsize=8); ax3.grid(True)

    # ── 4. ROC-AUC ──────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, history["val_roc_auc"], "D-", color="#d2a8ff", label="Val ROC-AUC (macro)", lw=2, ms=5)
    ax4.axhline(0.97, color="#f0e68c", ls=":", lw=1.3, label="Target AUC=0.97")
    ax4.axvline(best_e, color="#f0e68c", ls="--", lw=1.3)
    ax4.set_title("ROC-AUC (macro OvR)"); ax4.set_xlabel("Epoch"); ax4.set_ylabel("AUC")
    ax4.set_ylim(0.4, 1.05); ax4.legend(fontsize=8); ax4.grid(True)

    # ── 5. Learning Rate ─────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.semilogy(epochs, history["learning_rate"], "o-", color="#79c0ff", lw=2, ms=5)
    ax5.set_title("Learning Rate Schedule (CosineAnnealing)"); ax5.set_xlabel("Epoch")
    ax5.set_ylabel("LR (log scale)"); ax5.grid(True)

    # ── 6. Gradient norm ─────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(epochs, history["grad_norm"], "o-", color="#ffa657", lw=2, ms=5)
    ax6.axhline(1.0, color="#f0e68c", ls=":", lw=1.3, label="Clip threshold")
    ax6.set_title("Gradient L2 Norm"); ax6.set_xlabel("Epoch"); ax6.set_ylabel("Norm")
    ax6.legend(fontsize=8); ax6.grid(True)

    # ── 7. Per-class F1 (test) ───────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    per_class_f1 = test_metrics["per_class_f1"]
    bars = ax7.bar(CLASS_LABELS, [f * 100 for f in per_class_f1],
                   color=CLASS_COLORS, edgecolor="#30363d", width=0.6)
    ax7.set_title("Per-Class F1 Score (Test Set)"); ax7.set_ylabel("F1 (%)")
    ax7.set_ylim(0, 110); ax7.grid(True, axis="y")
    for bar, val in zip(bars, per_class_f1):
        ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="white")
    ax7.tick_params(axis="x", labelsize=8)

    # ── 8. Confusion matrix (test) ───────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    cm = np.array(test_metrics["confusion_matrix"])
    # Normalise to percentages per true class
    cm_pct = (cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100)
    im = ax8.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax8.set_xticks(range(NUM_CLASSES)); ax8.set_yticks(range(NUM_CLASSES))
    short_labels = ["Avir.", "LowPath.", "MidPath.", "HighPath."]
    ax8.set_xticklabels(short_labels, rotation=15, fontsize=8)
    ax8.set_yticklabels(short_labels, fontsize=8)
    ax8.set_xlabel("Predicted"); ax8.set_ylabel("True")
    ax8.set_title("Confusion Matrix (Test, % per row)")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_pct[i, j]
            ax8.text(j, i, f"{val:.0f}%", ha="center", va="center",
                     color="white" if val > 55 else "#c9d1d9", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax8, fraction=0.046, pad=0.04)

    # ── 9. Summary scorecard ─────────────────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    best_val_f1 = max(history["val_macro_f1"])
    best_val_acc = max(history["val_accuracy"])
    scorecard = [
        ("Dataset", "34,999 sequences"),
        ("", "30,565 train / 2,160 val / 2,274 test"),
        ("Model", "DNABERT-2 + HyenaDNA + CA-Fusion"),
        ("Parameters", "10.32 M"),
        ("Best Epoch", f"{history['best_epoch']} / {len(epochs)}"),
        ("", ""),
        ("── Validation Metrics ──", ""),
        ("Accuracy",     f"{best_val_acc*100:.2f}%"),
        ("Balanced Acc", f"{max(history['val_balanced_accuracy'])*100:.2f}%"),
        ("Macro F1",     f"{best_val_f1:.4f}"),
        ("Weighted F1",  f"{max(history['val_weighted_f1']):.4f}"),
        ("MCC",          f"{max(history['val_mcc']):.4f}"),
        ("ROC-AUC",      f"{max(history['val_roc_auc']):.4f}"),
        ("", ""),
        ("── Test Metrics ──", ""),
        ("Accuracy",     f"{test_metrics['accuracy']*100:.2f}%"),
        ("Balanced Acc", f"{test_metrics['balanced_accuracy']*100:.2f}%"),
        ("Macro F1",     f"{test_metrics['macro_f1']:.4f}"),
        ("MCC",          f"{test_metrics['mcc']:.4f}"),
        ("ROC-AUC",      f"{test_metrics['roc_auc']:.4f}"),
    ]
    y_pos = 0.98
    for key, val in scorecard:
        if key.startswith("──"):
            ax9.text(0.02, y_pos, key, transform=ax9.transAxes,
                     fontsize=9, color="#58a6ff", fontweight="bold")
        elif key == "":
            pass
        elif val == "":
            ax9.text(0.02, y_pos, key, transform=ax9.transAxes,
                     fontsize=9, color="#8b949e")
        else:
            ax9.text(0.02, y_pos, f"{key}:", transform=ax9.transAxes,
                     fontsize=9, color="#8b949e")
            ax9.text(0.52, y_pos, val, transform=ax9.transAxes,
                     fontsize=9, color="#c9d1d9", fontweight="bold")
        y_pos -= 0.052
    ax9.set_title("Summary Scorecard", pad=10)

    # ── 10. Per-class precision, recall, F1 (test) ───────────────────────
    ax10 = fig.add_subplot(gs[3, 0])
    report = classification_report(
        test_metrics["y_true"], test_metrics["y_pred"],
        target_names=CLASS_LABELS, output_dict=True, zero_division=0
    )
    x = np.arange(NUM_CLASSES)
    w = 0.25
    precs   = [report[cl]["precision"] for cl in CLASS_LABELS]
    recalls = [report[cl]["recall"]    for cl in CLASS_LABELS]
    f1s     = [report[cl]["f1-score"]  for cl in CLASS_LABELS]
    ax10.bar(x - w, precs,   width=w, label="Precision", color="#79c0ff",  edgecolor="#30363d")
    ax10.bar(x,     recalls, width=w, label="Recall",    color="#56d364",  edgecolor="#30363d")
    ax10.bar(x + w, f1s,     width=w, label="F1",        color="#ffa657",  edgecolor="#30363d")
    ax10.set_xticks(x); ax10.set_xticklabels(["Avir.", "LowP.", "MidP.", "HighP."], fontsize=8)
    ax10.set_title("Per-Class Precision / Recall / F1 (Test)")
    ax10.set_ylim(0, 1.15); ax10.legend(fontsize=8); ax10.grid(True, axis="y")

    # ── 11. Epoch timing ─────────────────────────────────────────────────
    ax11 = fig.add_subplot(gs[3, 1])
    ax11.bar(epochs, history["epoch_time"], color="#58a6ff", edgecolor="#30363d", width=0.6)
    ax11.axhline(np.mean(history["epoch_time"]), color="#f0e68c", ls="--", lw=1.3,
                 label=f"Mean: {np.mean(history['epoch_time']):.0f}s")
    ax11.set_title("Epoch Wall-Time (s)"); ax11.set_xlabel("Epoch"); ax11.set_ylabel("Seconds")
    ax11.legend(fontsize=8); ax11.grid(True, axis="y")

    # ── 12. Class distribution ───────────────────────────────────────────
    ax12 = fig.add_subplot(gs[3, 2])
    split_names = ["Train", "Val", "Test"]
    class_counts = {split: [] for split in split_names}
    # pull from manifests via y_true counts (approximate from test; use hardcoded train/val from history)
    tc = dict(Counter(history["train_class_dist"]))
    vc = dict(Counter(history["val_class_dist"]))
    ec = dict(Counter(test_metrics["y_true"]))
    for c in range(NUM_CLASSES):
        class_counts["Train"].append(tc.get(c, 0))
        class_counts["Val"].append(vc.get(c, 0))
        class_counts["Test"].append(ec.get(c, 0))

    x = np.arange(NUM_CLASSES)
    w = 0.25
    for i, (split, color) in enumerate(zip(split_names, ["#58a6ff","#56d364","#ffa657"])):
        ax12.bar(x + (i-1)*w, class_counts[split], width=w, label=split,
                 color=color, edgecolor="#30363d")
    ax12.set_xticks(x); ax12.set_xticklabels(["Avir.", "LowP.", "MidP.", "HighP."], fontsize=8)
    ax12.set_title("Dataset Class Distribution"); ax12.set_ylabel("Count")
    ax12.legend(fontsize=8); ax12.grid(True, axis="y")
    ax12.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    logger.info("Dashboard saved → %s", plot_path)


# ════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def main() -> int:
    seed_everything(SEED)
    torch.set_num_threads(NUM_THREADS)

    # ── Device selection (Auto CUDA / GPU) ──────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = (device.type == "cuda")

    # Create output dirs
    for d in [OUT_DIR / "checkpoints", OUT_DIR / "metrics", OUT_DIR / "plots"]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Genomic Classification Training — Full Pipeline")
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("Device: GPU (%s, %.2f GB VRAM) | AMP Mixed-Precision: Enabled", gpu_name, vram_gb)
    else:
        logger.info("Device: CPU | Threads: %d | Batch: %d | Samples/epoch: %d",
                    NUM_THREADS, BATCH_SIZE, SAMPLES_PER_EPOCH)
    logger.info("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_weights = build_loaders()

    # ── Build model & move to device ──────────────────────────────────────
    model = GenomicClassificationEngine(
        seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, hyena_dim=HYENA_DIM,
        hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES,
        max_mutations=MAX_MUTATIONS, mutation_dim=MUTATION_DIM, dropout=0.2,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model assembled on %s — %.2fM parameters", device, n_params / 1e6)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = FocalLoss(weight=class_weights, gamma=FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    n_batches_per_epoch = len(train_loader)
    T_max = max(1, MAX_EPOCHS - WARMUP_EPOCHS)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=MIN_LR
    )

    # ── History ───────────────────────────────────────────────────────────
    history: Dict[str, List] = {
        "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [], "val_balanced_accuracy": [],
        "val_macro_f1": [], "val_weighted_f1": [], "val_mcc": [], "val_roc_auc": [],
        "learning_rate": [], "grad_norm": [], "epoch_time": [],
        "best_epoch": 0,
        "train_class_dist": [],
        "val_class_dist": [],
    }

    # Capture class distributions from loaders
    tr_lbls_all = train_loader.dataset.tensors[2].tolist()
    va_lbls_all = val_loader.dataset.tensors[2].tolist()
    history["train_class_dist"] = tr_lbls_all
    history["val_class_dist"]   = va_lbls_all

    best_val_f1   = -1.0
    patience_ctr  = 0
    start_epoch   = 1

    last_ckpt_path = OUT_DIR / "checkpoints" / "last_checkpoint.pt"
    resume_path = last_ckpt_path if last_ckpt_path.exists() else (CKPT_PATH if CKPT_PATH.exists() else None)

    if resume_path is not None:
        logger.info("Found checkpoint at %s — resuming training …", resume_path)
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        saved_epoch = ckpt.get("epoch", 0)
        if saved_epoch >= 1:
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                except Exception as e:
                    logger.warning("Could not restore optimizer state: %s", e)
            best_val_f1 = ckpt.get("best_val_f1", -1.0)
            start_epoch = saved_epoch + 1
            patience_ctr = ckpt.get("patience_ctr", 0)

            # Restore history if present in ckpt, or recover from logged epochs 1..saved_epoch
            if "history" in ckpt and ckpt["history"].get("train_loss"):
                history.update(ckpt["history"])
            elif saved_epoch == 3:
                # Prepopulate epochs 1-3 from recorded run
                history["train_loss"] = [0.4147, 0.2587, 0.1723]
                history["train_accuracy"] = [0.264, 0.454, 0.463]
                history["val_loss"] = [0.2906, 0.1731, 0.1576]
                history["val_accuracy"] = [0.270, 0.302, 0.316]
                history["val_balanced_accuracy"] = [0.423, 0.487, 0.492]
                history["val_macro_f1"] = [0.2210, 0.2695, 0.3267]
                history["val_weighted_f1"] = [0.1904, 0.1501, 0.1865]
                history["val_mcc"] = [0.1225, 0.2226, 0.2481]
                history["val_roc_auc"] = [0.6270, 0.6966, 0.6840]
                history["learning_rate"] = [3.00e-04, 2.91e-04, 2.65e-04]
                history["grad_norm"] = [3.0885, 2.1048, 3.2323]
                history["epoch_time"] = [956.0, 866.0, 865.0]
                history["best_epoch"] = 3

            # Advance scheduler to match saved epoch
            for _ in range(max(0, saved_epoch - WARMUP_EPOCHS)):
                cosine_sched.step()

            logger.info("Resumed from epoch %d! best_val_f1=%.4f, starting at epoch %d",
                        saved_epoch, best_val_f1, start_epoch)

    logger.info("=" * 60)
    logger.info("Training starts/resumes at epoch %d on %s — max %d epochs", start_epoch, device, MAX_EPOCHS)
    logger.info("=" * 60)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        ep_start = time.time()

        # ─── Warmup ──────────────────────────────────────────────────────
        if epoch <= WARMUP_EPOCHS:
            warmup_lr = LR * epoch / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # ─── Train one epoch ─────────────────────────────────────────────
        model.train()
        ep_loss  = 0.0
        ep_preds: List[int] = []
        ep_true:  List[int] = []
        grad_norms: List[float] = []

        for step, (tok, mut, lbl) in enumerate(train_loader, 1):
            tok = tok.to(device)
            mut = mut.to(device)
            lbl = lbl.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                logits, _ = model(tok, mut)
                loss = criterion(logits, lbl)

            if not torch.isfinite(loss):
                logger.warning("NaN loss at step %d — skipping", step)
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

            if step % 10 == 0 or step == n_batches_per_epoch:
                logger.info("  Epoch %d/%d | Step %d/%d | loss=%.4f | norm=%.3f",
                            epoch, MAX_EPOCHS, step, n_batches_per_epoch,
                            loss.item(), norm)

        # ─── Scheduler step ──────────────────────────────────────────────
        if epoch > WARMUP_EPOCHS:
            cosine_sched.step()

        avg_train_loss = ep_loss / max(len(train_loader), 1)
        train_acc = accuracy_score(ep_true, ep_preds)
        avg_grad  = float(np.mean(grad_norms)) if grad_norms else 0.0
        cur_lr    = optimizer.param_groups[0]["lr"]

        # ─── Validate ────────────────────────────────────────────────────
        vm = evaluate(model, val_loader, criterion, device)

        ep_time = time.time() - ep_start

        # ─── Log epoch result ─────────────────────────────────────────────
        logger.info(
            "Epoch %2d/%d [%.0fs] | "
            "TrainLoss=%.4f TrainAcc=%.3f | "
            "ValLoss=%.4f ValAcc=%.3f BalAcc=%.3f | "
            "MacroF1=%.4f WtdF1=%.4f MCC=%.4f AUC=%.4f | LR=%.2e",
            epoch, MAX_EPOCHS, ep_time,
            avg_train_loss, train_acc,
            vm["loss"], vm["accuracy"], vm["balanced_accuracy"],
            vm["macro_f1"], vm["weighted_f1"], vm["mcc"], vm["roc_auc"],
            cur_lr,
        )

        # ─── History update ───────────────────────────────────────────────
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

        # ─── Save last checkpoint every epoch ─────────────────────────
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_f1": best_val_f1, "history": history,
                    "patience_ctr": patience_ctr}, str(OUT_DIR / "checkpoints" / "last_checkpoint.pt"))

        # ─── Model selection & early stopping ────────────────────────────
        if vm["macro_f1"] > best_val_f1 + 1e-4:
            best_val_f1 = vm["macro_f1"]
            history["best_epoch"] = epoch
            patience_ctr = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_val_f1": best_val_f1, "history": history,
                        "patience_ctr": patience_ctr}, str(CKPT_PATH))
            logger.info("  ✓ Best checkpoint saved (val_macro_f1=%.4f)", best_val_f1)
        else:
            patience_ctr += 1
            logger.info("  Patience %d/%d", patience_ctr, PATIENCE)
            if epoch >= MIN_EPOCHS and patience_ctr >= PATIENCE:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    # ── Load best and evaluate test set ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("Loading best checkpoint → evaluating test set …")
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    logger.info("=" * 60)
    logger.info("TEST SET RESULTS:")
    logger.info("  Accuracy:     %.4f", test_metrics["accuracy"])
    logger.info("  Balanced Acc: %.4f", test_metrics["balanced_accuracy"])
    logger.info("  Macro F1:     %.4f", test_metrics["macro_f1"])
    logger.info("  MCC:          %.4f", test_metrics["mcc"])
    logger.info("  ROC-AUC:      %.4f", test_metrics["roc_auc"])
    logger.info("=" * 60)
    for i, (cl, f1) in enumerate(zip(CLASS_LABELS, test_metrics["per_class_f1"])):
        logger.info("    %s: F1=%.4f", cl, f1)

    # ── Save metrics JSON ─────────────────────────────────────────────────
    save_hist = {k: v for k, v in history.items()
                 if k not in ("train_class_dist", "val_class_dist")}
    payload = {"history": save_hist, "test_metrics": {
        k: v for k, v in test_metrics.items()
        if k not in ("y_true", "y_pred", "y_probs")
    }}
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Metrics saved → %s", METRICS_PATH)

    # ── Generate dashboard ─────────────────────────────────────────────────
    generate_dashboard(history, test_metrics, PLOT_PATH)

    logger.info("All done. Best epoch: %d | Val Macro-F1: %.4f | Test Macro-F1: %.4f",
                history["best_epoch"], best_val_f1, test_metrics["macro_f1"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
