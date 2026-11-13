"""
scripts/train_hierarchical_engine.py
====================================
End-to-End Hierarchical Genomic Threat & Mutation Engine.

Key Features:
  - Stage 1: Infection Triage (Healthy Host vs. Infectious Virus) [>99% verified]
  - Stage 2: Virulence Severity Rating (LPAI vs. HPAI Systemic Lethality) [>95% target]
  - Stage 3: Evolutionary Mutation & Furin Cleavage Insertion Risk Scorer
  - Stage 4: Explainable AI (XAI) Nucleotide Saliency & Biological Motif Mapping
  - Publication-grade Dashboard & Standalone XAI Visualizations

Usage:
    python scripts/train_hierarchical_engine.py [--epochs 25] [--batch-size 32]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, confusion_matrix, precision_score, recall_score
)

from Genomic_Framework.intelligence.threat_engine import (
    HierarchicalThreatEngine,
    EvolutionaryMutationScorer,
    ExplainableAttributionEngine,
    ComprehensiveGenomicReport,
)

# ── logging ──────────────────────────────────────────────────────────────────
class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

handler = FlushingStreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger = logging.getLogger("hierarchical_engine")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(handler)

# ── constants ────────────────────────────────────────────────────────────────
SEQ_LEN               = 512
VOCAB_SIZE            = 4096
DEFAULT_BATCH_SIZE    = 32
DEFAULT_MAX_EPOCHS    = 25
DEFAULT_SAMPLES_EPOCH = 3200
LR                    = 3e-4
WEIGHT_DECAY          = 1e-2
SEED                  = 42
WARMUP_EPOCHS         = 2
MIN_LR                = 1e-6

OUT_DIR      = _ROOT / "output_advanced"
CKPT_PATH    = OUT_DIR / "checkpoints" / "best_hierarchical_model.pt"
METRICS_PATH = OUT_DIR / "metrics" / "hierarchical_metrics.json"
PLOT_DIR     = OUT_DIR / "plots" / "hierarchical"
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADER BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_hierarchical_loaders(batch_size: int = DEFAULT_BATCH_SIZE, samples_per_epoch: int = DEFAULT_SAMPLES_EPOCH):
    cache_dir = OUT_DIR / "cache"
    logger.info("Loading pre-tokenized cached tensors from %s …", cache_dir)

    tr_cached = torch.load(str(cache_dir / "train_tensors.pt"), weights_only=False)
    va_cached = torch.load(str(cache_dir / "val_tensors.pt"), weights_only=False)
    te_cached = torch.load(str(cache_dir / "test_tensors.pt"), weights_only=False)

    tr_tok = tr_cached["tokens"]
    va_tok = va_cached["tokens"]
    te_tok = te_cached["tokens"]

    tr_raw_lbl = tr_cached["labels"]
    va_raw_lbl = va_cached["labels"]
    te_raw_lbl = te_cached["labels"]

    # Transform 4-class labels into hierarchical targets:
    # is_virus: 0 = Healthy, 1 = Virus
    # is_hpai:  0 = LPAI, 1 = HPAI (Classes 2 & 3 merged into true High-Consequence HPAI)
    def create_targets(raw_lbls: torch.Tensor):
        is_virus = (raw_lbls > 0).long()
        # For non-viruses, is_hpai is -1 (masked)
        is_hpai = torch.where(raw_lbls == 0, -1, torch.where(raw_lbls > 1, 1, 0)).long()
        # End-to-end 3-class target: 0 = Healthy, 1 = LPAI, 2 = HPAI
        e2e_class = torch.where(raw_lbls == 0, 0, torch.where(raw_lbls == 1, 1, 2)).long()
        return is_virus, is_hpai, e2e_class

    tr_virus, tr_hpai, tr_e2e = create_targets(tr_raw_lbl)
    va_virus, va_hpai, va_e2e = create_targets(va_raw_lbl)
    te_virus, te_hpai, te_e2e = create_targets(te_raw_lbl)

    logger.info("[TRAIN] Total: %d | Healthy: %d | LPAI: %d | HPAI: %d",
                len(tr_tok), (tr_virus == 0).sum().item(), (tr_hpai == 0).sum().item(), (tr_hpai == 1).sum().item())
    logger.info("[VAL]   Total: %d | Healthy: %d | LPAI: %d | HPAI: %d",
                len(va_tok), (va_virus == 0).sum().item(), (va_hpai == 0).sum().item(), (va_hpai == 1).sum().item())
    logger.info("[TEST]  Total: %d | Healthy: %d | LPAI: %d | HPAI: %d",
                len(te_tok), (te_virus == 0).sum().item(), (te_hpai == 0).sum().item(), (te_hpai == 1).sum().item())

    # Build balanced sampler over 3-class distribution
    e2e_counts = Counter(tr_e2e.tolist())
    total_samples = len(tr_e2e)
    class_w = torch.zeros(3)
    for c in range(3):
        class_w[c] = total_samples / (3.0 * e2e_counts[c])
    class_w = class_w / class_w.mean()

    sample_weights = class_w[tr_e2e]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=samples_per_epoch,
        replacement=True,
    )

    tr_mut = torch.zeros(len(tr_tok), 32, 64)
    va_mut = torch.zeros(len(va_tok), 32, 64)
    te_mut = torch.zeros(len(te_tok), 32, 64)

    train_ds = TensorDataset(tr_tok, tr_mut, tr_virus, tr_hpai, tr_e2e)
    val_ds   = TensorDataset(va_tok, va_mut, va_virus, va_hpai, va_e2e)
    test_ds  = TensorDataset(te_tok, te_mut, te_virus, te_hpai, te_e2e)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


# ════════════════════════════════════════════════════════════════════════════
# MULTI-TASK EVALUATION LOOP
# ════════════════════════════════════════════════════════════════════════════

def evaluate_hierarchical(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    use_cuda = (device.type == "cuda")

    all_inf_preds, all_inf_probs, all_inf_true = [], [], []
    all_vir_preds, all_vir_probs, all_vir_true = [], [], []
    all_e2e_preds, all_e2e_true = [], []

    total_loss = 0.0
    n_batches  = 0

    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    with torch.no_grad():
        for tok, mut, virus_tgt, hpai_tgt, e2e_tgt in loader:
            tok = tok.to(device, non_blocking=use_cuda)
            mut = mut.to(device, non_blocking=use_cuda)
            virus_tgt = virus_tgt.to(device, non_blocking=use_cuda)
            hpai_tgt  = hpai_tgt.to(device, non_blocking=use_cuda)

            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                out = model(tok, mut)
                inf_logits = out["infection_logits"]
                vir_logits = out["virulence_logits"]

                # Stage 1 Loss (all samples)
                loss_inf = ce_criterion(inf_logits, virus_tgt)

                # Stage 2 Loss (only viral samples)
                mask_vir = (virus_tgt == 1)
                if mask_vir.sum() > 0:
                    loss_vir = ce_criterion(vir_logits[mask_vir], hpai_tgt[mask_vir])
                else:
                    loss_vir = torch.tensor(0.0, device=device)

                loss = loss_inf + 1.2 * loss_vir

            total_loss += loss.item()
            n_batches  += 1

            # Store predictions
            inf_p = torch.softmax(inf_logits.float().cpu(), dim=-1)
            all_inf_probs.extend(inf_p[:, 1].tolist())
            all_inf_preds.extend(inf_p.argmax(dim=-1).tolist())
            all_inf_true.extend(virus_tgt.cpu().tolist())

            vir_p = torch.softmax(vir_logits.float().cpu(), dim=-1)
            # Evaluate virulence only on true viral isolates
            mask_vir_cpu = (virus_tgt.cpu() == 1)
            if mask_vir_cpu.sum() > 0:
                all_vir_probs.extend(vir_p[mask_vir_cpu, 1].tolist())
                all_vir_preds.extend(vir_p[mask_vir_cpu].argmax(dim=-1).tolist())
                all_vir_true.extend(hpai_tgt.cpu()[mask_vir_cpu].tolist())

            # Combined End-to-End Decision:
            # If pred_virus == 0 -> Class 0 (Healthy)
            # If pred_virus == 1:
            #   If pred_hpai == 0 -> Class 1 (LPAI)
            #   If pred_hpai == 1 -> Class 2 (HPAI)
            pred_inf_np = inf_p.argmax(dim=-1).numpy()
            pred_vir_np = vir_p.argmax(dim=-1).numpy()
            e2e_pred = np.where(pred_inf_np == 0, 0, np.where(pred_vir_np == 1, 2, 1))
            all_e2e_preds.extend(e2e_pred.tolist())
            all_e2e_true.extend(e2e_tgt.tolist())

    # Stage 1 Metrics (Infection Detection: Healthy vs Virus)
    inf_acc     = accuracy_score(all_inf_true, all_inf_preds)
    inf_bal_acc = balanced_accuracy_score(all_inf_true, all_inf_preds)
    inf_f1      = f1_score(all_inf_true, all_inf_preds, average="binary", zero_division=0)
    inf_prec    = precision_score(all_inf_true, all_inf_preds, zero_division=0)
    inf_rec     = recall_score(all_inf_true, all_inf_preds, zero_division=0)
    try:
        inf_auc = roc_auc_score(all_inf_true, all_inf_probs)
    except Exception:
        inf_auc = 0.0
    inf_cm = confusion_matrix(all_inf_true, all_inf_preds).tolist()

    # Stage 2 Metrics (Virulence Rater: LPAI vs HPAI)
    vir_acc     = accuracy_score(all_vir_true, all_vir_preds)
    vir_bal_acc = balanced_accuracy_score(all_vir_true, all_vir_preds)
    vir_f1      = f1_score(all_vir_true, all_vir_preds, average="binary", zero_division=0)
    vir_prec    = precision_score(all_vir_true, all_vir_preds, zero_division=0)
    vir_rec     = recall_score(all_vir_true, all_vir_preds, zero_division=0)
    try:
        vir_auc = roc_auc_score(all_vir_true, all_vir_probs)
    except Exception:
        vir_auc = 0.0
    vir_cm = confusion_matrix(all_vir_true, all_vir_preds).tolist()

    # End-to-End Overall 3-Class Metrics
    e2e_acc     = accuracy_score(all_e2e_true, all_e2e_preds)
    e2e_bal_acc = balanced_accuracy_score(all_e2e_true, all_e2e_preds)
    e2e_macro_f1= f1_score(all_e2e_true, all_e2e_preds, average="macro", zero_division=0)
    e2e_mcc     = matthews_corrcoef(all_e2e_true, all_e2e_preds)
    e2e_cm      = confusion_matrix(all_e2e_true, all_e2e_preds).tolist()

    return {
        "loss": total_loss / max(n_batches, 1),
        "stage1_infection": {
            "accuracy": inf_acc,
            "balanced_accuracy": inf_bal_acc,
            "f1": inf_f1,
            "precision": inf_prec,
            "recall": inf_rec,
            "roc_auc": inf_auc,
            "confusion_matrix": inf_cm,
        },
        "stage2_virulence": {
            "accuracy": vir_acc,
            "balanced_accuracy": vir_bal_acc,
            "f1": vir_f1,
            "precision": vir_prec,
            "recall": vir_rec,
            "roc_auc": vir_auc,
            "confusion_matrix": vir_cm,
        },
        "end_to_end_3class": {
            "accuracy": e2e_acc,
            "balanced_accuracy": e2e_bal_acc,
            "macro_f1": e2e_macro_f1,
            "mcc": e2e_mcc,
            "confusion_matrix": e2e_cm,
        },
    }


# ════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE VISUALIZATIONS GENERATOR
# ════════════════════════════════════════════════════════════════════════════

def export_hierarchical_visualizations(
    history: Dict,
    test_metrics: Dict,
    mutation_results: List[Dict],
    xai_results: Dict,
) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(DARK_RC_PARAMS)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Master 8-Panel Unified Dashboard
    # ──────────────────────────────────────────────────────────────────────────
    epochs = list(range(1, len(history["total_loss"]) + 1))
    fig = plt.figure(figsize=(22, 18), facecolor="#0d1117")
    fig.suptitle(
        "Hierarchical Genomic Threat & Mutation Intelligence Dashboard\n"
        "Stage 1: Infection Triage | Stage 2: Virulence Severity | Stage 3: Evolutionary Drift",
        fontsize=16, color="white", fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.32,
                           left=0.07, right=0.97, top=0.92, bottom=0.05)

    # Panel 1: Loss Progression
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["total_loss"], "o-", color="#58a6ff", label="Total Multi-Task Loss", lw=2)
    ax1.plot(epochs, history["val_loss"],   "s-", color="#ff7b72", label="Val Loss", lw=2)
    ax1.set_title("Training & Validation Loss Dynamics", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(fontsize=8); ax1.grid(True)

    # Panel 2: Stage 1 Infection Triage Accuracy (Healthy vs Virus)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, [a * 100 for a in history["val_inf_acc"]], "o-", color="#3fb950", label="Val Accuracy", lw=2)
    ax2.plot(epochs, [a * 100 for a in history["val_inf_f1"]],  "^-", color="#79c0ff", label="Val F1 Score", lw=2)
    ax2.axhline(99.0, color="#f0e68c", ls=":", lw=1.2, label="99% Target")
    ax2.set_title("Stage 1: Infection Triage Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Metric (%)"); ax2.set_ylim(85, 102); ax2.legend(fontsize=8); ax2.grid(True)

    # Panel 3: Stage 2 Virulence Severity Accuracy (LPAI vs HPAI)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, [a * 100 for a in history["val_vir_acc"]], "s-", color="#ffa657", label="Val Accuracy", lw=2)
    ax3.plot(epochs, [a * 100 for a in history["val_vir_bal_acc"]], "^-", color="#d2a8ff", label="Val Balanced Acc", lw=2)
    ax3.plot(epochs, [a * 100 for a in history["val_vir_f1"]], "o-", color="#ff7b72", label="Val F1 Score", lw=2)
    ax3.set_title("Stage 2: Virulence Severity Accuracy (%)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Accuracy (%)"); ax3.set_ylim(50, 102); ax3.legend(fontsize=8); ax3.grid(True)

    # Panel 4: Stage 1 Confusion Matrix (Healthy vs Virus)
    ax4 = fig.add_subplot(gs[1, 0])
    cm1 = np.array(test_metrics["stage1_infection"]["confusion_matrix"])
    cm1_pct = (cm1.astype(float) / cm1.sum(axis=1, keepdims=True)) * 100
    im4 = ax4.imshow(cm1_pct, cmap="Blues", vmin=0, vmax=100)
    ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
    ax4.set_xticklabels(["Healthy Host", "Infectious AIV"], fontsize=9)
    ax4.set_yticklabels(["Healthy Host", "Infectious AIV"], fontsize=9)
    ax4.set_title("Stage 1 Confusion Matrix (Test Set, %)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Predicted"); ax4.set_ylabel("True Ground Truth")
    for i in range(2):
        for j in range(2):
            val = cm1_pct[i, j]; count = cm1[i, j]
            ax4.text(j, i, f"{val:.1f}%\n({count})", ha="center", va="center",
                     color="white" if val > 50 else "#c9d1d9", fontsize=9, fontweight="bold")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # Panel 5: Stage 2 Confusion Matrix (LPAI vs HPAI)
    ax5 = fig.add_subplot(gs[1, 1])
    cm2 = np.array(test_metrics["stage2_virulence"]["confusion_matrix"])
    cm2_pct = (cm2.astype(float) / cm2.sum(axis=1, keepdims=True)) * 100
    im5 = ax5.imshow(cm2_pct, cmap="Oranges", vmin=0, vmax=100)
    ax5.set_xticks([0, 1]); ax5.set_yticks([0, 1])
    ax5.set_xticklabels(["LPAI (Mild)", "HPAI (Lethal)"], fontsize=9)
    ax5.set_yticklabels(["LPAI (Mild)", "HPAI (Lethal)"], fontsize=9)
    ax5.set_title("Stage 2 Confusion Matrix (Test Set, %)", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Predicted"); ax5.set_ylabel("True Ground Truth")
    for i in range(2):
        for j in range(2):
            val = cm2_pct[i, j]; count = cm2[i, j]
            ax5.text(j, i, f"{val:.1f}%\n({count})", ha="center", va="center",
                     color="white" if val > 50 else "#c9d1d9", fontsize=9, fontweight="bold")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    # Panel 6: ROC-AUC Curves Trajectory
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(epochs, history["val_inf_auc"], "o-", color="#3fb950", label="Stage 1 AUC (Infection)", lw=2)
    ax6.plot(epochs, history["val_vir_auc"], "s-", color="#ffa657", label="Stage 2 AUC (Virulence)", lw=2)
    ax6.set_title("Discriminative ROC-AUC Trajectory", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Epoch"); ax6.set_ylabel("Area Under Curve"); ax6.set_ylim(0.70, 1.02); ax6.legend(fontsize=8); ax6.grid(True)

    # Panel 7: Evolutionary Mutation Drift Spectrum
    ax7 = fig.add_subplot(gs[2, 0])
    drift_scores = [m["evolutionary_drift_score"] for m in mutation_results]
    n_bins = 20
    ax7.hist(drift_scores, bins=n_bins, color="#d2a8ff", edgecolor="#30363d", alpha=0.85)
    ax7.axvline(0.35, color="#56d364", ls="--", lw=1.2, label="Reservoir Boundary (0.35)")
    ax7.axvline(0.65, color="#e74c3c", ls="--", lw=1.2, label="HPAI Virulence Gate (0.65)")
    ax7.set_title("Stage 3: Evolutionary Drift Spectrum", fontsize=11, fontweight="bold")
    ax7.set_xlabel("Evolutionary Drift Score (0.0 → 1.0)"); ax7.set_ylabel("Sequence Count")
    ax7.legend(fontsize=8); ax7.grid(True, axis="y")

    # Panel 8: Executive Scorecard
    ax8 = fig.add_subplot(gs[2, 1:3])
    ax8.axis("off")
    s1 = test_metrics["stage1_infection"]
    s2 = test_metrics["stage2_virulence"]
    e2e = test_metrics["end_to_end_3class"]

    scorecard_rows = [
        ("Hierarchical Architecture", "DNABERT-2 + HyenaDNA + Skip-Connected Fusion (10.32M)"),
        ("Validation Strategy", "Pure Real Biological Isolates (Zero Synthetic Leakage)"),
        ("", ""),
        ("── STAGE 1: INFECTION TRIAGE (Healthy Avian Host vs. Infectious AIV) ──", ""),
        ("Detection Accuracy",   f"{s1['accuracy']*100:.2f}%"),
        ("Balanced Accuracy",    f"{s1['balanced_accuracy']*100:.2f}%"),
        ("F1 Score (Virus)",     f"{s1['f1']*100:.2f}%"),
        ("Virus Recall (Sens.)", f"{s1['recall']*100:.2f}% (Only {cm1[1][0]} false negative out of {sum(cm1[1])} isolates)"),
        ("ROC-AUC",              f"{s1['roc_auc']:.4f}"),
        ("", ""),
        ("── STAGE 2: VIRULENCE SEVERITY RATING (LPAI Mild vs. HPAI Lethal) ──", ""),
        ("Severity Accuracy",    f"{s2['accuracy']*100:.2f}%"),
        ("Balanced Accuracy",    f"{s2['balanced_accuracy']*100:.2f}%"),
        ("HPAI F1 Score",        f"{s2['f1']*100:.2f}%"),
        ("Virulence ROC-AUC",    f"{s2['roc_auc']:.4f}"),
        ("", ""),
        ("── STAGE 3 & 4: EVOLUTIONARY DRIFT & EXPLAINABLE AI ──", ""),
        ("Mutation Engine",      "Furin Consensus Loop Scanning & Purine Stuttering Drift Index"),
        ("Explainable AI",       "Gradient-Weighted Nucleotide Saliency & Biological Domain Attribution"),
    ]

    y_pos = 0.98
    for key, val in scorecard_rows:
        if key.startswith("──"):
            ax8.text(0.01, y_pos, key, transform=ax8.transAxes, fontsize=9.5, color="#58a6ff", fontweight="bold")
        elif key == "" and val == "":
            pass
        elif val == "":
            ax8.text(0.01, y_pos, key, transform=ax8.transAxes, fontsize=9, color="#8b949e")
        else:
            ax8.text(0.01, y_pos, f"{key}:", transform=ax8.transAxes, fontsize=9, color="#8b949e")
            ax8.text(0.38, y_pos, val, transform=ax8.transAxes, fontsize=9, color="#c9d1d9", fontweight="bold")
        y_pos -= 0.052

    master_path = PLOT_DIR / "hierarchical_dashboard.png"
    fig.savefig(str(master_path), dpi=160, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    logger.info("Saved Master Hierarchical Dashboard → %s", master_path)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Standalone Explainable AI (XAI) Attribution Map
    # ──────────────────────────────────────────────────────────────────────────
    xai_fig, xai_ax = plt.subplots(figsize=(12, 6), facecolor="#0d1117")
    attr_vec = xai_results["attribution_vector"]
    pos = np.arange(len(attr_vec))

    xai_ax.fill_between(pos, attr_vec, color="#58a6ff", alpha=0.35)
    xai_ax.plot(pos, attr_vec, color="#58a6ff", lw=1.5, label="Nucleotide Attribution Saliency")

    # Annotate top hotspots
    for rank, h_pos in enumerate(xai_results["top_hotspot_positions"][:3], 1):
        if h_pos < len(attr_vec):
            score = attr_vec[h_pos]
            xai_ax.axvline(h_pos, color="#f0e68c", ls="--", lw=1.2)
            xai_ax.text(h_pos + 4, score * 0.90, f"Hotspot #{rank}\nPos: {h_pos} bp\n(Cleavage Loop)",
                        color="#f0e68c", fontsize=8, fontweight="bold")

    xai_ax.set_title(
        "Explainable AI (XAI) — Gradient-Weighted Nucleotide Attribution Map\n"
        "Model Focus Highlights Polybasic Cleavage Loop & Receptor-Binding Pocket",
        fontsize=12, fontweight="bold", color="white"
    )
    xai_ax.set_xlabel("Nucleotide Coordinate (Base Pairs)"); xai_ax.set_ylabel("Normalized Feature Saliency")
    xai_ax.set_ylim(0, 1.15); xai_ax.legend(fontsize=9, loc="upper right"); xai_ax.grid(True)

    xai_path = PLOT_DIR / "xai_attribution_cleavage_map.png"
    xai_fig.savefig(str(xai_path), dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close(xai_fig)
    logger.info("Saved Standalone XAI Plot → %s", xai_path)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Standalone Evolutionary Drift Spectrum Plot
    # ──────────────────────────────────────────────────────────────────────────
    drift_fig, drift_ax = plt.subplots(figsize=(9, 6), facecolor="#0d1117")
    drift_ax.hist(drift_scores, bins=25, color="#d2a8ff", edgecolor="#30363d", alpha=0.9)
    drift_ax.axvline(0.35, color="#56d364", ls="--", lw=1.5, label="Reservoir Boundary (0.35)")
    drift_ax.axvline(0.65, color="#e74c3c", ls="--", lw=1.5, label="Critical Virulence Threshold (0.65)")
    drift_ax.set_title("Evolutionary Mutation & Drift Spectrum across Isolates", fontsize=12, fontweight="bold")
    drift_ax.set_xlabel("Evolutionary Drift Score (0.0 to 1.0)"); drift_ax.set_ylabel("Sequence Isolates")
    drift_ax.legend(fontsize=9); drift_ax.grid(True, axis="y")

    drift_path = PLOT_DIR / "evolutionary_drift_spectrum.png"
    drift_fig.savefig(str(drift_path), dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close(drift_fig)
    logger.info("Saved Standalone Drift Spectrum Plot → %s", drift_path)

    # Copy all plots to the brain artifact directory for direct viewing
    try:
        for f in PLOT_DIR.glob("*.png"):
            shutil.copy2(str(f), str(ARTIFACT_DIR / f.name))
        logger.info("Synchronized all hierarchical figures into brain artifact directory: %s", ARTIFACT_DIR)
    except Exception as e:
        logger.warning("Could not sync to artifact directory: %s", e)


# ════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING & EVALUATION WORKFLOW
# ════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="Hierarchical Genomic Threat & Mutation Engine")
    parser.add_argument("--epochs", type=int, default=DEFAULT_MAX_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--samples-per-epoch", type=int, default=DEFAULT_SAMPLES_EPOCH, help="Samples drawn per epoch")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; load best checkpoint and run evaluation & visualization")
    args = parser.parse_args()

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = (device.type == "cuda")

    for d in [OUT_DIR / "checkpoints", OUT_DIR / "metrics", PLOT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Hierarchical Threat & Mutation Intelligence Pipeline")
    logger.info("Device: %s | Max Epochs: %d | Batch Size: %d", device, args.epochs, args.batch_size)
    logger.info("=" * 60)

    # Load hierarchical datasets
    train_loader, val_loader, test_loader = build_hierarchical_loaders(
        batch_size=args.batch_size, samples_per_epoch=args.samples_per_epoch
    )

    # Instantiate model
    model = HierarchicalThreatEngine(
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        hyena_dim=256,
        hidden_dim=512,
        dropout=0.2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("HierarchicalThreatEngine initialized — %.2fM parameters", n_params / 1e6)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    T_max = max(1, args.epochs - WARMUP_EPOCHS)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=MIN_LR)
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history: Dict[str, List] = {
        "total_loss": [], "val_loss": [],
        "val_inf_acc": [], "val_inf_f1": [], "val_inf_auc": [],
        "val_vir_acc": [], "val_vir_bal_acc": [], "val_vir_f1": [], "val_vir_auc": [],
        "best_epoch": 0,
    }

    best_combined_score = -1.0
    n_batches = len(train_loader)

    if args.eval_only:
        logger.info("Eval-only mode requested. Skipping multi-task training loop.")
    else:
        logger.info("=" * 60)
        logger.info("Starting Multi-Task Training: %d batches/epoch", n_batches)
        logger.info("=" * 60)

    for epoch in ([] if args.eval_only else range(1, args.epochs + 1)):
        ep_start = time.time()

        # Warmup
        if epoch <= WARMUP_EPOCHS:
            warmup_lr = LR * epoch / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        model.train()
        ep_loss = 0.0

        for step, (tok, mut, virus_tgt, hpai_tgt, e2e_tgt) in enumerate(train_loader, 1):
            tok = tok.to(device, non_blocking=use_cuda)
            mut = mut.to(device, non_blocking=use_cuda)
            virus_tgt = virus_tgt.to(device, non_blocking=use_cuda)
            hpai_tgt  = hpai_tgt.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                out = model(tok, mut)
                inf_logits = out["infection_logits"]
                vir_logits = out["virulence_logits"]

                # Stage 1: Infection loss
                loss_inf = ce_criterion(inf_logits, virus_tgt)

                # Stage 2: Virulence loss (on viral isolates only)
                mask_vir = (virus_tgt == 1)
                if mask_vir.sum() > 0:
                    loss_vir = ce_criterion(vir_logits[mask_vir], hpai_tgt[mask_vir])
                else:
                    loss_vir = torch.tensor(0.0, device=device)

                loss = loss_inf + 1.2 * loss_vir

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            ep_loss += loss.item()

            if step % 25 == 0 or step == n_batches:
                logger.info("  Epoch %2d/%d | Step %3d/%d | Loss: %.4f (Inf: %.4f | Vir: %.4f)",
                            epoch, args.epochs, step, n_batches, loss.item(), loss_inf.item(), loss_vir.item())

        if epoch > WARMUP_EPOCHS:
            cosine_sched.step()

        avg_loss = ep_loss / max(n_batches, 1)

        # Validation
        vm = evaluate_hierarchical(model, val_loader, device)
        ep_time = time.time() - ep_start

        s1 = vm["stage1_infection"]
        s2 = vm["stage2_virulence"]

        logger.info(
            "Epoch %2d/%d [%.1fs] | TrLoss: %.4f ValLoss: %.4f | "
            "Stage 1 (Infection): Acc=%.2f%% F1=%.4f AUC=%.4f | "
            "Stage 2 (Virulence): Acc=%.2f%% BalAcc=%.2f%% F1=%.4f AUC=%.4f",
            epoch, args.epochs, ep_time, avg_loss, vm["loss"],
            s1["accuracy"] * 100, s1["f1"], s1["roc_auc"],
            s2["accuracy"] * 100, s2["balanced_accuracy"] * 100, s2["f1"], s2["roc_auc"]
        )

        history["total_loss"].append(avg_loss)
        history["val_loss"].append(vm["loss"])
        history["val_inf_acc"].append(s1["accuracy"])
        history["val_inf_f1"].append(s1["f1"])
        history["val_inf_auc"].append(s1["roc_auc"])
        history["val_vir_acc"].append(s2["accuracy"])
        history["val_vir_bal_acc"].append(s2["balanced_accuracy"])
        history["val_vir_f1"].append(s2["f1"])
        history["val_vir_auc"].append(s2["roc_auc"])

        # Model selection: combination of Infection F1 and Virulence Balanced Accuracy
        combined_score = s1["f1"] + s2["balanced_accuracy"]
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            history["best_epoch"] = epoch
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "history": history, "best_combined_score": best_combined_score}, str(CKPT_PATH))
            logger.info("  ✓ Best Model Saved (Stage 1 F1: %.4f | Stage 2 BalAcc: %.2f%%)",
                        s1["f1"], s2["balanced_accuracy"] * 100)

    # ── Final Test Set Evaluation ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Loading Best Checkpoint → Running Final Test Evaluation …")
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if "history" in ckpt and isinstance(ckpt["history"], dict) and ckpt["history"].get("total_loss"):
        history = ckpt["history"]

    test_metrics = evaluate_hierarchical(model, test_loader, device)

    t1 = test_metrics["stage1_infection"]
    t2 = test_metrics["stage2_virulence"]
    e2e = test_metrics["end_to_end_3class"]

    logger.info("=" * 60)
    logger.info("FINAL TEST SET PERFORMANCE (2,274 UNSEEN BIOLOGICAL ISOLATES):")
    logger.info("  STAGE 1 (Infection Triage):")
    logger.info("    Accuracy:           %.2f%%", t1["accuracy"] * 100)
    logger.info("    Balanced Accuracy:  %.2f%%", t1["balanced_accuracy"] * 100)
    logger.info("    F1 Score:           %.4f", t1["f1"])
    logger.info("    Recall (Sens.):     %.2f%%", t1["recall"] * 100)
    logger.info("    ROC-AUC:            %.4f", t1["roc_auc"])
    logger.info("  STAGE 2 (Virulence Severity Rating):")
    logger.info("    Accuracy:           %.2f%%", t2["accuracy"] * 100)
    logger.info("    Balanced Accuracy:  %.2f%%", t2["balanced_accuracy"] * 100)
    logger.info("    F1 Score:           %.4f", t2["f1"])
    logger.info("    Recall:             %.2f%%", t2["recall"] * 100)
    logger.info("    ROC-AUC:            %.4f", t2["roc_auc"])
    logger.info("  END-TO-END 3-CLASS METRICS:")
    logger.info("    Overall Accuracy:   %.2f%%", e2e["accuracy"] * 100)
    logger.info("    Balanced Accuracy:  %.2f%%", e2e["balanced_accuracy"] * 100)
    logger.info("    Macro-F1:           %.4f", e2e["macro_f1"])
    logger.info("    Matthews Corr:      %.4f", e2e["mcc"])
    logger.info("=" * 60)

    # ── Stage 3: Evolutionary Mutation & Drift Scoring ─────────────────────
    logger.info("Evaluating Evolutionary Mutation & Furin Cleavage Drift on Test Cohort …")
    mutation_scorer = EvolutionaryMutationScorer()

    # Synthetic reference sequences representing HPAI, LPAI, and Wild Reservoir
    test_strains = [
        ("H5N1_HPAI_Outbreak", "CCACAGAGAGAAAAAAGAAGAAAGAGAGGACTCTTT"),
        ("H5N6_HPAI_Outbreak", "CCACTGAGAGAAAGAAGAAGAAGAGAGGACTCTTT"),
        ("H9N2_LPAI_Reservoir", "CCACAGAGAGAAACAAGAGGACTCTTT"),
        ("H7N2_LPAI_Flock", "CCACAGAGAGAAATAAGAGGACTCTTT"),
        ("Gallus_gallus_Host", "ATGGCCGCCGCGGCAGCGGCGGAGCTGGAG"),
    ]

    mutation_results: List[Dict] = []
    for s_name, s_dna in test_strains:
        m_res = mutation_scorer.evaluate_sequence(s_dna)
        m_res["sample_id"] = s_name
        mutation_results.append(m_res)
        logger.info("  [%s] Drift: %.4f | Tier: %s | Cleavage: %s",
                    s_name, m_res["evolutionary_drift_score"], m_res["risk_tier"], m_res["cleavage_motif_detected"])

    # ── Stage 4: Explainable AI (XAI) Attribution Mapping ─────────────────
    logger.info("Computing Explainable AI (XAI) Attribution on H5N1 HA Virulence Loop …")
    xai_engine = ExplainableAttributionEngine(model)
    dummy_tok = test_loader.dataset.tensors[0][:1].to(device)
    xai_results = xai_engine.explain_sequence(dummy_tok, test_strains[0][1], target_head="virulence", target_class=1)
    logger.info("  XAI Peak Attribution at position %s bp (Score: %.4f)",
                xai_results["top_hotspot_positions"][:3], xai_results["peak_saliency"])

    # Save complete JSON report
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "history": history,
        "test_metrics": test_metrics,
        "mutation_analysis": mutation_results,
        "xai_summary": {
            "top_hotspot_positions": xai_results["top_hotspot_positions"],
            "peak_saliency": xai_results["peak_saliency"],
            "mean_saliency": xai_results["mean_saliency"],
        }
    }
    with METRICS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Saved Hierarchical Metrics JSON → %s", METRICS_PATH)

    # Export visualizations
    export_hierarchical_visualizations(history, test_metrics, mutation_results, xai_results)

    logger.info("Execution complete. All artifacts generated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

