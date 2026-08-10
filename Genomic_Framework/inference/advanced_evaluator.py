"""
Genomic_Framework/inference/advanced_evaluator.py
==================================================
Production-Grade Testing, Evaluation, and Inference Engine

Features:
  - Batch inference without gradients (@torch.no_grad())
  - Automatic hardware detection (CPU/CUDA/MPS)
  - Full metrics computation: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, Cohen's Kappa
  - Sample-level prediction export: Sample ID, Ground Truth, Predicted Class, Probabilities, Confidence
  - Multi-format Export: CSV, JSON, Excel (.xlsx)
  - Plot Generation: Confusion Matrix Heatmap, Classification Report Table, ROC Curves, PR Curves (PNG)
  - Automated Research Reports: Markdown & HTML evaluation reports
"""

from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)

from Genomic_Framework.pipeline.preprocessor import CLASS_LABELS, NUM_CLASSES
from Genomic_Framework.training.advanced_trainer import detect_device

logger = logging.getLogger(__name__)


class AdvancedEvaluator:
    """
    Complete Testing & Inference Pipeline.
    Loads trained model checkpoints, runs zero-gradient batch evaluation,
    computes research metrics, generates visualizations, and exports predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: Union[str, Path],
        output_dir: Union[str, Path] = "output_advanced",
        device: Optional[torch.device] = None,
        class_labels: Optional[List[str]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.metric_dir = self.output_dir / "metrics"
        self.plot_dir = self.output_dir / "plots"
        self.pred_dir = self.output_dir / "predictions"
        self.log_dir = self.output_dir / "logs"
        self.report_dir = self.output_dir / "reports"

        for d in [self.metric_dir, self.plot_dir, self.pred_dir, self.log_dir, self.report_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.device = device or detect_device("auto")
        self.class_labels = class_labels or CLASS_LABELS
        self.num_classes = len(self.class_labels)

        self.model = model.to(self.device)
        self.checkpoint_path = Path(checkpoint_path)
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")

        logger.info("Loading model weights for inference from: %s", self.checkpoint_path)
        state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        model_state = state.get("model_state", state)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model_state)
        else:
            # Handle DataParallel prefix if present in checkpoint
            clean_state = {}
            for k, v in model_state.items():
                clean_key = k.replace("module.", "") if k.startswith("module.") else k
                clean_state[clean_key] = v
            self.model.load_state_dict(clean_state)

        self.model.eval()
        logger.info("Model successfully loaded and set to evaluation mode.")

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        save_outputs: bool = True,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run inference over test_loader, calculate all metrics, generate plots,
        and export predictions.
        """
        start_time = time.time()
        logger.info("Executing Batch Inference over test dataset (%d samples) ...", len(test_loader.dataset))

        all_y_true: List[int] = []
        all_y_pred: List[int] = []
        all_y_probs: List[List[float]] = []
        all_sample_ids: List[str] = []

        sample_counter = 1

        for batch_idx, batch in enumerate(tqdm(test_loader, desc="  Inference Progress", leave=False)):
            token_ids, mut_tensor, labels = batch
            token_ids = token_ids.to(self.device, non_blocking=True)
            mut_tensor = mut_tensor.to(self.device, non_blocking=True)

            logits, _ = self.model(token_ids, mut_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            lbls = labels.numpy()

            for i in range(len(lbls)):
                all_y_true.append(int(lbls[i]))
                all_y_pred.append(int(preds[i]))
                all_y_probs.append(probs[i].tolist())
                all_sample_ids.append(f"Sample_{sample_counter:05d}")
                sample_counter += 1

        exec_duration = time.time() - start_time
        logger.info("Inference completed in %.3f seconds (%.2f ms/sample)", exec_duration, (exec_duration / max(1, len(all_y_true))) * 1000)

        # ----------------------------------------------------------------------
        # Compute Evaluation Metrics
        # ----------------------------------------------------------------------
        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)
        y_probs = np.array(all_y_probs)

        acc = float(accuracy_score(y_true, y_pred))
        # FIX: Add balanced accuracy (critical for imbalanced datasets)
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        # FIX: Add weighted F1 and micro F1
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        micro_f1    = float(f1_score(y_true, y_pred, average="micro",    zero_division=0))
        mcc = float(matthews_corrcoef(y_true, y_pred))
        # FIX: Guard cohen_kappa for single-class edge case
        try:
            kappa = float(cohen_kappa_score(y_true, y_pred))
        except ValueError:
            kappa = 0.0

        per_p = precision_score(y_true, y_pred, average=None, labels=list(range(self.num_classes)), zero_division=0)
        per_r = recall_score(y_true, y_pred, average=None, labels=list(range(self.num_classes)), zero_division=0)
        per_f = f1_score(y_true, y_pred, average=None, labels=list(range(self.num_classes)), zero_division=0)

        per_p_dict = {self.class_labels[i]: float(per_p[i]) for i in range(self.num_classes)}
        per_r_dict = {self.class_labels[i]: float(per_r[i]) for i in range(self.num_classes)}
        per_f_dict = {self.class_labels[i]: float(per_f[i]) for i in range(self.num_classes)}

        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes))).tolist()
        cls_report_dict = classification_report(y_true, y_pred, target_names=self.class_labels, output_dict=True, zero_division=0)
        cls_report_str = classification_report(y_true, y_pred, target_names=self.class_labels, zero_division=0)

        # FIX: Per-class ROC-AUC with presence guard (avoids NaN when class absent from test)
        roc_auc_macro = float("nan")
        pr_auc_macro = 0.0
        roc_auc_dict: Dict[str, float] = {}
        pr_auc_dict: Dict[str, float] = {}
        roc_curves: Dict[str, Dict[str, List[float]]] = {}
        pr_curves: Dict[str, Dict[str, List[float]]] = {}
        unique_classes = set(y_true.tolist())
        classes_in_test = [i for i in range(self.num_classes) if i in unique_classes]
        missing_classes = [self.class_labels[i] for i in range(self.num_classes) if i not in unique_classes]

        if missing_classes:
            logger.warning(
                "[AUDIT] Classes MISSING from test set: %s. ROC-AUC for these classes will be NaN.",
                missing_classes,
            )

        if len(classes_in_test) > 1:
            try:
                y_true_onehot = np.eye(self.num_classes)[y_true]
                # Macro ROC-AUC only over classes that are present in test
                present_onehot = y_true_onehot[:, classes_in_test]
                present_probs  = y_probs[:, classes_in_test]
                roc_auc_macro = float(roc_auc_score(present_onehot, present_probs, average="macro", multi_class="ovr"))

                pr_aucs = []
                for i in range(self.num_classes):
                    c_name = self.class_labels[i]
                    if i not in unique_classes:
                        # Class absent from test — mark as NaN, skip curve
                        roc_auc_dict[c_name] = float("nan")
                        pr_auc_dict[c_name]  = float("nan")
                        continue

                    # Per-class ROC-AUC (one-vs-rest)
                    fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
                    roc_auc_val = float(auc(fpr, tpr))
                    roc_auc_dict[c_name] = roc_auc_val
                    roc_curves[c_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

                    # Per-class PR-AUC
                    p_c, r_c, _ = precision_recall_curve(y_true_onehot[:, i], y_probs[:, i])
                    pr_val = float(auc(r_c, p_c))
                    pr_auc_dict[c_name] = pr_val
                    pr_curves[c_name] = {"precision": p_c.tolist(), "recall": r_c.tolist()}
                    pr_aucs.append(pr_val)

                pr_auc_macro = float(np.mean(pr_aucs)) if pr_aucs else float("nan")
            except Exception as exc:
                logger.warning("ROC/PR AUC calculation exception: %s", exc)
        else:
            logger.warning(
                "[AUDIT] Only %d unique class(es) found in y_true. ROC-AUC requires >= 2 classes. "
                "Check your test split for class coverage.",
                len(classes_in_test),
            )

        eval_results: Dict[str, Any] = {
            "execution_duration_sec": exec_duration,
            "samples_evaluated": len(y_true),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,           # FIX: added
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,             # FIX: added
            "micro_f1": micro_f1,                   # FIX: added
            "roc_auc_macro": roc_auc_macro,
            "pr_auc_macro": pr_auc_macro,
            "mcc": mcc,
            "cohen_kappa": kappa,
            "classes_in_test": [self.class_labels[i] for i in classes_in_test],
            "classes_missing_from_test": missing_classes,  # FIX: audit field
            "per_class_precision": per_p_dict,
            "per_class_recall": per_r_dict,
            "per_class_f1": per_f_dict,
            "per_class_roc_auc": roc_auc_dict,
            "per_class_pr_auc": pr_auc_dict,
            "confusion_matrix": cm,
            "classification_report": cls_report_dict,
        }

        # Build prediction records
        predictions: List[Dict[str, Any]] = []
        for i in range(len(y_true)):
            gt_lbl = self.class_labels[y_true[i]]
            pred_lbl = self.class_labels[y_pred[i]]
            probs_list = y_probs[i].tolist()
            conf_score = float(np.max(probs_list))

            rec = {
                "Sample_ID": all_sample_ids[i],
                "Ground_Truth": gt_lbl,
                "Predicted_Class": pred_lbl,
                "Confidence_Score": conf_score,
            }
            for idx, c_name in enumerate(self.class_labels):
                rec[f"Probability_{c_name}"] = probs_list[idx]
            predictions.append(rec)

        if save_outputs:
            formats = formats or ["csv", "json", "excel"]
            self._save_predictions(predictions, formats)
            self._save_metrics(eval_results)
            self._generate_plots(cm, cls_report_str, roc_curves, pr_curves, eval_results, y_probs)
            self._generate_reports(eval_results, cls_report_str)

        return eval_results

    # --------------------------------------------------------------------------
    # Prediction Exporters
    # --------------------------------------------------------------------------

    def _save_predictions(self, predictions: List[Dict[str, Any]], formats: List[str]) -> None:
        df = pd.DataFrame(predictions)

        if "csv" in formats:
            csv_path = self.pred_dir / "predictions.csv"
            df.to_csv(csv_path, index=False)
            logger.info("Saved CSV predictions to %s", csv_path)

        if "json" in formats:
            json_path = self.pred_dir / "predictions.json"
            json_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
            logger.info("Saved JSON predictions to %s", json_path)

        if "excel" in formats:
            xlsx_path = self.pred_dir / "predictions.xlsx"
            try:
                df.to_excel(xlsx_path, index=False, engine="openpyxl")
                logger.info("Saved Excel predictions to %s", xlsx_path)
            except Exception as exc:
                logger.warning("Excel export failed (openpyxl missing/error): %s", exc)

    def _save_metrics(self, metrics: Dict[str, Any]) -> None:
        # FIX: Save JSON metrics
        json_path = self.metric_dir / "evaluation_metrics.json"
        json_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        logger.info("Saved evaluation metrics JSON to %s", json_path)

        # FIX: Also save classification report as CSV for easier analysis
        cls_report = metrics.get("classification_report", {})
        if cls_report:
            rows = []
            for cls_name, vals in cls_report.items():
                if isinstance(vals, dict):
                    rows.append({
                        "class": cls_name,
                        "precision": vals.get("precision", ""),
                        "recall": vals.get("recall", ""),
                        "f1-score": vals.get("f1-score", ""),
                        "support": vals.get("support", ""),
                    })
            if rows:
                import csv as csv_module
                csv_path = self.metric_dir / "classification_report.csv"
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv_module.DictWriter(f, fieldnames=["class", "precision", "recall", "f1-score", "support"])
                    writer.writeheader()
                    writer.writerows(rows)
                logger.info("Saved classification report CSV to %s", csv_path)

    # --------------------------------------------------------------------------
    # Visualizations (Plots)
    # --------------------------------------------------------------------------

    def _generate_plots(
        self,
        cm: List[List[int]],
        cls_report_str: str,
        roc_curves: Dict[str, Dict[str, List[float]]],
        pr_curves: Dict[str, Dict[str, List[float]]],
        metrics: Dict[str, Any],
        y_probs: "np.ndarray | None" = None,
    ) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # 1. Confusion Matrix Heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            cm_arr = np.array(cm)
            im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
            ax.figure.colorbar(im, ax=ax)
            ax.set(
                xticks=np.arange(self.num_classes),
                yticks=np.arange(self.num_classes),
                xticklabels=self.class_labels,
                yticklabels=self.class_labels,
                title="Confusion Matrix",
                ylabel="Ground Truth Class",
                xlabel="Predicted Class",
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 1.0
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    ax.text(
                        j, i, format(cm_arr[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm_arr[i, j] > thresh else "black",
                    )
            plt.tight_layout()
            cm_path = self.plot_dir / "confusion_matrix.png"
            fig.savefig(cm_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved Confusion Matrix plot to %s", cm_path)

            # 2. ROC Curves (Multi-class One-vs-Rest)
            if roc_curves:
                fig, ax = plt.subplots(figsize=(8, 6))
                for c_name, curve in roc_curves.items():
                    auc_val = metrics.get("per_class_roc_auc", {}).get(c_name, float("nan"))
                    label_str = f"{c_name} (AUC = {auc_val:.3f})" if not (isinstance(auc_val, float) and np.isnan(auc_val)) else f"{c_name} (AUC = N/A)"
                    ax.plot(curve["fpr"], curve["tpr"], lw=2, label=label_str)
                ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance")
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                roc_macro = metrics['roc_auc_macro']
                roc_title = f"ROC Curves (Macro AUC = {roc_macro:.3f})" if not (isinstance(roc_macro, float) and np.isnan(roc_macro)) else "ROC Curves (Macro AUC = N/A - some classes absent)"
                ax.set_title(roc_title)
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                roc_path = self.plot_dir / "roc_curves.png"
                fig.savefig(roc_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                logger.info("Saved ROC Curves plot to %s", roc_path)

            # 3. Precision-Recall Curves
            if pr_curves:
                fig, ax = plt.subplots(figsize=(8, 6))
                for c_name, curve in pr_curves.items():
                    auc_val = metrics.get("per_class_pr_auc", {}).get(c_name, 0.0)
                    ax.plot(curve["recall"], curve["precision"], lw=2, label=f"{c_name} (AUC = {auc_val:.3f})")
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title(f"Precision-Recall Curves (Macro PR-AUC = {metrics['pr_auc_macro']:.3f})")
                ax.legend(loc="lower left")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                pr_path = self.plot_dir / "precision_recall_curves.png"
                fig.savefig(pr_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                logger.info("Saved PR Curves plot to %s", pr_path)

            # 4. Prediction Confidence Histogram
            confidence_vals: List[float] = []
            if y_probs is not None and len(y_probs) > 0:
                confidence_vals = [float(np.max(p)) for p in y_probs]

            if confidence_vals:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(confidence_vals, bins=20, color="#2196F3", edgecolor="white", alpha=0.85)
                ax.set_xlabel("Prediction Confidence (Max Softmax Probability)", fontsize=12)
                ax.set_ylabel("Sample Count", fontsize=12)
                ax.set_title("Prediction Confidence Histogram", fontsize=14, fontweight="bold")
                ax.axvline(float(np.mean(confidence_vals)), color="red", linestyle="--", linewidth=2,
                           label=f"Mean = {np.mean(confidence_vals):.3f}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                conf_path = self.plot_dir / "confidence_histogram.png"
                fig.savefig(conf_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                logger.info("Saved confidence histogram to %s", conf_path)

            # 5. FIX: Class Distribution in Test Set (Pie Chart)
            cm_arr2 = np.array(cm)
            class_totals = cm_arr2.sum(axis=1)  # actual samples per class
            nonzero_mask = class_totals > 0
            if nonzero_mask.sum() > 0:
                fig, ax = plt.subplots(figsize=(7, 7))
                labels_nz = [self.class_labels[i] for i in range(self.num_classes) if nonzero_mask[i]]
                sizes_nz  = class_totals[nonzero_mask]
                ax.pie(sizes_nz, labels=labels_nz, autopct="%1.1f%%",
                       startangle=140, colors=["#4CAF50", "#2196F3", "#FF9800", "#F44336"][:len(labels_nz)])
                ax.set_title("Test Set Class Distribution", fontsize=14, fontweight="bold")
                plt.tight_layout()
                dist_path = self.plot_dir / "test_class_distribution.png"
                fig.savefig(dist_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                logger.info("Saved test class distribution chart to %s", dist_path)

        except Exception as exc:
            logger.warning("Failed to render plot images: %s", exc)

    # --------------------------------------------------------------------------
    # Report Generation (Markdown & HTML)
    # --------------------------------------------------------------------------

    def _generate_reports(self, metrics: Dict[str, Any], cls_report_str: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # 1. Markdown Report
        md_content = f"""# Genomic Sequence Classifier — Evaluation & Inference Report

**Generated:** {timestamp}  
**Execution Duration:** {metrics['execution_duration_sec']:.2f} seconds  
**Total Samples Evaluated:** {metrics['samples_evaluated']}  
**Device:** {self.device}  
**Model Checkpoint:** `{self.checkpoint_path.resolve()}`  

---

## 🏆 Overall Performance Summary

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **{metrics['accuracy']:.4f}** |
| **Macro F1-Score** | **{metrics['macro_f1']:.4f}** |
| **Macro Precision** | {metrics['macro_precision']:.4f} |
| **Macro Recall** | {metrics['macro_recall']:.4f} |
| **ROC-AUC (Macro)** | **{metrics['roc_auc_macro']:.4f}** |
| **PR-AUC (Macro)** | **{metrics['pr_auc_macro']:.4f}** |
| **Matthews Correlation (MCC)** | {metrics['mcc']:.4f} |
| **Cohen's Kappa** | {metrics['cohen_kappa']:.4f} |

---

## 📊 Per-Class Performance Metrics

| Class Name | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
        for c_name in self.class_labels:
            p = metrics['per_class_precision'].get(c_name, 0.0)
            r = metrics['per_class_recall'].get(c_name, 0.0)
            f = metrics['per_class_f1'].get(c_name, 0.0)
            roc = metrics['per_class_roc_auc'].get(c_name, 0.0)
            pr = metrics['per_class_pr_auc'].get(c_name, 0.0)
            md_content += f"| **{c_name}** | {p:.4f} | {r:.4f} | **{f:.4f}** | {roc:.4f} | {pr:.4f} |\n"

        md_content += f"""
---

## 📋 Full Classification Report

```text
{cls_report_str}
```

---

## 🖼 Visualizations
- Confusion Matrix: `plots/confusion_matrix.png`
- ROC Curves: `plots/roc_curves.png`
- Precision-Recall Curves: `plots/precision_recall_curves.png`
"""
        md_path = self.report_dir / "evaluation_report.md"
        md_path.write_text(md_content, encoding="utf-8")
        logger.info("Saved Markdown report to %s", md_path)

        # 2. HTML Report
        rows_html = ""
        for c_name in self.class_labels:
            p = metrics['per_class_precision'].get(c_name, 0.0)
            r = metrics['per_class_recall'].get(c_name, 0.0)
            f = metrics['per_class_f1'].get(c_name, 0.0)
            roc = metrics['per_class_roc_auc'].get(c_name, 0.0)
            pr = metrics['per_class_pr_auc'].get(c_name, 0.0)
            rows_html += f"<tr><td><b>{c_name}</b></td><td>{p:.4f}</td><td>{r:.4f}</td><td><b>{f:.4f}</b></td><td>{roc:.4f}</td><td>{pr:.4f}</td></tr>\n"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Genomic Classifier — Evaluation & Inference Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background-color: #f4f6f9; color: #333; }}
        h1 {{ color: #1e3d59; border-bottom: 3px solid #1e3d59; padding-bottom: 10px; }}
        h2 {{ color: #17b978; margin-top: 30px; }}
        .meta-card {{ background: #ffffff; border-radius: 8px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 25px; }}
        table {{ width: 100%; border-collapse: collapse; background: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-top: 15px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #1e3d59; color: #ffffff; font-weight: 600; }}
        tr:hover {{ background-color: #f1f5f9; }}
        pre {{ background: #2b2d42; color: #edf2f4; padding: 15px; border-radius: 6px; overflow-x: auto; font-size: 14px; }}
    </style>
</head>
<body>
    <h1>Genomic Classifier — Evaluation & Inference Report</h1>
    <div class="meta-card">
        <p><b>Execution Timestamp:</b> {timestamp}</p>
        <p><b>Duration:</b> {metrics['execution_duration_sec']:.2f} seconds</p>
        <p><b>Samples Evaluated:</b> {metrics['samples_evaluated']}</p>
        <p><b>Device:</b> {self.device}</p>
        <p><b>Model Checkpoint:</b> {self.checkpoint_path.name}</p>
    </div>

    <h2>🏆 Overall Performance Summary</h2>
    <table>
        <tr><th>Metric</th><th>Score</th></tr>
        <tr><td>Accuracy</td><td><b>{metrics['accuracy']:.4f}</b></td></tr>
        <tr><td>Macro F1-Score</td><td><b>{metrics['macro_f1']:.4f}</b></td></tr>
        <tr><td>Macro Precision</td><td>{metrics['macro_precision']:.4f}</td></tr>
        <tr><td>Macro Recall</td><td>{metrics['macro_recall']:.4f}</td></tr>
        <tr><td>ROC-AUC (Macro)</td><td><b>{metrics['roc_auc_macro']:.4f}</b></td></tr>
        <tr><td>PR-AUC (Macro)</td><td><b>{metrics['pr_auc_macro']:.4f}</b></td></tr>
        <tr><td>Matthews Correlation Coefficient (MCC)</td><td>{metrics['mcc']:.4f}</td></tr>
        <tr><td>Cohen's Kappa</td><td>{metrics['cohen_kappa']:.4f}</td></tr>
    </table>

    <h2>📊 Per-Class Breakdown</h2>
    <table>
        <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>ROC-AUC</th><th>PR-AUC</th></tr>
        {rows_html}
    </table>

    <h2>📋 Classification Report</h2>
    <pre>{cls_report_str}</pre>
</body>
</html>"""

        html_path = self.report_dir / "evaluation_report.html"
        html_path.write_text(html_content, encoding="utf-8")
        logger.info("Saved HTML report to %s", html_path)
