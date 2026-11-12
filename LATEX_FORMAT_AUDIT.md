# LaTeX Formatting & Layout Audit Report

**Document**: IEEE Conference Research Paper  
**Document Class**: `\documentclass[conference]{IEEEtran}`  
**Project**: Avian Influenza Genomic Intelligence  
**Audit Focus**: IEEE Two-Column Alignment, Table Widths, Float Overlaps, and Margin Symmetry  

---

## 1. Executive Summary
A comprehensive structural and typographical audit was conducted across all LaTeX source files, tables, figures, and environments. The root causes behind horizontal float overflows, column collisions, and margin asymmetry were identified and systematically resolved using native IEEEtran layout rules and `tabularx` column budgeting.

---

## 2. Table-by-Table Root Cause Analysis & Corrections

| Table # | Caption / Subject | Root Cause of Issue | Proposed Correction | Type of Change |
|---|---|---|---|---|
| **Table I** | *Comparison with Representative Studies* | 7 unconstrained `l` columns in a full-width environment (`table*`) caused text to overflow `\textwidth`. | Converted to `tabularx{\textwidth}{@{}l c l X X l X@{}}` with `\footnotesize` and automatic wrapping on methodology, task, and limitations columns. | Layout Only |
| **Table II** | *Pathogenicity Class Assignment* | Long subtype list in column 3 (`lll`) pushed width beyond `\columnwidth`. | Converted to `tabularx{\columnwidth}{@{}c l X@{}}` ensuring the subtype description wraps within the single column. | Layout Only |
| **Table III** | *Dataset Composition Across Pipeline Stages* | 5 numeric/text columns (`lrrrr`) containing large numbers (e.g., `4,782,973,902`) exceeded column boundaries. | Converted to `tabularx{\columnwidth}{@{}X r r r@{}}` with concise header formatting and `\footnotesize`. | Layout Only |
| **Table IV** | *Augmentation Multipliers Across Classes* | 5 unconstrained columns (`lrrrr`) caused horizontal clipping. | Converted to `tabularx{\columnwidth}{@{}X r r c r@{}}` with compact headings (`Before`, `After`, `Mult.`, `Share`). | Layout Only |
| **Table V** | *Dataset Distribution Across Splits and Classes* | 6 unconstrained numeric columns (`lrrrrr`) overflowed into Column 2 by ~25pt. | Re-engineered with `tabularx{\columnwidth}{@{}X r r r r r@{}}`, `\footnotesize`, and compact column headings (`Class`, `Files`, `Train`, `Val`, `Test`, `Total`). | Layout Only |
| **Table VI** | *Training and Optimization Hyperparameters* | Column 2 contained long optimizer/scheduler descriptions that exceeded column margin. | Converted to `tabularx{\columnwidth}{@{}l X@{}}` allowing automatic wrapping in column 2. | Layout Only |
| **Table VII** | *Layer-by-Layer Parameter Distribution* | Long module labels (e.g. `\textbf{Total (GenomicClassificationEngine)}`) collided with Column 2 and references. | Converted to `tabularx{\columnwidth}{@{}X r r@{}}` with wrapped module descriptions and compact summary row `\textbf{Total (Genomic Engine)}`. | Layout Only |
| **Table VIII** | *Hold-Out Test Benchmark Performance* | Metric names (`Matthews Correlation Coefficient...`) exceeded column width in unconstrained `lc` table. | Converted to `tabularx{\columnwidth}{@{}X r@{}}` with clean right-aligned metric values. | Layout Only |
| **Table IX** | *Per-Class Performance Breakdown* | 6 full-length metric columns (`Support`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`) caused severe column collision. | Converted to `tabularx{\columnwidth}{@{}X r r r r r@{}}` with standardized abbreviations (`Supp.`, `Prec.`, `Rec.`, `F1`, `AUC`) and 1 decimal place. | Layout Only |
| **Table X** | *Ablation Study Design Protocol* | Variant names in Column 1 pushed checkboxes into neighboring column. | Converted to `tabularx{\columnwidth}{@{}X c c c c@{}}` with clean text wrapping. | Layout Only |

---

## 3. Figure & Float Audit

| Figure # | Target Path | Layout Mode | Width Constraint | Status |
|---|---|---|---|---|
| **Fig. 1** | `figures/architecture.png` | `figure*` (Span 2 Columns) | `width=0.95\textwidth` | Verified, fits page margins cleanly |
| **Fig. 2** | `figures/pipeline.png` | `figure` (Single Column) | `width=\columnwidth` | Verified, zero column spillover |
| **Fig. 3** | `figures/model_architecture.png` | `figure` (Single Column) | `width=\columnwidth` | Verified, crisp resolution |
| **Fig. 4** | `figures/training_curves.png` | `figure` (Single Column) | `width=\columnwidth` | Verified, 300 DPI |
| **Fig. 5** | `figures/confusion_matrix.png` | `figure` (Single Column) | `width=0.85\columnwidth` | Centered, zero overlap |
| **Fig. 6** | `figures/class_performance.png` | `figure` (Single Column) | `width=\columnwidth` | Verified, crisp text |
| **Fig. 7** | `figures/shap_summary.png` | `figure` (Single Column) | `width=\columnwidth` | Verified, no obstruction of references |

---

## 4. Margin Symmetry and Typography Rules
- **No Manual Margin Manipulation**: Verified absence of `geometry`, `fullpage`, `\textwidth` overrides, or `\oddsidemargin` modifications.
- **Natural Column Balancing**: IEEEtran controls column separation (`\columnsep = 1pc`) and margins (`0.75in` top/bottom, `0.625in` left/right).
- **Float Placement**: Float specifiers standardized to `[t]` (top) and `[!t]` to avoid floating collisions with body paragraphs.
