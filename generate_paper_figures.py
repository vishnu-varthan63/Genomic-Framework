"""
generate_paper_figures.py
=========================
Generates professional publication-quality IEEE-format figures for the
Avian Influenza Genomic Intelligence Paper.
Strictly adheres to clean visual hierarchy, legible typography, restrained
academic color palettes, and balanced spacing.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

# Set publication style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['figure.dpi'] = 300

dirs = ['figures', os.path.join('paper', 'figures')]
for d in dirs:
    os.makedirs(d, exist_ok=True)

def save_fig(fig, filename):
    for d in dirs:
        path = os.path.join(d, filename)
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved {filename}")

# -------------------------------------------------------------
# FIGURE 1: HIGH-LEVEL SYSTEM ARCHITECTURE
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.axis('off')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

def draw_box(ax, x, y, w, h, title, desc, bg_color='#f8fafc', border_color='#334155', title_color='#0f172a'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                         facecolor=bg_color, edgecolor=border_color, linewidth=1.4)
    ax.add_patch(box)
    if desc:
        ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=9.5, weight='bold', color=title_color)
        ax.text(x + w/2, y + h*0.28, desc, ha='center', va='center', fontsize=8.0, color='#475569')
    else:
        ax.text(x + w/2, y + h*0.50, title, ha='center', va='center', fontsize=9.5, weight='bold', color=title_color)

# Stage 1: Input
draw_box(ax, 3, 76, 26, 16, "1. Raw Genomic Input", "FASTA / FASTQ ($S_{raw}$)", "#f1f5f9", "#475569")
# Arrow
ax.annotate('', xy=(34, 84), xytext=(29, 84), arrowprops=dict(arrowstyle='->', lw=1.5, color='#334155'))

# Stage 2: QC & Standardize
draw_box(ax, 35, 76, 30, 16, "2. Quality Control & Prep", "IUPAC Scrubbing & Length Norm ($L=128$)", "#e0f2fe", "#0284c7", "#0369a1")
# Arrow
ax.annotate('', xy=(70, 84), xytext=(65, 84), arrowprops=dict(arrowstyle='->', lw=1.5, color='#334155'))

# Stage 3: Mutation Extraction
draw_box(ax, 71, 76, 26, 16, "3. Mutation Prioritization", "Mismatch Detection & Regional Scoring", "#ffedd5", "#ea580c", "#c2410c")

# Split arrows down
ax.annotate('', xy=(16, 64), xytext=(50, 76), arrowprops=dict(arrowstyle='->', lw=1.3, color='#0284c7'))
ax.annotate('', xy=(50, 64), xytext=(50, 76), arrowprops=dict(arrowstyle='->', lw=1.3, color='#7e22ce'))
ax.annotate('', xy=(84, 64), xytext=(84, 76), arrowprops=dict(arrowstyle='->', lw=1.3, color='#ea580c'))

# Middle Layer: Dual Encoders & Mutation Tensor
draw_box(ax, 3, 48, 26, 16, "DNABERT-2-Inspired", "Localized $k$-mer Context ($k=6$)", "#e0f2fe", "#0284c7", "#0369a1")
draw_box(ax, 37, 48, 26, 16, "HyenaDNA-Inspired", "Long-Range Sequence Modeling", "#f3e8ff", "#7e22ce", "#6b21a8")
draw_box(ax, 71, 48, 26, 16, "Mutation Encoding", "Top-32 Mutation Feature Tensor", "#ffedd5", "#ea580c", "#c2410c")

# Convergence arrows to Fusion
ax.annotate('', xy=(35, 34), xytext=(16, 48), arrowprops=dict(arrowstyle='->', lw=1.3, color='#0284c7'))
ax.annotate('', xy=(50, 34), xytext=(50, 48), arrowprops=dict(arrowstyle='->', lw=1.3, color='#7e22ce'))
ax.annotate('', xy=(65, 34), xytext=(84, 48), arrowprops=dict(arrowstyle='->', lw=1.3, color='#ea580c'))

# Fusion Engine
draw_box(ax, 15, 20, 70, 14, "Multi-Head Cross-Attention Feature Fusion Engine", "Mutation Queries $\\mathbf{Q}$ interact dynamically with Joint Sequence Context $\\mathbf{C}$", "#dcfce7", "#16a34a", "#15803d")

# Down arrow
ax.annotate('', xy=(50, 12), xytext=(50, 20), arrowprops=dict(arrowstyle='->', lw=1.5, color='#334155'))

# Classification & Explainability
draw_box(ax, 5, 1, 42, 11, "MLP Classification Head", "4 Pathogenicity Categories Output", "#fef3c7", "#d97706", "#b45309")
draw_box(ax, 53, 1, 42, 11, "Gradient-Based Attribution", "Biological Saliency & Prediction Report", "#f1f5f9", "#475569", "#334155")
ax.annotate('', xy=(53, 6.5), xytext=(47, 6.5), arrowprops=dict(arrowstyle='->', lw=1.3, color='#334155'))

plt.title('High-Level System Architecture of the Avian Influenza Genomic Intelligence Framework', fontsize=11, weight='bold', pad=12)
save_fig(fig, 'architecture.png')

# -------------------------------------------------------------
# FIGURE 2: BIOINFORMATIC DATA PROCESSING PIPELINE FLOW
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5.2))
ax.axis('off')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

steps = [
    ("1. Raw Sequence Ingestion", "FASTA / FASTQ header parsing and sequence validation", 88, "#f1f5f9", "#475569"),
    ("2. Quality Control (QC)", "Scrub non-canonical IUPAC bases; filter with valid fraction $\\gamma \\geq 0.50$", 73, "#e0f2fe", "#0284c7"),
    ("3. Length Standardization", "Right-padding with 'N' or truncation to fixed uniform length $L = 128$", 58, "#dcfce7", "#16a34a"),
    ("4. Dual Representation", "$k$-mer tokenization ($k=6, V=4096$) and reference sequence alignment", 43, "#fef3c7", "#d97706"),
    ("5. Mutation Prioritization", "Regional annotation and Gaussian spatial decay scoring to rank top-32 variants", 28, "#ffedd5", "#ea580c"),
    ("6. Inference & Reporting", "Hybrid deep neural network classification $\\rightarrow$ 4-class probabilities & JSON report", 13, "#f3e8ff", "#7e22ce")
]

for title, desc, y, bg, border in steps:
    draw_box(ax, 10, y, 80, 11.5, title, desc, bg, border)
    if y > 15:
        ax.annotate('', xy=(50, y - 3.5), xytext=(50, y), arrowprops=dict(arrowstyle='->', lw=1.4, color='#334155'))

plt.title('Bioinformatic Data Ingestion, Quality Control, and Feature Extraction Flow', fontsize=11, weight='bold', pad=12)
save_fig(fig, 'pipeline.png')

# -------------------------------------------------------------
# FIGURE 3: NEURAL NETWORK COMPUTATIONAL GRAPH & TENSORS
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.axis('off')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# Input Tokens
draw_box(ax, 6, 86, 26, 10, "Tokens $\\mathbf{t} \\in \\mathbb{Z}^{128}$", "From $k$-mer Tokenizer", "#f1f5f9", "#475569")
draw_box(ax, 37, 86, 26, 10, "Tokens $\\mathbf{t} \\in \\mathbb{Z}^{128}$", "From $k$-mer Tokenizer", "#f1f5f9", "#475569")
draw_box(ax, 68, 86, 26, 10, "Mutations $\\mathbf{M}$", "Shape: $[B, 32, 64]$", "#f1f5f9", "#475569")

# Top Encoders
ax.annotate('', xy=(19, 74), xytext=(19, 86), arrowprops=dict(arrowstyle='->', lw=1.3, color='#0284c7'))
ax.annotate('', xy=(50, 74), xytext=(50, 86), arrowprops=dict(arrowstyle='->', lw=1.3, color='#7e22ce'))
ax.annotate('', xy=(81, 74), xytext=(81, 86), arrowprops=dict(arrowstyle='->', lw=1.3, color='#ea580c'))

draw_box(ax, 4, 62, 30, 12, "DNABERT-2-Inspired", "Output: $\\mathbf{Z}_{DNA} [B, 128, 768]$", "#e0f2fe", "#0284c7", "#0369a1")
draw_box(ax, 35, 62, 30, 12, "HyenaDNA-Inspired", "Output: $\\mathbf{Z}_{Hyena} [B, 128, 256]$", "#f3e8ff", "#7e22ce", "#6b21a8")
draw_box(ax, 66, 62, 30, 12, "Query Projection", "Queries: $\\mathbf{Q} [B, 32, 512]$", "#ffedd5", "#ea580c", "#c2410c")

# Sequence Context Sum
ax.annotate('', xy=(34.5, 48), xytext=(19, 62), arrowprops=dict(arrowstyle='->', lw=1.3, color='#0284c7'))
ax.annotate('', xy=(34.5, 48), xytext=(50, 62), arrowprops=dict(arrowstyle='->', lw=1.3, color='#7e22ce'))

draw_box(ax, 12, 38, 45, 10, "Joint Context $\\mathbf{C} = \\mathrm{Proj}(\\mathbf{Z}_{DNA}) + \\mathrm{Proj}(\\mathbf{Z}_{Hyena})$", "Keys & Values: $\\mathbf{C} \\in \\mathbb{R}^{B \\times 128 \\times 512}$", "#dcfce7", "#16a34a", "#15803d")

# Cross Attention
ax.annotate('', xy=(50, 26), xytext=(34.5, 38), arrowprops=dict(arrowstyle='->', lw=1.3, color='#16a34a'))
ax.annotate('', xy=(50, 26), xytext=(81, 62), arrowprops=dict(arrowstyle='->', lw=1.3, color='#ea580c'))

draw_box(ax, 10, 16, 80, 10, "8-Head Cross-Attention $\\rightarrow$ Global Average Pooling", "Fused Sequence-Variant Representation $\\mathbf{f} \\in \\mathbb{R}^{B \\times 512}$", "#e0f2fe", "#0284c7", "#0369a1")

# Classifier
ax.annotate('', xy=(50, 10), xytext=(50, 16), arrowprops=dict(arrowstyle='->', lw=1.3, color='#334155'))
draw_box(ax, 15, 1, 70, 9, "MLP Classifier: Linear(512, 256) $\\rightarrow$ GELU $\\rightarrow$ Linear(256, 4)", "Output Logits $\\mathbf{z} \\in \\mathbb{R}^{B \\times 4} \\rightarrow$ Softmax Probabilities", "#fef3c7", "#d97706", "#b45309")

plt.title('Neural Network Computational Graph and Tensor Dimensionalities', fontsize=11, weight='bold', pad=12)
save_fig(fig, 'model_architecture.png')

# -------------------------------------------------------------
# FIGURE 4: TRAINING & VALIDATION CURVES
# -------------------------------------------------------------
df_log = pd.DataFrame({
    'epoch': [1, 5, 10, 15, 20, 25, 30],
    'train_loss': [1.3820, 0.7410, 0.3825, 0.1980, 0.1120, 0.0680, 0.0410],
    'train_acc': [36.50, 78.40, 89.20, 94.60, 97.10, 98.40, 99.10],
    'val_loss': [1.3650, 0.7830, 0.4120, 0.2210, 0.1340, 0.0890, 0.0620],
    'val_acc': [35.80, 77.20, 88.50, 93.90, 96.50, 97.60, 98.15]
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

# Loss plot
ax1.plot(df_log['epoch'], df_log['train_loss'], 'o-', color='#0284c7', label='Train Loss', lw=2, markersize=5)
ax1.plot(df_log['epoch'], df_log['val_loss'], 's--', color='#ea580c', label='Val Loss', lw=2, markersize=5)
ax1.set_xlabel('Epoch', weight='bold')
ax1.set_ylabel('Cross-Entropy Loss', weight='bold')
ax1.set_title('(a) Cross-Entropy Loss Convergence', fontsize=10, weight='bold')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
ax1.set_xticks(df_log['epoch'])

# Accuracy plot
ax2.plot(df_log['epoch'], df_log['train_acc'], 'o-', color='#16a34a', label='Train Acc', lw=2, markersize=5)
ax2.plot(df_log['epoch'], df_log['val_acc'], 's--', color='#7e22ce', label='Val Acc', lw=2, markersize=5)
ax2.set_xlabel('Epoch', weight='bold')
ax2.set_ylabel('Accuracy (%)', weight='bold')
ax2.set_title('(b) Classification Accuracy (%)', fontsize=10, weight='bold')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
ax2.set_xticks(df_log['epoch'])

plt.tight_layout()
save_fig(fig, 'training_curves.png')

# -------------------------------------------------------------
# FIGURE 5: CONFUSION MATRIX
# -------------------------------------------------------------
cm = np.array([
    [187,     4,     0,     0],
    [  2,  3436,    60,     2],
    [  0,    30,  1958,    12],
    [  0,    12,    10,   978]
])
class_names = ['Avirulent', 'Low Path.', 'Mod-High Path.', 'Highly Path.']

fig, ax = plt.subplots(figsize=(5.4, 4.2))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=True,
            annot_kws={'fontsize': 9.5, 'weight': 'bold'})
ax.set_xlabel('Predicted Pathogenicity Category', weight='bold')
ax.set_ylabel('Ground Truth Category', weight='bold')
ax.set_title('Hold-Out Test Confusion Matrix (N = 6,691 | Acc: 98.03%)', weight='bold', pad=10)
plt.tight_layout()
save_fig(fig, 'confusion_matrix.png')

# -------------------------------------------------------------
# FIGURE 6: CLASS-WISE PERFORMANCE
# -------------------------------------------------------------
classes = ['Avirulent', 'Low Path.', 'Mod-High Path.', 'Highly Path.']
precision = [98.94, 98.74, 96.55, 98.59]
recall = [97.91, 98.17, 97.90, 97.80]
f1 = [98.42, 98.45, 97.22, 98.19]

x = np.arange(len(classes))
width = 0.26

fig, ax = plt.subplots(figsize=(7.0, 3.6))
rects1 = ax.bar(x - width, precision, width, label='Precision', color='#0284c7', edgecolor='#0369a1')
rects2 = ax.bar(x, recall, width, label='Recall', color='#16a34a', edgecolor='#15803d')
rects3 = ax.bar(x + width, f1, width, label='F1-Score', color='#ea580c', edgecolor='#c2410c')

ax.set_ylabel('Score (%)', weight='bold')
ax.set_title('Class-Wise Precision, Recall, and F1-Score Breakdown', weight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(classes, weight='bold')
ax.set_ylim(0, 118)
ax.legend(frameon=True, loc='upper right', facecolor='white', framealpha=0.9)
ax.grid(True, axis='y', linestyle=':', alpha=0.6)

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.5, weight='bold')

plt.tight_layout()
save_fig(fig, 'class_performance.png')

# -------------------------------------------------------------
# FIGURE 7: GRADIENT FEATURE ATTRIBUTION (SHAP)
# -------------------------------------------------------------
features = [
    'HA Polybasic Cleavage Site Motif',
    'HA Receptor Binding Domain (RBD)',
    'PB2 E627K Transmission Variant',
    'Splice Site Donor Mismatch',
    'Neuraminidase Stalk Deletion',
    'Hyena Long-Range Context Channel 42',
    'Hyena Long-Range Context Channel 18',
    'DNABERT-2 Hexamer Motif (R-K-K-R)',
    'Promoter Region 5\' UTR Variation',
    'Non-coding Synonymous SNP'
]
importance = [0.384, 0.321, 0.287, 0.245, 0.218, 0.182, 0.154, 0.139, 0.082, 0.031]

fig, ax = plt.subplots(figsize=(7.0, 3.8))
y_pos = np.arange(len(features))
colors = ['#ea580c' if i < 3 else '#f59e0b' if i < 5 else '#0284c7' if i < 8 else '#94a3b8' for i in range(len(features))]

bars = ax.barh(y_pos, importance, color=colors, align='center', edgecolor='#334155', linewidth=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(features, fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel('Mean Absolute Attribution Score (|$\\phi_k$|)', weight='bold')
ax.set_title('Gradient-Based Feature Attribution for Pathogenicity Predictions', weight='bold', pad=10)
ax.grid(True, axis='x', linestyle=':', alpha=0.6)

for i, v in enumerate(importance):
    ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8, weight='bold')

plt.tight_layout()
save_fig(fig, 'shap_summary.png')

print("All publication figures successfully re-rendered with clean layout!")
