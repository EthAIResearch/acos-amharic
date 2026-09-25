#!/usr/bin/env python3
"""
Generate Presentation Visual Figures for Amharic ACOS Research Slides
"""
import os
import sys

# Ensure local user site-packages are accessible
sys.path.insert(0, '/home/codeknight/.local/lib/python3.14/site-packages')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs("docs/figures", exist_ok=True)

# Set style with default Latin font and Ethiopic fallback
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Noto Sans Ethiopic']
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8

# ==============================================================================
# Figure 1: Pipeline Error Compounding Cascade
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 3.8), dpi=300)
stages = ['Span Extraction\n(ATE & OTE)', 'Candidate\nPairing', 'Category\nClassification', 'Sentiment\nClassification', 'End-to-End\nFull Quad']
accuracies = [50.6, 68.0, 71.0, 88.0, 14.2]
cumulative = [50.6, 34.4, 24.4, 21.5, 14.2]

x = np.arange(len(stages))
width = 0.35

rects1 = ax.bar(x - width/2, accuracies, width, label='Stage-Isolated F1 (%)', color='#94A3B8', edgecolor='#64748B')
rects2 = ax.bar(x + width/2, cumulative, width, label='Compound Retained F1 (%)', color='#EF4444', edgecolor='#B91C1C')

ax.set_ylabel('F1 Score / Retention (%)', fontsize=10, fontweight='bold', color='#1E293B')
ax.set_title(r'The Error Compounding Law in 6-Stage Pipeline ($F1_{end} \approx \prod F1_i$)', fontsize=11, fontweight='bold', color='#0F172A', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=8.5, color='#334155')
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#E2E8F0', fontsize=8.5)

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='#B91C1C')

plt.tight_layout()
fig.savefig('docs/figures/fig1_pipeline_cascade.png')
plt.close(fig)

# ==============================================================================
# Figure 2: SDRN vs Span-ASTE Ceiling Recall
# ==============================================================================
fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=300)
categories = ['Candidate Ceiling Recall\n(Upper Bound)', 'Actual Captured Pairs\n(Default Threshold)', 'Actual Captured Triplets\n(ASTE F1)']
sdrn_vals = [33.04, 34.85, 24.10]
aste_vals = [81.10, 30.80, 27.09]
run2_vals = [80.46, 34.02, 29.37]
run3_vals = [81.91, 45.78, 28.13]

x = np.arange(len(categories))
w = 0.20

ax.bar(x - 1.5*w, sdrn_vals, w, label='Joint SDRN (BiLSTM+CRF)', color='#94A3B8', edgecolor='#64748B')
ax.bar(x - 0.5*w, aste_vals, w, label='Span-ASTE Run 1 (MLP)', color='#38BDF8', edgecolor='#0284C7')
ax.bar(x + 0.5*w, run2_vals, w, label='Span-ASTE Run 2 (Biaffine)', color='#3B82F6', edgecolor='#1D4ED8')
ax.bar(x + 1.5*w, run3_vals, w, label='Span-ASTE Run 3 (MeanPool+Focal)', color='#10B981', edgecolor='#047857')

ax.set_ylabel('Percentage / Recall (%)', fontsize=9.5, fontweight='bold', color='#1E293B')
ax.set_title('Evolution of Candidate Ceiling Recall & Pair Conversion', fontsize=11, fontweight='bold', color='#0F172A', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9, color='#334155')
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#E2E8F0', fontsize=8)

plt.tight_layout()
fig.savefig('docs/figures/fig2_ceiling_recall.png')
plt.close(fig)

# ==============================================================================
# Figure 3: Deep Biaffine Bilinear Attention Math
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 3.4), dpi=300)
ax.axis('off')

# Draw schematic boxes for Biaffine
# Aspect subspace
box_aspect = patches.FancyBboxPatch((0.05, 0.60), 0.22, 0.25, boxstyle="round,pad=0.03", fc="#DBEAFE", ec="#2563EB", lw=1.5)
ax.add_patch(box_aspect)
ax.text(0.16, 0.725, "Target Span $s_t$\n$h_t = \\text{MLP}_t(s_t)$\n$\\in \\mathbb{R}^{d_{biaf}}$", ha='center', va='center', fontsize=9, fontweight='bold', color='#1E3A8A')

# Opinion subspace
box_opn = patches.FancyBboxPatch((0.05, 0.15), 0.22, 0.25, boxstyle="round,pad=0.03", fc="#FEF3C7", ec="#D97706", lw=1.5)
ax.add_patch(box_opn)
ax.text(0.16, 0.275, "Opinion Span $s_o$\n$h_o = \\text{MLP}_o(s_o)$\n$\\in \\mathbb{R}^{d_{biaf}}$", ha='center', va='center', fontsize=9, fontweight='bold', color='#92400E')

# Bilinear Tensor Box
box_u = patches.FancyBboxPatch((0.38, 0.42), 0.26, 0.42, boxstyle="round,pad=0.03", fc="#F1F5F9", ec="#475569", lw=1.5)
ax.add_patch(box_u)
ax.text(0.51, 0.68, "Bilinear Scoring Tensor $\\mathbf{U}$", ha='center', va='center', fontsize=9, fontweight='bold', color='#0F172A')
ax.text(0.51, 0.52, r"$S_{ij,c}^{bilinear} = h_i^t \mathbf{U}_c h_j^o$" + "\n" + r"$\in \mathbb{R}^{C \times N_t \times N_o}$", ha='center', va='center', fontsize=8.5, color='#334155')

# Affine Context Sync Box
box_aff = patches.FancyBboxPatch((0.38, 0.08), 0.26, 0.28, boxstyle="round,pad=0.03", fc="#E0F2FE", ec="#0284C7", lw=1.5)
ax.add_patch(box_aff)
ax.text(0.51, 0.25, "Cross-Span Sync Context", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0369A1')
ax.text(0.51, 0.15, r"$\tilde{h}_t = \mathrm{softmax}(A) h_o$" + "\n" + r"$\mathrm{MLP}_{aff}([h_t; h_o; \tilde{h}_t; \dots])$", ha='center', va='center', fontsize=8, color='#0284C7')

# Output Box
box_out = patches.FancyBboxPatch((0.74, 0.30), 0.22, 0.40, boxstyle="round,pad=0.03", fc="#DCFCE7", ec="#16A34A", lw=1.5)
ax.add_patch(box_out)
ax.text(0.85, 0.55, "Relation Logits", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#14532D')
ax.text(0.85, 0.42, "$S = S^{bilin} + S^{aff}$\nSoftmax over 4 classes:\n[INV, POS, NEG, NEU]", ha='center', va='center', fontsize=8, color='#166534')

# Connect arrows
ax.annotate("", xy=(0.38, 0.68), xytext=(0.27, 0.725), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2563EB"))
ax.annotate("", xy=(0.38, 0.55), xytext=(0.27, 0.275), arrowprops=dict(arrowstyle="->", lw=1.5, color="#D97706"))
ax.annotate("", xy=(0.38, 0.22), xytext=(0.27, 0.725), arrowprops=dict(arrowstyle="->", lw=1.2, color="#0284C7", linestyle="--"))
ax.annotate("", xy=(0.38, 0.18), xytext=(0.27, 0.275), arrowprops=dict(arrowstyle="->", lw=1.2, color="#0284C7", linestyle="--"))
ax.annotate("", xy=(0.74, 0.55), xytext=(0.64, 0.60), arrowprops=dict(arrowstyle="->", lw=1.5, color="#475569"))
ax.annotate("", xy=(0.74, 0.45), xytext=(0.64, 0.22), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0284C7"))

ax.set_title("Deep Biaffine Attention & Cross-Span Synchronization (Dozat 2017, Nguyen 2018)", fontsize=10.5, fontweight='bold', color='#0F172A', pad=8)
plt.tight_layout()
fig.savefig('docs/figures/fig3_biaffine_tensor.png')
plt.close(fig)

# ==============================================================================
# Figure 4: Prefix Sum Span Mean Pooling
# ==============================================================================
fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=300)
ax.axis('off')
ax.set_ylim(-0.02, 1.05)
ax.set_xlim(0.0, 1.0)

# Tokens
tokens = ["ምግቡ", "በጣም", "ጥሩ", "አይደለም", "ግን", "አገልግሎቱ"]
x_coords = np.linspace(0.08, 0.88, len(tokens))

for i, (tok, xc) in enumerate(zip(tokens, x_coords)):
    rect = patches.FancyBboxPatch((xc - 0.05, 0.52), 0.10, 0.28, boxstyle="round,pad=0.02", fc="#F8FAFC", ec="#64748B", lw=1.2)
    ax.add_patch(rect)
    ax.text(xc, 0.69, f"$x_{i}$", ha='center', va='center', fontsize=9, fontweight='bold', color='#0F172A')
    ax.text(xc, 0.59, tok, ha='center', va='center', fontsize=8, color='#334155', fontfamily='Noto Sans Ethiopic')

# Span highlighting for [1, 4): "በጣም ጥሩ አይደለም"
span_rect = patches.FancyBboxPatch((x_coords[1] - 0.06, 0.47), x_coords[3] - x_coords[1] + 0.12, 0.38,
                                   boxstyle="round,pad=0.03", fc="none", ec="#EF4444", lw=2, linestyle="--")
ax.add_patch(span_rect)
ax.text((x_coords[1] + x_coords[3])/2, 0.90, r"Multi-Word Opinion Span $[s=1, e=4)$", ha='center', va='center', fontsize=9, fontweight='bold', color='#B91C1C')

# Prefix sum formula
formula_box = patches.FancyBboxPatch((0.12, 0.06), 0.76, 0.30, boxstyle="round,pad=0.03", fc="#EFF6FF", ec="#3B82F6", lw=1.2)
ax.add_patch(formula_box)
ax.text(0.50, 0.24, r"$O(1)$ Vectorized Cumulative Prefix-Sum Mean Pooling", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1E40AF')
ax.text(0.50, 0.12, r"$h_{\mathrm{mean}} = \frac{\mathbf{prefix}[e] - \mathbf{prefix}[s]}{e - s} \rightarrow h_{\mathrm{span}} = [x_{\mathrm{start}}; x_{\mathrm{end}}; h_{\mathrm{mean}}; f_{\mathrm{width}}] \in \mathbb{R}^{2324}$",
        ha='center', va='center', fontsize=8.5, color='#1E3A8A')

ax.set_title("Eliminating Multi-Word Boundary Truncation via Vectorized Span Pooling", fontsize=10.5, fontweight='bold', color='#0F172A', pad=8)
plt.tight_layout()
fig.savefig('docs/figures/fig4_prefix_meanpool.png')
plt.close(fig)

# ==============================================================================
# Figure 5: Comprehensive Cross-Model Benchmark Comparison
# ==============================================================================
fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=300)
models = ['Pipeline\n(6-Stage)', 'Joint SDRN\n(BiLSTM+CRF)', 'Span-ASTE\nRun 1 (MLP)', 'Span-ASTE\nRun 2 (Biaf)', 'Span-ASTE\nRun 3 (Pool)', 'ByT5-Base\n(Dev Ep 9)', 'ByT5-Base\n(Final Test)']
pair_f1 = [14.2, 34.85, 32.06, 33.31, 32.04, 29.51, 26.69]
ate_f1 = [50.0, 48.0, 50.59, 53.76, 55.20, 30.3, 29.94]   # ACSE/ATE
full_quad_f1 = [14.2, 0.0, 0.0, 0.0, 0.0, 17.90, 16.14]  # Exact full quad
implicit_f1 = [0.0, 0.0, 0.0, 0.0, 0.0, 19.28, 14.87]    # Implicit quad

x = np.arange(len(models))
w = 0.20

ax.bar(x - 1.5*w, ate_f1, w, label='Aspect F1 (ATE / ACSE)', color='#93C5FD', edgecolor='#2563EB')
ax.bar(x - 0.5*w, pair_f1, w, label='Pair F1 (AOPE)', color='#3B82F6', edgecolor='#1D4ED8')
ax.bar(x + 0.5*w, full_quad_f1, w, label='Full Quad F1 (A,C,S,O)', color='#10B981', edgecolor='#047857')
ax.bar(x + 1.5*w, implicit_f1, w, label='Implicit Quad F1', color='#F59E0B', edgecolor='#B45309')

ax.set_ylabel('F1 Score (%)', fontsize=9.5, fontweight='bold', color='#1E293B')
ax.set_title('Cross-Model Progression: From Extractive Plateau to Generative ByT5 Breakthrough', fontsize=10.5, fontweight='bold', color='#0F172A', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=7.5, color='#334155')
ax.set_ylim(0, 70)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#E2E8F0', fontsize=7.5, loc='upper left')

# Highlight ByT5 final test
ax.annotate('Final Test: 274 Implicit\nQuad TPs (14.87% F1)',
            xy=(6 + 1.5*w, 14.87), xytext=(4.5, 50),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#B45309"),
            fontsize=7.5, fontweight='bold', color='#B45309',
            bbox=dict(boxstyle="round,pad=0.3", fc="#FEF3C7", ec="#F59E0B", lw=1))

plt.tight_layout()
fig.savefig('docs/figures/fig5_cross_model_benchmark.png')
plt.close(fig)

print("All 5 presentation visual figures generated successfully!")
