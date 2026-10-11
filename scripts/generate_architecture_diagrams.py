#!/usr/bin/env python3
"""
Generate publication-quality, detailed architecture diagrams for:
1. SDRN (Synchronous Double-Channel Recurrent Network)
2. Span-ASTE (Span-Level Interaction Learning)
3. ByT5-Base (Byte-Level Generative ACOS)

Designed specifically for 16:9 widescreen presentation geometry (12.0" x 4.15")
to guarantee zero vertical overlap with slide titles or takeaway banners.
"""

import os
import sys

sys.path.insert(0, '/home/codeknight/.local/lib/python3.14/site-packages')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("docs/figures", exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Noto Sans Ethiopic']
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8


# ==============================================================================
# 1. SDRN COMPLETE ARCHITECTURE FLOWCHART
# ==============================================================================
def generate_sdrn_diagram():
    # 12.0 x 4.15 inches matches exact slide content viewport
    fig, ax = plt.subplots(figsize=(12.0, 4.15), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 4.15)

    # Background canvas
    bg = patches.Rectangle((0, 0), 12.0, 4.15, fc='#F8FAFC', ec='none')
    ax.add_patch(bg)

    # 1. Bottom: Input & Context Encoder
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.15), 11.0, 0.55, boxstyle="round,pad=0.03", fc="#E2E8F0", ec="#64748B", lw=1.2))
    ax.text(6.0, 0.47, "Input Sentence X = [x_1, x_2, ..., x_N] (Amharic Civic Commentary)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0F172A')
    ax.text(6.0, 0.27, r"AfroXLMR Context Encoder $\rightarrow$ Contextual Embeddings $H \in \mathbb{R}^{N \times d}$ ($d=768$) with Subword Pooling", ha='center', va='center', fontsize=8.5, color='#334155')

    # Connecting arrows from encoder to channels
    ax.annotate("", xy=(3.0, 0.95), xytext=(3.0, 0.70), arrowprops=dict(arrowstyle="->", lw=1.8, color="#0284C7"))
    ax.annotate("", xy=(9.0, 0.95), xytext=(9.0, 0.70), arrowprops=dict(arrowstyle="->", lw=1.8, color="#EA580C"))

    # 2. Left Channel: Entity Channel
    ent_container = patches.FancyBboxPatch((0.5, 0.95), 5.2, 2.30, boxstyle="round,pad=0.04", fc="#EFF6FF", ec="#3B82F6", lw=1.5, linestyle="--")
    ax.add_patch(ent_container)
    ax.text(3.1, 3.10, "ENTITY CHANNEL (Span Tagging)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1E40AF')

    # Entity components inside
    ax.add_patch(patches.FancyBboxPatch((0.7, 2.35), 4.8, 0.55, boxstyle="round,pad=0.02", fc="#DBEAFE", ec="#2563EB", lw=1.0))
    ax.text(3.1, 2.68, "BiLSTM Recurrent Layer", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E3A8A')
    ax.text(3.1, 2.48, r"$h_i^{e,(t)} = \mathrm{BiLSTM}(h_i^{e,(t-1)} + \Delta_{RSM,i}^{(t)})$", ha='center', va='center', fontsize=8.0, color='#1E40AF')

    ax.add_patch(patches.FancyBboxPatch((0.7, 1.65), 4.8, 0.55, boxstyle="round,pad=0.02", fc="#BFDBFE", ec="#1D4ED8", lw=1.0))
    ax.text(3.1, 1.98, "Linear-Chain 5-Way CRF Layer", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E3A8A')
    ax.text(3.1, 1.78, r"$\mathrm{Score}(X, y) = \sum E_{i,y_i} + \sum T_{y_i, y_{i+1}} \rightarrow \{O, B\text{-}ASP, I\text{-}ASP, B\text{-}OPN, I\text{-}OPN\}$", ha='center', va='center', fontsize=7.5, color='#1E3A8A')

    ax.add_patch(patches.FancyBboxPatch((0.7, 1.05), 4.8, 0.48, boxstyle="round,pad=0.02", fc="#93C5FD", ec="#1E40AF", lw=1.0))
    ax.text(3.1, 1.33, "Entity Softmax Distribution", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0F172A')
    ax.text(3.1, 1.15, r"$p_i = \mathrm{Softmax}(W_e h_i^e) \in \mathbb{R}^5$", ha='center', va='center', fontsize=8.0, color='#1E3A8A')

    # 3. Right Channel: Relation Channel
    rel_container = patches.FancyBboxPatch((6.3, 0.95), 5.2, 2.30, boxstyle="round,pad=0.04", fc="#FFF7ED", ec="#F97316", lw=1.5, linestyle="--")
    ax.add_patch(rel_container)
    ax.text(8.9, 3.10, "RELATION CHANNEL (Pair Matrix)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#9A3412')

    # Relation components inside
    ax.add_patch(patches.FancyBboxPatch((6.5, 2.35), 4.8, 0.55, boxstyle="round,pad=0.02", fc="#FED7AA", ec="#EA580C", lw=1.0))
    ax.text(8.9, 2.68, "RelLSTM Recurrent Cell", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#7C2D12')
    ax.text(8.9, 2.48, r"$r_{ij}^{(t)} = \mathrm{RelLSTM}(r_{ij}^{(t-1)} + B_{ESM,ij}^{(t)})$", ha='center', va='center', fontsize=8.0, color='#9A3412')

    ax.add_patch(patches.FancyBboxPatch((6.5, 1.65), 4.8, 0.55, boxstyle="round,pad=0.02", fc="#FDBA74", ec="#C2410C", lw=1.0))
    ax.text(8.9, 1.98, r"$N \times N$ Correlation Scoring Matrix", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#7C2D12')
    ax.text(8.9, 1.78, r"$P(rel_{ij}=1) = \sigma(W_{rel} r_{ij} + b_{rel}) \quad (\text{Threshold } \hat{\delta} = 0.20)$", ha='center', va='center', fontsize=7.5, color='#7C2D12')

    ax.add_patch(patches.FancyBboxPatch((6.5, 1.05), 4.8, 0.48, boxstyle="round,pad=0.02", fc="#FB923C", ec="#9A3412", lw=1.0))
    ax.text(8.9, 1.33, "Pair Feature Construction", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#431407')
    ax.text(8.9, 1.15, r"$r_{ij}^{(0)} = [h_i ; h_j ; |h_i - h_j| ; h_i \odot h_j] \in \mathbb{R}^{4d}$", ha='center', va='center', fontsize=8.0, color='#431407')

    # 4. Synchronous Recurrence Feedback Arrows
    # ESM: Entity -> Relation (Red arrow)
    ax.annotate("", xy=(6.5, 2.50), xytext=(5.5, 1.25),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="#DC2626", connectionstyle="arc3,rad=-0.20"))
    ax.text(6.0, 1.95, r"$\mathbf{ESM:}$ $M_{ij} = p_i \otimes p_j \rightarrow B_{ESM,ij}$",
            ha='center', va='center', fontsize=7.5, fontweight='bold', color="#DC2626",
            bbox=dict(boxstyle="round,pad=0.15", fc="#FEE2E2", ec="#DC2626", lw=0.8))

    # RSM: Relation -> Entity (Green arrow)
    ax.annotate("", xy=(5.5, 2.55), xytext=(6.5, 1.90),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="#16A34A", connectionstyle="arc3,rad=-0.20"))
    ax.text(6.0, 2.80, r"$\mathbf{RSM:}$ $\Delta_{RSM,i} = W_{rsm} \sum_j \alpha_{ij} h_j^e$",
            ha='center', va='center', fontsize=7.5, fontweight='bold', color="#16A34A",
            bbox=dict(boxstyle="round,pad=0.15", fc="#DCFCE7", ec="#16A34A", lw=0.8))

    # 5. Top Output: Joint Extraction
    ax.add_patch(patches.FancyBboxPatch((2.5, 3.45), 7.0, 0.55, boxstyle="round,pad=0.03", fc="#FEF08A", ec="#CA8A04", lw=1.2))
    ax.text(6.0, 3.77, r"JOINT AOPE OUTPUT: $\mathcal{S}_{pair} = \{(a_k, o_k)\}$", ha='center', va='center', fontsize=10, fontweight='bold', color='#713F12')
    ax.text(6.0, 3.58, "Decoded Spans (CRF) aligned with Accepted Relation Matrix Edges (R >= 0.20)", ha='center', va='center', fontsize=8.5, color='#854D0E')

    ax.annotate("", xy=(4.5, 3.45), xytext=(3.1, 3.25), arrowprops=dict(arrowstyle="->", lw=1.5, color="#CA8A04"))
    ax.annotate("", xy=(7.5, 3.45), xytext=(8.9, 3.25), arrowprops=dict(arrowstyle="->", lw=1.5, color="#CA8A04"))

    plt.tight_layout(pad=0.1)
    fig.savefig('docs/figures/sdrn_architecture_flow.png', bbox_inches='tight')
    plt.close(fig)
    print("Generated docs/figures/sdrn_architecture_flow.png")


# ==============================================================================
# 2. SPAN-ASTE COMPLETE ARCHITECTURE FLOWCHART
# ==============================================================================
def generate_span_aste_diagram():
    fig, ax = plt.subplots(figsize=(12.0, 4.15), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 4.15)

    # Background canvas
    bg = patches.Rectangle((0, 0), 12.0, 4.15, fc='#F8FAFC', ec='none')
    ax.add_patch(bg)

    # 1. Word Encoder & Input
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.15), 11.0, 0.50, boxstyle="round,pad=0.03", fc="#E2E8F0", ec="#64748B", lw=1.2))
    ax.text(6.0, 0.45, "Input Words $X = [w_1, ..., w_N]$ $\\rightarrow$ Pretrained AfroXLMR Encoder", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0F172A')
    ax.text(6.0, 0.27, r"Word Representations $H = [h_1, ..., h_N] \in \mathbb{R}^{N \times d}$ ($d=768$) via Subword Mean Pooling", ha='center', va='center', fontsize=8.0, color='#334155')

    ax.annotate("", xy=(6.0, 0.85), xytext=(6.0, 0.65), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0284C7"))

    # 2. Candidate Span Enumeration & Prefix Pooling
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.85), 11.0, 0.68, boxstyle="round,pad=0.03", fc="#EFF6FF", ec="#3B82F6", lw=1.2))
    ax.text(6.0, 1.34, r"SPAN CANDIDATE ENUMERATION & $O(1)$ PREFIX-SUM POOLING", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1E40AF')
    ax.text(6.0, 1.14, r"Enumerate all contiguous spans $\mathcal{S} = \{ [i, j) \mid 1 \leq i < j \leq N, j - i \leq L=8 \}$ (Reachable gold pairs: 81.91%)", ha='center', va='center', fontsize=8.0, color='#1E3A8A')
    ax.text(6.0, 0.96, r"$e_s = [h_i ; h_{j-1} ; \mathrm{MeanPool}(i, j) ; w_{j-i}] \in \mathbb{R}^{2324}$ $\quad \vert \quad$ $\mathrm{MeanPool}(i, j) = \frac{P_j - P_i}{j - i}$ in $O(1)$ time", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#1D4ED8')

    ax.annotate("", xy=(6.0, 1.75), xytext=(6.0, 1.53), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0284C7"))

    # 3. Dual Mention Pruning
    ax.add_patch(patches.FancyBboxPatch((0.5, 1.75), 11.0, 0.62, boxstyle="round,pad=0.03", fc="#FEF3C7", ec="#D97706", lw=1.2))
    ax.text(6.0, 2.21, "DUAL MENTION PRUNING (TOP-K CANDIDATE FILTERING)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#92400E')
    ax.text(6.0, 2.01, r"Aspect Scorer: $s_{asp}(s) = \mathbf{w}_{asp}^T \mathrm{ReLU}(W_{asp} e_s)$ $\quad \vert \quad$ Opinion Scorer: $s_{opn}(s) = \mathbf{w}_{opn}^T \mathrm{ReLU}(W_{opn} e_s)$", ha='center', va='center', fontsize=8.0, color='#78350F')
    ax.text(6.0, 1.84, r"Retain Top $K_a = \lambda_a N$ aspect candidates and $K_o = \lambda_o N$ opinion candidates $\rightarrow$ Prunes $>85\%$ of spurious pairs", ha='center', va='center', fontsize=7.5, color='#B45309')

    ax.annotate("", xy=(6.0, 2.58), xytext=(6.0, 2.37), arrowprops=dict(arrowstyle="->", lw=1.5, color="#D97706"))

    # 4. Deep Biaffine Bilinear Tensor Scoring
    ax.add_patch(patches.FancyBboxPatch((0.5, 2.58), 11.0, 0.82, boxstyle="round,pad=0.03", fc="#F0FDF4", ec="#16A34A", lw=1.2))
    ax.text(6.0, 3.23, "DEEP BIAFFINE RELATION & SENTIMENT INTERACTION (RUN 2 & 3)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#166534')
    ax.text(6.0, 3.03, r"Bilinear Scoring Tensor: $S(a, o)_c = {e_a'}^T \mathbf{U}_c e_o' + \mathbf{W}_c [e_a' ; e_o' ; d(a, o) ; c_{ctx}] + b_c$", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#14532D')
    ax.text(6.0, 2.84, r"$\mathbf{U}_c$: Bilinear tensor measuring directional compatibility $\quad \vert \quad$ $d(a, o)$: Span distance $\quad \vert \quad$ $c_{ctx}$: Intervening tokens", ha='center', va='center', fontsize=7.5, color='#15803D')
    ax.text(6.0, 2.68, r"Cost-Sensitive Loss $\mathcal{L}_{rel}$ with class weights $\alpha = [0.1, 1.0, 1.0, 2.0]$ for [INVALID, POS, NEG, NEU]", ha='center', va='center', fontsize=7.5, color='#166534')

    ax.annotate("", xy=(6.0, 3.58), xytext=(6.0, 3.40), arrowprops=dict(arrowstyle="->", lw=1.5, color="#16A34A"))

    # 5. Output: Triplets
    ax.add_patch(patches.FancyBboxPatch((1.5, 3.58), 9.0, 0.48, boxstyle="round,pad=0.03", fc="#DCFCE7", ec="#15803D", lw=1.2))
    ax.text(6.0, 3.88, r"EXTRACTED ASTE TRIPLETS: $\mathcal{T} = \{(a_k, o_k, s_k)\}$", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#14532D')
    ax.text(6.0, 3.70, r"Non-INVALID pairs emit (Aspect Span, Opinion Span, Sentiment Polarity $\in \{\mathrm{POS}, \mathrm{NEG}, \mathrm{NEU}\}$)", ha='center', va='center', fontsize=8.0, color='#166534')

    plt.tight_layout(pad=0.1)
    fig.savefig('docs/figures/span_aste_architecture_flow.png', bbox_inches='tight')
    plt.close(fig)
    print("Generated docs/figures/span_aste_architecture_flow.png")


# ==============================================================================
# 3. BYT5-BASE COMPLETE ARCHITECTURE FLOWCHART
# ==============================================================================
def generate_byt5_diagram():
    fig, ax = plt.subplots(figsize=(12.0, 4.15), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 4.15)

    # Background canvas
    bg = patches.Rectangle((0, 0), 12.0, 4.15, fc='#F8FAFC', ec='none')
    ax.add_patch(bg)

    # 1. Input: Raw UTF-8 Bytes
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.15), 11.0, 0.50, boxstyle="round,pad=0.03", fc="#E2E8F0", ec="#64748B", lw=1.2))
    ax.text(6.0, 0.45, "Raw Amharic Sentence $\\rightarrow$ UTF-8 Byte Stream $[b_1, b_2, ..., b_M]$ (Values $0 \\dots 255$)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0F172A')
    ax.text(6.0, 0.27, "Zero Tokenizer Vocabulary $\\quad \\vert \\quad$ Strict 0% Out-Of-Vocabulary (OOV) $\\quad \\vert \\quad$ Preserves Ethiopic Fidel Morphology", ha='center', va='center', fontsize=8.0, color='#334155')

    ax.annotate("", xy=(6.0, 0.85), xytext=(6.0, 0.65), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0284C7"))

    # 2. 12-Layer ByT5 Encoder
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.85), 11.0, 0.65, boxstyle="round,pad=0.03", fc="#EFF6FF", ec="#3B82F6", lw=1.2))
    ax.text(6.0, 1.33, "12-LAYER ByT5 ENCODER (580M PARAMETERS, BFLOAT16 PRECISION)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1E40AF')
    ax.text(6.0, 1.13, r"$\mathbf{H}_{enc} = \mathrm{Encoder}(b_1, \dots, b_M) \in \mathbb{R}^{M \times 1536}$ $\quad \vert \quad$ Sequence length up to $M=768$ bytes", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#1D4ED8')
    ax.text(6.0, 0.95, r"BF16 8 exponent bits ($10^{\pm 38}$) eliminates FP16 NaN collapse ($>65,504$ overflow at step 350)", ha='center', va='center', fontsize=7.5, color='#1E3A8A')

    # Cross-Attention Bridge
    ax.annotate("", xy=(6.0, 1.95), xytext=(6.0, 1.50), arrowprops=dict(arrowstyle="->", lw=2.0, color="#7C3AED"))
    ax.text(6.0, 1.72, r"Multi-Head Cross-Attention: $\mathrm{Softmax}\left(\frac{Q_{dec} K_{enc}^T}{\sqrt{d}}\right) V_{enc}$ (Dynamic Span Copying)",
            ha='center', va='center', fontsize=8.0, fontweight='bold', color="#6D28D9",
            bbox=dict(boxstyle="round,pad=0.15", fc="#F5F3FF", ec="#7C3AED", lw=0.8))

    # 3. 12-Layer ByT5 Decoder
    ax.add_patch(patches.FancyBboxPatch((0.5, 1.95), 11.0, 0.68, boxstyle="round,pad=0.03", fc="#FAF5FF", ec="#9333EA", lw=1.2))
    ax.text(6.0, 2.45, "12-LAYER ByT5 DECODER (AUTOREGRESSIVE GENERATION, ADAFACTOR OPTIMIZER)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#6B21A8')
    ax.text(6.0, 2.25, r"$P(Y \mid X) = \prod_{t=1}^K P(y_t \mid y_{<t}, \mathbf{H}_{enc})$ $\quad \vert \quad$ Batched Greedy Decoding (90s vs 45min, 30x speedup)", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#581C87')
    ax.text(6.0, 2.07, r"Adafactor factorized second moments ($\hat{V} \approx \frac{R C}{\mathrm{mean}(R)}$) cuts optimizer VRAM by 60%", ha='center', va='center', fontsize=7.5, color='#7E22CE')

    ax.annotate("", xy=(6.0, 2.85), xytext=(6.0, 2.63), arrowprops=dict(arrowstyle="->", lw=1.5, color="#9333EA"))

    # 4. Target Linearization & Grammar
    ax.add_patch(patches.FancyBboxPatch((0.5, 2.85), 11.0, 0.62, boxstyle="round,pad=0.03", fc="#FEF3C7", ec="#D97706", lw=1.2))
    ax.text(6.0, 3.32, "LINEARIZED STRUCTURED TARGET EMISSION (NATIVE NULL SUPPORT)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#92400E')
    ax.text(6.0, 3.12, r"Generated String: $[ \text{aspect} \mid \text{category} \mid \text{sentiment} \mid \text{opinion} ] \; ; \; \dots$", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#B45309')
    ax.text(6.0, 2.96, r"Implicit fields emit literal 'NULL' $\rightarrow$ Conquers Extractive Wall (First-ever 274 implicit TPs in Amharic!)", ha='center', va='center', fontsize=7.5, color='#78350F')

    ax.annotate("", xy=(6.0, 3.65), xytext=(6.0, 3.47), arrowprops=dict(arrowstyle="->", lw=1.5, color="#D97706"))

    # 5. Deterministic Quadruple Parser
    ax.add_patch(patches.FancyBboxPatch((1.5, 3.65), 9.0, 0.45, boxstyle="round,pad=0.03", fc="#DCFCE7", ec="#16A34A", lw=1.2))
    ax.text(6.0, 3.94, "DETERMINISTIC QUADRUPLE PARSER $\\rightarrow$ ACOS QUADRUPLES", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#14532D')
    ax.text(6.0, 3.76, "Parses brackets & pipes into structured (Aspect, Category, Sentiment, Opinion) tuples (Strict Exact Match)", ha='center', va='center', fontsize=8.0, color='#15803D')

    plt.tight_layout(pad=0.1)
    fig.savefig('docs/figures/byt5_architecture_flow.png', bbox_inches='tight')
    plt.close(fig)
    print("Generated docs/figures/byt5_architecture_flow.png")


if __name__ == "__main__":
    generate_sdrn_diagram()
    generate_span_aste_diagram()
    generate_byt5_diagram()
    print("All architecture flow diagrams compiled successfully!")
