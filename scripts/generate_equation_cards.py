#!/usr/bin/env python3
"""
Generate rendered mathematical equation cards for SDRN, Span-ASTE, and ByT5 presentations.
Uses Matplotlib Mathtext / LaTeX typesetting to produce crisp, high-resolution formula cards.
Guarantees zero uncompiled LaTeX strings in PowerPoint text frames.
"""

import os
import sys

sys.path.insert(0, '/home/codeknight/.local/lib/python3.14/site-packages')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("docs/figures", exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']


def render_eq_card(filename, title, equations, explanation_bullets, width=10.5, height=3.6):
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Card background
    card = patches.FancyBboxPatch((0.1, 0.1), 9.8, 9.8, boxstyle="round,pad=0.1", fc="#F8FAFC", ec="#0284C7", lw=1.5)
    ax.add_patch(card)

    # Title banner
    banner = patches.FancyBboxPatch((0.2, 8.2), 9.6, 1.5, boxstyle="round,pad=0.05", fc="#E0F2FE", ec="#0369A1", lw=1)
    ax.add_patch(banner)
    ax.text(5.0, 8.95, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#0369A1')

    # Equations block
    y_eq = 6.9
    spacing = 1.35 if len(equations) <= 2 else 1.05
    for eq in equations:
        ax.text(5.0, y_eq, eq, ha='center', va='center', fontsize=10.5, fontweight='bold', color='#0F172A')
        y_eq -= spacing

    # Explanations
    y_exp = 3.4 if len(equations) <= 2 else 2.8
    for bullet in explanation_bullets:
        ax.text(0.6, y_exp, bullet, ha='left', va='center', fontsize=9, color='#334155')
        y_exp -= 1.05

    plt.tight_layout(pad=0.1)
    fig.savefig(f"docs/figures/{filename}", bbox_inches='tight')
    plt.close(fig)
    print(f"Generated docs/figures/{filename}")


# ==============================================================================
# SDRN EQUATION CARDS
# ==============================================================================

# 1. SDRN Context & Subword Pooling
render_eq_card(
    "eq_sdrn_encoder.png",
    "AfroXLMR Contextual Encoding & Subword Pooling Formulation",
    [
        r"$H = \mathrm{AfroXLMR}([x_1, \dots, x_N]) \in \mathbb{R}^{N \times d}, \quad \text{where } d = 768$",
        r"$H_{word} = M_{align} H_{sub}, \quad M_{align} \in \mathbb{R}^{N_{word} \times N_{sub}} \quad (\text{Word-Level Masked Mean})$"
    ],
    [
        r"• Why AfroXLMR: Pretrained on multilingual African corpora, capturing long-distance Amharic SOV dependencies.",
        r"• Why Subword Pooling: Averages constituent wordpiece vectors to align representations with authentic lexical words."
    ]
)

# 2. SDRN Entity Channel Recurrence & Emission
render_eq_card(
    "eq_sdrn_entity_channel.png",
    "SDRN Entity Channel: Recurrence, Emissions & 5-Way Tag Space",
    [
        r"$H_e^{(0)} = H \in \mathbb{R}^{N \times d}, \quad h_i^{e,(t)} = \mathrm{BiLSTM}\left(h_i^{e,(t-1)} + \Delta_{RSM, i}^{(t)}\right)$",
        r"$E_i^{(t)} = W_e h_i^{e,(t)} + b_e \in \mathbb{R}^5, \quad \mathcal{Y} = \{O, B\text{-}ASP, I\text{-}ASP, B\text{-}OPN, I\text{-}OPN\}$"
    ],
    [
        r"• Why 5-Way BIO Tagging: Unifies aspect and opinion extraction into mutually-exclusive labels, preventing dual-tag conflicts.",
        r"• Why BiLSTM Recurrence: Models bidirectional word sequence context while dynamically absorbing relational feedback $\Delta_{RSM}$."
    ]
)

# 3. SDRN Relation Channel Recurrence & Matrix
render_eq_card(
    "eq_sdrn_relation_channel.png",
    r"SDRN Relation Channel: $N \times N$ Token-Pair Correlation & RelLSTM",
    [
        r"$r_{ij}^{(0)} = \left[ h_i \,;\, h_j \,;\, |h_i - h_j| \,;\, h_i \odot h_j \right] \in \mathbb{R}^{4d}, \quad S_{ij} = \frac{h_i^T W_r h_j}{\sqrt{d}}$",
        r"$r_{ij}^{(t)} = \mathrm{RelLSTM}\left(r_{ij}^{(t-1)} + B_{ESM, ij}^{(t)}\right), \quad P(rel_{ij} = 1) = \sigma\left(W_{rel} r_{ij}^{(t)} + b_{rel}\right)$"
    ],
    [
        r"• Why $N \times N$ Matrix: Directly evaluates all pairwise word relations, bypassing surface distance in Amharic's SOV structure.",
        r"• Difference & Product Terms: $|h_i - h_j|$ captures feature contrast; $h_i \odot h_j$ models multiplicative affinity."
    ]
)

# 4. SDRN ESM & RSM Feedback Loops
render_eq_card(
    "eq_sdrn_esm_rsm.png",
    "SDRN Synchronous Bidirectional Feedback Equations (Chen et al. 2020)",
    [
        r"$\mathbf{ESM:}$ $\quad p_i = \mathrm{Softmax}(W_e h_i^e) \in \mathbb{R}^5, \quad M_{ij} = p_i \otimes p_j, \quad B_{ESM,ij} = W_{esm} \mathrm{vec}(M_{ij})$",
        r"$\mathbf{RSM:}$ $\quad \alpha_{ij} = \mathrm{Softmax}_j(W_\alpha r_{ij}), \quad c_i = \sum_{j=1}^N \alpha_{ij} h_j^e, \quad \Delta_{RSM,i} = W_{rsm} c_i$"
    ],
    [
        r"• Why ESM is used: Suppresses relations involving background $O$ tokens, forcing the relation matrix to focus on entities.",
        r"• Why RSM is used: Pulls target aspect semantics into ambiguous opinion verbs, rescuing dropped boundaries."
    ]
)

# 5. SDRN CRF & Loss
render_eq_card(
    "eq_sdrn_crf_loss.png",
    "SDRN Linear-Chain CRF & Multi-Task Loss Formulation",
    [
        r"$\mathrm{Score}(X, y) = \sum_{i=1}^N E_{i, y_i} + \sum_{i=1}^{N-1} T_{y_i, y_{i+1}} \quad \vert \quad Z(X) = \sum_{y'} \exp(\mathrm{Score}(X, y'))$",
        r"$\mathcal{L}_{total} = \left[ \log Z(X) - \mathrm{Score}(X, y^*) \right] + \lambda_{rel} \mathrm{BCE}(R^{(T)}, Y^*)$"
    ],
    [
        r"• Why CRF is used: Enforces valid sequential transitions ($T$), eliminating broken tags (e.g., $O \rightarrow I\text{-}ASP$).",
        r"• Supervision Timing: Supervised ONLY at final recurrent iteration $T$; intermediate steps freely synchronize."
    ]
)


# ==============================================================================
# SPAN-ASTE EQUATION CARDS
# ==============================================================================

# 6. Span-ASTE Enumeration & Candidate Space
render_eq_card(
    "eq_span_aste_enumeration.png",
    "Span Candidate Enumeration Mechanics & Quadratic Complexity",
    [
        r"$\mathcal{S} = \left\{ s = [i, j) \;\mid\; 1 \leq i < j \leq N, \; (j - i) \leq L=8 \right\}$",
        r"$|\mathcal{S}| \approx N \cdot L - \frac{L(L-1)}{2} \approx 212 \text{ candidate spans for } N=30 \quad (\mathcal{O}(N^2 \times L^2) \text{ candidate pairs})$"
    ],
    [
        r"• Why Span Enumeration: Bypasses fragile token-level BIO decisions by proposing whole contiguous spans as atomic units.",
        r"• Shatters the Ceiling: Expands candidate reachability from SDRN's 33.04% to 81.91%, making +1,424 gold pairs accessible."
    ]
)

# 7. Span-ASTE Prefix Pooling & Vector
render_eq_card(
    "eq_span_aste_prefix_rep.png",
    r"Span-ASTE $O(1)$ Prefix-Sum Mean Pooling & 2324-D Span Vector",
    [
        r"$P_0 = 0, \quad P_t = \sum_{k=0}^{t-1} h_k \quad \Longrightarrow \quad \mathrm{MeanPool}(i, j) = \frac{P_j - P_i}{j - i} \quad \in \mathcal{O}(1)$",
        r"$e_s = [h_i \,;\, h_{j-1} \,;\, \mathrm{MeanPool}(i, j) \,;\, w_{j-i}] \in \mathbb{R}^{2324} \quad (3 \times 768 + 20)$"
    ],
    [
        r"• Why Prefix Pooling: Captures all internal evaluative words and negation prefixes in multi-word expressions in $O(1)$ time.",
        r"• Why Width Embeddings $w_{j-i}$: Encodes structural length priors (e.g. opinions rarely exceed 4 words in Amharic)."
    ]
)

# 8. Span-ASTE Mention Pruning
render_eq_card(
    "eq_span_aste_pruning.png",
    "Dual MLP Mention Pruning Heads & Top-K Candidate Selection",
    [
        r"$s_{asp}(s) = \mathbf{w}_{asp}^T \mathrm{ReLU}(W_{asp} e_s + b_{asp}), \quad s_{opn}(s) = \mathbf{w}_{opn}^T \mathrm{ReLU}(W_{opn} e_s + b_{opn})$",
        r"$\mathcal{L}_{mention} = \mathrm{BCE}(s_{asp}, y_{asp}) + \mathrm{BCE}(s_{opn}, y_{opn}) \quad \vert \quad \text{Keep Top } K_a = 0.4 N, \; K_o = 0.4 N$"
    ],
    [
        r"• Why Mention Pruning: Computing heavy tensor interactions on all ~45,000 raw span pairs causes GPU Out-of-Memory (OOM).",
        r"• How it helps: Filters out >85% of background non-entity spans before expensive pairwise biaffine relation modeling."
    ]
)

# 9. Span-ASTE Deep Biaffine Tensor
render_eq_card(
    "eq_span_aste_biaffine.png",
    "Deep Biaffine Bilinear Tensor Scoring (Dozat 2017 / Xu et al. 2021)",
    [
        r"$S(a, o)_c = {e_a'}^T \mathbf{U}_c e_o' + \mathbf{W}_c [e_a' ; e_o' ; d(a, o) ; c_{ctx}] + b_c, \quad c \in \{\mathrm{INV}, \mathrm{POS}, \mathrm{NEG}, \mathrm{NEU}\}$",
        r"$\mathcal{L}_{rel} = -\sum_{(a, o)} \alpha_{y^*} \log \left( \frac{\exp(S(a, o)_{y^*})}{\sum_c \exp(S(a, o)_c)} \right), \quad \alpha = [0.1, 1.0, 1.0, 2.0]$"
    ],
    [
        r"• Why Bilinear Tensor $\mathbf{U}_c$: Directly computes directional multiplicative affinity between aspect and opinion representations.",
        r"• Why Cost-Sensitive $\alpha$: Downweights dominant INVALID pairs by 10x, boosting candidate conversion rate to 45.04%."
    ]
)


# ==============================================================================
# BYT5 EQUATION CARDS
# ==============================================================================

# 10. ByT5 Seq2Seq Probability Card
render_eq_card(
    "eq_byt5_seq2seq.png",
    "ByT5 Autoregressive Byte Sequence-to-Sequence Formulation",
    [
        r"$P(Y \mid X) = \prod_{t=1}^K P(y_t \mid y_{<t}, \mathbf{H}_{enc}), \quad \mathbf{H}_{enc} = \mathrm{Encoder}(b_1, \dots, b_M) \in \mathbb{R}^{M \times 1536}$",
        r"$\mathrm{CrossAttn}(Q_{dec}, K_{enc}, V_{enc}) = \mathrm{Softmax}\left(\frac{Q_{dec} K_{enc}^T}{\sqrt{d}}\right) V_{enc} \quad \rightarrow \quad [ a \mid c \mid s \mid o ] \; ; \; \dots$"
    ],
    [
        r"• Why Byte Seq2Seq: 0% OOV on Amharic Ethiopic script, simultaneously extracts all 4 fields without pipeline cascading.",
        r"• Cross-Attention Copying: Dynamically copies exact surface spans from the encoder while predicting abstract categories."
    ]
)

# 11. ByT5 Linearization & Canonical Ordering
render_eq_card(
    "eq_byt5_linearization.png",
    "Structured Quadruple Linearization Grammar & Canonical Sorting",
    [
        r"$\mathcal{T}_{linear} = [ a_1 \mid c_1 \mid s_1 \mid o_1 ] \;;\; [ a_2 \mid c_2 \mid s_2 \mid o_2 ] \;;\; \dots \quad \vert \quad a, o \in \text{Span} \cup \{\text{'NULL'}\}$",
        r"$\text{Canonical Order: } \mathrm{Sort}\left(a_{start} \rightarrow o_{start} \rightarrow \mathrm{Alphabetical}(c)\right)$"
    ],
    [
        r"• Native NULL Emission: Unstated aspects and opinions are generated as literal 'NULL' strings (274 implicit TPs recovered!).",
        r"• Canonical Sorting: Eliminates set-permutation ambiguity during autoregressive generation, accelerating training convergence."
    ]
)

# 12. ByT5 Adafactor & BF16 Card
render_eq_card(
    "eq_byt5_adafactor_ls.png",
    "ByT5 Engineering Math: BF16 Dynamic Range & Adafactor Factorization",
    [
        r"$\mathrm{BF16:} \text{ 8 exponent bits } (10^{\pm 38}) \text{ vs FP16 5 exponent bits } (10^{\pm 5}, \text{max } 65504) \rightarrow \text{Zero NaN Collapse}$",
        r"$\mathrm{Adafactor:} \quad \hat{V}_{ij} \approx \frac{R_i C_j}{\mathrm{mean}(R)} \in \mathcal{O}(n + m) \quad \vert \quad q(v \mid y_t) = (1 - \epsilon)\mathbb{I}(v = y_t) + \frac{\epsilon}{|\mathcal{V}|}$"
    ],
    [
        r"• Why Adafactor: Slashing optimizer VRAM by 60% enabled 580M-parameter ByT5 training with effective batch size 16.",
        r"• Why Label Smoothing $\epsilon=0.1$: Prevents logit overconfidence on noisy silver annotations, stabilizing generalization."
    ]
)

print("All 12 equation cards generated successfully!")
