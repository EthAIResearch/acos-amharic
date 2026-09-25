# Stage ByT5 Generative ACOS: Architecture & Experiment Analysis

## 1. Executive Summary & Problem Context
Across Phases 1 through 5, all models in this repository operated under an **extractive paradigm**:
- **Phase 1 (6-Stage Pipeline):** Broken by error compounding cascade ($F_1 \approx \prod F_1^{(i)} \rightarrow 14.2\%$).
- **Phase 2 (Joint SDRN):** Bounded by token-level BIO fragmentation ceiling (33.04% candidate ceiling recall).
- **Phases 3–5 (Span-ASTE Runs 1–3):** Raised upstream Aspect F1 to a project-record **55.20%** and candidate ceiling recall to **81.91%**, yet hit the **Inherent Extractive Wall**:
  $$\text{Recall}_{\text{implicit}} \equiv 0.00\%$$
  Because **42.7% of all quadruples in the Amharic dataset contain implicit aspects (`a = NULL`) or implicit opinions (`o = NULL`)**, extractive models can never extract tokens that do not physically exist in the input sentence.

**Phase 6 executes a paradigm shift to Generative Sequence-to-Sequence Modeling using `google/byt5-base` (580M parameters)**, directly linearizing input sentences into structured quadruple strings.

---

## 2. Core Architectural & Mathematical Design

### A. Autoregressive Sequence-to-Sequence Formulation
Given an Amharic input byte sequence $X = [x_1, \dots, x_N]$, the decoder autoregressively models the conditional probability distribution over target tokens:
$$P(Y \mid X) = \prod_{m=1}^M P(y_m \mid y_{<m}, X)$$
where $Y$ is the linearized representation of $K$ quadruples:
$$Y = \left[ [ a_1 \mid c_1 \mid s_1 \mid o_1 ] ; \dots ; [ a_K \mid c_K \mid s_K \mid o_K ] \right]$$

### B. Raw UTF-8 Byte Processing (Strict 0% OOV)
- Operates directly on raw UTF-8 bytes (256 byte IDs).
- Eliminates fixed subword vocabularies (SentencePiece/WordPiece).
- Preserves morphological clitics (negation clitics like *'al-'*, object pronouns like *'-gn'*, conjunctions like *'-m'*).
- Unseen slang, phonetic variations, and neologisms (which constitute 66.2% of opinion spans) are represented with zero `<unk>` tokens.

### C. Native Implicit Extraction (`NULL`)
When an aspect or opinion is implicit in the text, the model autoregressively generates the literal token `NULL`. This unifies explicit and implicit extraction into a single model without heuristic post-processing rules.

---

## 3. Engineering & Stability Solutions

### A. Precision: BFloat16 vs. FP16 NaN Collapse
- In initial trials under standard FP16, byte sequences (~3-4× longer than subwords) triggered gradient underflow and exponent overflow ($> 65,504$), crashing training loss to `NaN` at step 350.
- Resolved by upgrading to **BFloat16 (BF16)** with 8 exponent bits ($10^{\pm 38}$ dynamic range), achieving rock-solid numerical stability.
- CPU executions gracefully normalize to FP32.

### B. Adafactor Optimizer (VRAM Reduction)
- Replaced AdamW with **Adafactor** using factorized second moments ($V \approx R \cdot C / \text{mean}(R)$).
- Saved **60% optimizer VRAM**, enabling batch size 4 with gradient accumulation 4 (effective batch size 16) without OOM.

### C. Fast Batched Greedy Decoding
- Epoch validation generation using standard beam search took over 45 minutes per epoch.
- Implemented batched greedy autoregressive decoding (batch size 16), reducing validation time to **90 seconds per epoch (30× speedup)**.

---

## 4. Empirical Evaluation Metrics

### A. Development Set Best Checkpoint (Epoch 9)
- **Full Quadruple (A, C, S, O):** Precision 18.20% | Recall 17.62% | **F1: 17.90%** (182 TPs)
- **Implicit Quadruples:** Precision 23.49% | Recall 16.36% | **F1: 19.28%** (70 TPs) — **Historic Milestone**
- **Explicit Quadruples:** Precision 15.95% | Recall 18.51% | **F1: 17.14%** (112 TPs)
- **Category-Sentiment (C, S):** Precision 49.50% | Recall 47.92% | **F1: 48.70%** (495 TPs)
- **AOPE Pair (A, O):** Precision 30.00% | Recall 29.04% | **F1: 29.51%** (300 TPs)
- **ASTE Triplet (A, S, O):** Precision 27.40% | Recall 26.52% | **F1: 26.96%** (274 TPs)
- **ACSE (A, C, S):** Precision 30.30% | Recall 29.33% | **F1: 29.81%** (303 TPs)

### B. Full Held-Out Test Set (Final Evaluation)
- **Full Quadruple (A, C, S, O):** Precision 16.52% | Recall 15.78% | **F1: 16.14%** (**811 TPs**)
- **Implicit Quadruples:** Precision 17.90% | Recall 12.71% | **F1: 14.87%** (**274 TPs**)
- **Explicit Quadruples:** Precision 15.90% | Recall 18.00% | **F1: 16.88%** (**537 TPs**)
- **Category-Sentiment (C, S):** Precision 49.40% | Recall 47.20% | **F1: 48.27%** (**2,425 TPs**)
- **AOPE Pair (A, O):** Precision 27.32% | Recall 26.10% | **F1: 26.69%** (**1,341 TPs**)
- **ASTE Triplet (A, S, O):** Precision 24.24% | Recall 23.16% | **F1: 23.69%** (**1,190 TPs**)
- **ACSE (A, C, S):** Precision 30.64% | Recall 29.27% | **F1: 29.94%** (**1,504 TPs**)

---

## 5. Master Cross-Model Comparison

| Model Generation | Paradigm | Aspect F1 | AOPE Pair F1 | Full Quad F1 | Implicit Quad F1 | Implicit TPs |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1. 6-Stage Modular Pipeline | Cascaded Extractive | 50.60% | 14.20% | 14.20% | 0.00% | 0 |
| 2. Joint SDRN (BiLSTM+CRF) | Token Joint Tagging | 48.00% | 34.85% | N/A | 0.00% | 0 |
| 3. Span-ASTE Run 1 (MLP) | Span Enumeration | 50.59% | 30.80% | N/A | 0.00% | 0 |
| 4. Span-ASTE Run 2 (Biaffine) | Span + Bilinear | 53.76% | 34.02% | N/A | 0.00% | 0 |
| 5. Span-ASTE Run 3 (MeanPool) | Span + Focal Pool | **55.20%** | **32.04%** | N/A | 0.00% | 0 |
| 6a. ByT5-Base (Dev Best Ep 9) | Byte Autoregressive | 29.81%* | 29.51% | **17.90%** | **19.28%** | 70 |
| 6b. ByT5-Base (Final Test) | Byte Autoregressive | 29.94%* | 26.69% | **16.14%** | **14.87%** | **274** |

---

## 6. Strategic Future Directions
1. **Curriculum Pre-Training:** Pre-training on explicit quadruples to bootstrap span localization before joint training on implicit cases.
2. **Constrained Decoding (Trie / FSM):** Enforcing valid category tokens ($c \in \mathcal{C}$) and sentiment polarities ($s \in \mathcal{S}$) during generation.
3. **Foundation Scaling:** Scaling from ByT5-Base (580M) to ByT5-Large (1.2B) via LoRA.
