# Stage 3: Category Classification — Design & Architecture

## Overview

Stage 3 classifies each candidate aspect-opinion pair into one of the **22 fixed categories** defined in `label_space.json`.

- **Input**: A sentence with word-level tokens, an aspect span $(a_{start}, a_{end})$, and an opinion span $(o_{start}, o_{end})$.
- **Scope**: Explicit-both pairs ($a_{start} \neq -1$ and $o_{start} \neq -1$). Implicit aspect/opinion handling is deferred to Stage 5.
- **Output**: Category label ID $\in [0, 21]$.

---

## Model Architecture

The classifier (`PairClassifier` in `src/stage3_category/model.py`) uses a shared transformer encoder:

1. **Subword Masking**: Binary masks are constructed over the subwords for the aspect span, the opinion span, and the full sentence attention mask.
2. **Masked Mean Pooling**:
   $$\mathbf{h}_{\text{aspect}} = \text{MeanPool}(\mathbf{H}, \mathbf{M}_{\text{aspect}})$$
   $$\mathbf{h}_{\text{opinion}} = \text{MeanPool}(\mathbf{H}, \mathbf{M}_{\text{opinion}})$$
   $$\mathbf{h}_{\text{sentence}} = \text{MeanPool}(\mathbf{H}, \mathbf{M}_{\text{attention}})$$
3. **Representation Fusion**: Concatenates all three representations:
   $$\mathbf{h}_{\text{pair}} = [\mathbf{h}_{\text{aspect}} \,;\, \mathbf{h}_{\text{opinion}} \,;\, \mathbf{h}_{\text{sentence}}] \in \mathbb{R}^{3 \times d_{\text{hidden}}}$$
4. **Classification Head**:
   $$\text{logits} = \mathbf{W}_2 \cdot \text{Dropout}(\text{ReLU}(\mathbf{W}_1 \mathbf{h}_{\text{pair}}))$$

This architecture is modular and reused for Stage 4 (Sentiment Classification).

---

## Class Imbalance & Weighting

The 22-category taxonomy exhibits strong class imbalance across the dataset (e.g., `GOVERNANCE#TRANSPARENCY` accounts for ~25% of training quads, while 10 categories have low support $<30$ examples, and 2 categories have 0 examples in the explicit subset).

To stabilize training and ensure minority classes receive learning signal without exploding gradients, we apply capped inverse-frequency class weighting (`compute_class_weights` in `src/common/pair_utils.py`):

$$w_c = \min\left(\frac{N_{\text{total}}}{C \times N_c}, \, \text{cap}\right)$$

Default cap is set to $15.0$.
