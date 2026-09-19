"""
Joint Aspect-Opinion Pair Extraction (AOPE) model, adapted from SDRN
(Chen et al., ACL 2020, "Synchronous Double-channel Recurrent Network for
Aspect-Opinion Pair Extraction"). Replaces this project's separate
Stage 1 (tagging) + Stage 2 (heuristic pairing) with a single model that
extracts aspect/opinion spans AND detects their relations jointly,
synchronized across recurrent steps -- directly targeting the error
propagation that a strict pipeline suffers (SDRN's own ablation:
`SDRN w/o ESM&RSM`, i.e. no synchronization, underperforms the full
synchronized model by 1-3 F1 points even on an easier, less imbalanced
benchmark than ours).

Two channels, sharing one encoder:
  - Entity channel: two BIO heads (aspect, opinion) -- reuses this
    project's existing dual-head scheme (bio_labels.py / align.py) rather
    than SDRN's single 5-way BA/IA/BP/IP/O tag sequence; functionally
    equivalent, and lets this module reuse already-tested utilities.
  - Relation channel: pairwise self-attention over all token pairs,
    producing an N x N relation matrix, supervised against gold
    aspect-opinion token-pair relations (relation_utils.py).

Synchronization (only at intermediate steps, matching the paper):
  - ESM (Entity -> Relation): current step's decoded entity spans define
    which token pairs are "same-entity"; their averaged context vectors
    feed into the next step's relation-channel hidden state.
  - RSM (Relation -> Entity): current step's relation scores (thresholded)
    weight an average of context vectors that feeds into the next step's
    entity-channel hidden state.

Only the FINAL recurrent step is supervised (Eq. 3/6/14/15 in the paper) --
earlier steps only refine the hidden states through synchronization.
"""
import os
import sys

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from bio_labels import decode_bio_spans

NUM_BIO_LABELS = 3  # O, B, I


class JointAOPESDRN(nn.Module):
    def __init__(self, model_name: str, num_recurrent_steps: int = 2,
                 relation_threshold: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.config.hidden_size
        self.T = num_recurrent_steps
        self.beta = relation_threshold  # filters weak relation scores in RSM, Eq. 12

        self.dropout = nn.Dropout(dropout)
        self.aspect_head = nn.Linear(h, NUM_BIO_LABELS)
        self.opinion_head = nn.Linear(h, NUM_BIO_LABELS)

        # Relation scoring (Eq. 4-5): bilinear-style pairwise attention
        self.rel_w1 = nn.Linear(h, h, bias=False)
        self.rel_w2 = nn.Linear(h, h, bias=False)
        self.rel_score = nn.Linear(h, 1, bias=False)

        # ESM (Eq. 10): entity semantics + context -> next relation hidden state
        self.esm_proj = nn.Linear(h * 2, h)
        # RSM (Eq. 13): relation semantics + context -> next entity hidden state
        self.rsm_proj = nn.Linear(h * 2, h)

    def _relation_logits(self, h_r):
        # h_r: (B, T, H) -> (B, T, T) pairwise scores, Eq. 4-5
        B, N, H = h_r.shape
        a = self.rel_w1(h_r).unsqueeze(2).expand(B, N, N, H)
        b = self.rel_w2(h_r).unsqueeze(1).expand(B, N, N, H)
        scores = self.rel_score(torch.tanh(a + b)).squeeze(-1)  # (B, N, N)
        return scores

    def _entity_semantics(self, aspect_logits, opinion_logits, h_s, attention_mask):
        """ESM: for each token, average context vectors of tokens sharing
        its (decoded) aspect or opinion span -- a discrete approximation of
        SDRN's soft same-entity probability (Eq. 8-9), computed per-example
        since span decoding is inherently sequential/discrete."""
        B = h_s.shape[0]
        u = torch.zeros_like(h_s)
        a_pred = aspect_logits.argmax(-1).cpu().tolist()
        o_pred = opinion_logits.argmax(-1).cpu().tolist()
        id2tag = {0: "O", 1: "B", 2: "I"}

        for b in range(B):
            valid_len = int(attention_mask[b].sum().item())
            a_tags = [id2tag[t] for t in a_pred[b][:valid_len]]
            o_tags = [id2tag[t] for t in o_pred[b][:valid_len]]
            spans = decode_bio_spans(a_tags) + decode_bio_spans(o_tags)
            token_to_span = {}
            for span in spans:
                for i in range(span[0], span[1]):
                    token_to_span[i] = span
            for i in range(valid_len):
                span = token_to_span.get(i)
                if span is None:
                    u[b, i] = h_s[b, i]  # no entity -> fall back to own context vector
                else:
                    u[b, i] = h_s[b, span[0]:span[1]].mean(dim=0)
        return u

    def _relation_semantics(self, rel_scores, h_s, attention_mask):
        """RSM: for each token, weighted average of context vectors using
        thresholded relation scores (Eq. 11-12)."""
        weights = torch.sigmoid(rel_scores)
        weights = torch.where(weights >= self.beta, weights, torch.zeros_like(weights))
        mask2d = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
        weights = weights * mask2d
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        weights = weights / denom
        r = torch.bmm(weights, h_s)
        return r

    def forward(self, input_ids, attention_mask, aspect_labels=None, opinion_labels=None,
                relation_labels=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_s = self.dropout(out.last_hidden_state)  # (B, N, H), fixed context representation

        h_o = h_s  # entity-channel hidden state, starts from context representation
        h_r = h_s  # relation-channel hidden state

        aspect_logits = self.aspect_head(h_o)
        opinion_logits = self.opinion_head(h_o)
        rel_logits = self._relation_logits(h_r)

        for _ in range(self.T - 1):
            # ESM: entity -> relation channel
            u = self._entity_semantics(aspect_logits, opinion_logits, h_s, attention_mask)
            h_r = torch.tanh(self.esm_proj(torch.cat([u, h_s], dim=-1)))

            # RSM: relation -> entity channel
            r = self._relation_semantics(rel_logits, h_s, attention_mask)
            h_o = torch.tanh(self.rsm_proj(torch.cat([r, h_s], dim=-1)))

            aspect_logits = self.aspect_head(h_o)
            opinion_logits = self.opinion_head(h_o)
            rel_logits = self._relation_logits(h_r)

        loss = None
        if aspect_labels is not None and opinion_labels is not None and relation_labels is not None:
            ce = nn.CrossEntropyLoss(ignore_index=-100)
            loss_a = ce(aspect_logits.reshape(-1, NUM_BIO_LABELS), aspect_labels.reshape(-1))
            loss_o = ce(opinion_logits.reshape(-1, NUM_BIO_LABELS), opinion_labels.reshape(-1))

            mask2d = (attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)).float()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            rel_loss_raw = bce(rel_logits, relation_labels.float())
            loss_r = (rel_loss_raw * mask2d).sum() / mask2d.sum().clamp(min=1.0)

            loss = loss_a + loss_o + loss_r

        return {
            "loss": loss,
            "aspect_logits": aspect_logits,
            "opinion_logits": opinion_logits,
            "relation_logits": rel_logits,
        }
