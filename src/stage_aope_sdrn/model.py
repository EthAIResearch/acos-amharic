"""
Exact SDRN implementation for Joint Aspect-Opinion Pair Extraction (AOPE),
faithfully following the ACL 2020 paper equations and architecture
(Chen et al., ACL 2020, "Synchronous Double-channel Recurrent Network for
Aspect-Opinion Pair Extraction").

Two synchronous channels sharing one pretrained encoder:
  1. Entity extraction unit (target channel):
     - Single unified 5-way BIO classification head:
       0: 'O', 1: 'B-ASP', 2: 'I-ASP', 3: 'B-OPN', 4: 'I-OPN'
     - Decoded with a 5-tag Linear-Chain CRF (crf.py) enforcing syntactic BIO constraints.
  2. Relation detection unit (relation channel):
     - Biaffine self-attention producing row-stochastic attention matrix G^t in R^{N x N} (Eq. 4-5).
     - Supervised with class-weighted cross-entropy (0.01 negative, 1.0 positive) (Eq. 15).

Synchronization across recurrent steps (Eq. 8-13):
  - ESM (Entity -> Relation channel, Eq. 8-10):
    Tokens sharing the same predicted entity span are grouped into entity blocks T_{i,j},
    normalizing context vectors u_{t,i} = sum_j phi(T_{i,j}) h_j^s to update:
    h_{t+1}^r = tanh(relationSyn_s(h^s) + relationSyn_u(u_t)).
  - RSM (Relation -> Entity channel, Eq. 11-13):
    Thresholded relation scores (G_{i,j} >= beta) weight context vectors r_{t,i} = sum_j phi(phi_beta(G_{i,j})) h_j^s:
    h_{t+1}^o = tanh(targetSyn_s(h^s) + targetSyn_r(r_t)).

Final joint objective (Eq. 16):
  L(theta) = L_E + L_R
"""
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.insert(0, os.path.dirname(__file__))
from bio_labels import decode_5way_bio_spans, decode_bio_spans
from crf import LinearChainCRF
from loss import MultiClassDiceLoss

NUM_BIO_LABELS = 5  # 0: O, 1: B-ASP, 2: I-ASP, 3: B-OPN, 4: I-OPN


class RelationAttention(nn.Module):
    """
    Biaffine relation attention (Eq. 4-5 in Chen et al., ACL 2020):
      gamma(h_i^r, h_j^r) = v * tanh(W_ta h_i^r + W_ja h_j^r + b)
      G^t_{i, j} = softmax_j(gamma(h_i^r, h_j^r))
    Produces a row-stochastic attention matrix where sum_j G^t_{i, j} = 1.
    """

    def __init__(self, hidden_dim: int, attention_dim: int | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim or hidden_dim

        self.w_ta = nn.Parameter(torch.empty(self.attention_dim, self.hidden_dim))
        self.w_ja = nn.Parameter(torch.empty(self.attention_dim, self.hidden_dim))
        self.b = nn.Parameter(torch.zeros(1, 1, 1, self.attention_dim))
        self.v = nn.Parameter(torch.empty(1, self.attention_dim))

        nn.init.xavier_uniform_(self.w_ta)
        nn.init.xavier_uniform_(self.w_ja)
        nn.init.xavier_uniform_(self.v)

        self.softmax = nn.Softmax(dim=2)

    def forward(
        self,
        relation_hidden: torch.Tensor,
        mask2d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        relation_hidden: (B, N, H)
        mask2d: (B, N, N) boolean mask (True for content token pairs, False for padding/special)
        Returns: (B, N, N) row-stochastic attention matrix G^t
        """
        ta = F.linear(relation_hidden, self.w_ta).unsqueeze(2)  # (B, N, 1, A)
        ja = F.linear(relation_hidden, self.w_ja).unsqueeze(1)  # (B, 1, N, A)
        alpha = torch.tanh(ta + ja + self.b)                   # (B, N, N, A)
        scores = F.linear(alpha, self.v).squeeze(-1)            # (B, N, N)

        if mask2d is not None:
            # -1e4 fits safely in float16 (max magnitude 65504) without overflow, and exp(-10000) = 0.0
            scores = scores.masked_fill(~mask2d, -1e4)


        g = self.softmax(scores)  # (B, N, N)

        if mask2d is not None:
            g = g * mask2d.float()
        return g


class JointAOPESDRN(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_recurrent_steps: int = 2,
        relation_threshold: float = 0.1,
        dropout: float = 0.1,
        bio_class_weights: list | None = None,
        span_loss_weight: float = 1.0,
        dice_loss_weight: float = 0.0,
        use_crf: bool = True,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.config.hidden_size
        self.T = num_recurrent_steps
        self.beta = relation_threshold  # filters weak relation scores in RSM (Eq. 12)
        self.span_loss_weight = span_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.use_crf = use_crf

        if bio_class_weights is not None:
            if len(bio_class_weights) == 3:
                w_o, w_b, w_i = bio_class_weights
                bio_class_weights = [w_o, w_b, w_i, w_b, w_i]
            bio_w = torch.tensor(bio_class_weights, dtype=torch.float32)
        else:
            bio_w = torch.ones(NUM_BIO_LABELS, dtype=torch.float32)
        self.register_buffer("bio_weights", bio_w)

        self.dropout = nn.Dropout(dropout)

        # Entity channel: unified 5-way BIO projection (Eq. 1-2)
        self.target_head = nn.Linear(h, NUM_BIO_LABELS)

        # CRF layer for 5-tag sequence decoding with syntactic BIO constraints
        if self.use_crf:
            self.crf = LinearChainCRF(
                num_tags=NUM_BIO_LABELS,
                bio_weights=self.bio_weights,
                enforce_bio_constraints=True,
            )

        # Auxiliary Multi-Class Generalized Dice Loss (Li et al., ACL 2020)
        if self.dice_loss_weight > 0.0:
            self.dice_loss = MultiClassDiceLoss(
                num_classes=NUM_BIO_LABELS,
                weight=self.bio_weights,
            )
        else:
            self.dice_loss = None

        # Target Synchronization (RSM, Eq. 11-13)
        self.targetSyn_r = nn.Linear(h, h, bias=False)
        self.targetSyn_s = nn.Linear(h, h, bias=False)

        # Relation Synchronization (ESM, Eq. 8-10)
        self.relationSyn_u = nn.Linear(h, h, bias=False)
        self.relationSyn_s = nn.Linear(h, h, bias=False)

        # Relation detection unit: biaffine relation attention (Eq. 4-5)
        self.relation_attention = RelationAttention(hidden_dim=h, attention_dim=h)

        # Official relation loss negative/positive class weights [0.01, 1.0] (Eq. 15)
        self.register_buffer("relation_loss_weights", torch.tensor([0.01, 1.0], dtype=torch.float32))

    def decode_tags(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor | None = None,
        opinion_emission_bias: float = 0.0,
        **kwargs,
    ) -> list[list[int]]:
        """
        Decodes subword BIO tags for a batch of sequences. `mask` should be
        content_mask (real-word positions only). If use_crf=True, performs
        global Viterbi decoding with BIO constraints; otherwise greedy argmax.
        Returns (B, N) list of tag IDs in {0, 1, 2, 3, 4}.
        """
        if self.use_crf:
            preds = self.crf.decode(
                logits, mask=mask, opinion_emission_bias=opinion_emission_bias
            )
            return preds.cpu().tolist()
        weighted = logits * self.bio_weights.view(1, 1, -1)
        if opinion_emission_bias != 0.0 and weighted.shape[-1] >= 5:
            weighted = weighted.clone()
            weighted[:, :, 3:5] += opinion_emission_bias
        return weighted.argmax(-1).cpu().tolist()


    def _make_entity_tensor(
        self,
        tag_preds: list[list[int]],
        batch_size: int,
        seq_len: int,
        device: torch.device,
        content_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        ESM entity block matrix construction (Eq. 8-9 in Chen et al. 2020).
        For each detected aspect or opinion span [s, e), sets T[b, s:e, s:e] = 1.0,
        grouping all tokens that belong to the same entity.
        """
        t_tensor = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float32, device=device)

        for b in range(batch_size):
            valid_positions = torch.nonzero(content_mask[b], as_tuple=True)[0].tolist()
            if not valid_positions:
                continue

            # Extract spans within valid positions
            seq = tag_preds[b]
            a_begin = -1
            o_begin = -1

            for pos in valid_positions:
                tag = seq[pos]
                # Close aspect span if tag is not continuation (I-ASP = 2)
                if a_begin != -1 and tag != 2:
                    t_tensor[b, a_begin:pos, a_begin:pos] = 1.0
                    a_begin = -1
                # Close opinion span if tag is not continuation (I-OPN = 4)
                if o_begin != -1 and tag != 4:
                    t_tensor[b, o_begin:pos, o_begin:pos] = 1.0
                    o_begin = -1

                if tag == 1:  # B-ASP
                    a_begin = pos
                elif tag == 3:  # B-OPN
                    o_begin = pos

            # Close any trailing spans
            if a_begin != -1:
                t_tensor[b, a_begin:valid_positions[-1] + 1, a_begin:valid_positions[-1] + 1] = 1.0
            if o_begin != -1:
                t_tensor[b, o_begin:valid_positions[-1] + 1, o_begin:valid_positions[-1] + 1] = 1.0

        return t_tensor

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        relation_labels: torch.Tensor | None = None,
        content_mask: torch.Tensor | None = None,
        aspect_labels: torch.Tensor | None = None,
        opinion_labels: torch.Tensor | None = None,
        subword_to_word: torch.Tensor | None = None,
        word_mask: torch.Tensor | None = None,
        word_labels: torch.Tensor | None = None,
        word_relation_labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        if content_mask is None:
            content_mask = attention_mask
        content_mask = content_mask.bool()

        # Convert separate aspect_labels and opinion_labels to 5-way labels if needed
        if labels is None and aspect_labels is not None and opinion_labels is not None:
            labels = torch.zeros_like(aspect_labels)
            labels = torch.where(aspect_labels == 1, 1, labels)
            labels = torch.where(aspect_labels == 2, 2, labels)
            labels = torch.where((labels == 0) & (opinion_labels == 1), 3, labels)
            labels = torch.where((labels == 0) & (opinion_labels == 2), 4, labels)
            labels = torch.where(aspect_labels == -100, -100, labels)

        # Context representation sequence H^s (Eq. 3)
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_sub = self.dropout(encoder_out.last_hidden_state)  # (B, L, H)

        if subword_to_word is not None:
            # Phase 3: Word-Level SDRN (pooling subwords into word representations)
            # subword_to_word: (B, M, L)
            h_word = torch.bmm(subword_to_word.to(h_sub.dtype), h_sub)  # (B, M, H)
            h_s = h_word
            B, N = h_s.shape[:2]
            active_mask = word_mask.bool() if word_mask is not None else (h_s.abs().sum(dim=-1) > 0)
            target_labels = word_labels if word_labels is not None else labels
            target_rel_labels = word_relation_labels if word_relation_labels is not None else relation_labels
        else:
            # Subword-level SDRN (Legacy)
            h_s = h_sub
            B, N = input_ids.shape
            active_mask = content_mask
            target_labels = labels
            target_rel_labels = relation_labels

        mask2d = active_mask.unsqueeze(1) & active_mask.unsqueeze(2)

        # Initialize T_tensor and R_tensor to zero (Section 3.3.1 & 3.3.2)
        t_tensor = torch.zeros((B, N, N), dtype=torch.float32, device=h_s.device)
        r_tensor = torch.zeros((B, N, N), dtype=torch.float32, device=h_s.device)

        target_emissions = None
        relation_score = None

        # Synchronous recurrent iterations (Eq. 8-13)
        for step in range(self.T):
            # 1. Target Synchronization Mechanism (RSM, Eq. 11-13)
            # Filter weak relation scores below beta
            r_filtered = torch.where(r_tensor >= self.beta, r_tensor, torch.zeros_like(r_tensor))
            r_filtered = r_filtered * mask2d.float()

            target_weighted = torch.bmm(r_filtered, h_s)
            target_div = r_filtered.sum(dim=-1, keepdim=True)
            target_div = target_div + (target_div == 0).float()
            target_r = target_weighted / target_div

            # Update target hidden state h_o (Eq. 13)
            target_hidden = torch.tanh(self.targetSyn_s(h_s) + self.targetSyn_r(target_r))

            # 2. Relation Synchronization Mechanism (ESM, Eq. 8-10)
            t_masked = t_tensor * mask2d.float()
            relation_weighted = torch.bmm(t_masked, h_s)
            relation_div = t_masked.sum(dim=-1, keepdim=True)
            relation_div = relation_div + (relation_div == 0).float()
            relation_u = relation_weighted / relation_div

            # Update relation hidden state h_r (Eq. 10)
            relation_hidden = torch.tanh(self.relationSyn_s(h_s) + self.relationSyn_u(relation_u))

            # 3. Entity Extraction Unit: compute 5-way emission scores (Eq. 1-2)
            target_emissions = self.target_head(target_hidden)

            # 4. Relation Detection Unit: compute row-stochastic attention matrix G^t (Eq. 4-5)
            relation_score = self.relation_attention(relation_hidden, mask2d=mask2d)

            # If not at the final recurrent step, update T and R for next iteration
            if step < self.T - 1:
                # Update R_tensor with current relation scores
                r_tensor = relation_score
                # Update T_tensor with decoded entity spans from current emissions
                tag_preds = self.decode_tags(target_emissions, mask=active_mask)
                t_tensor = self._make_entity_tensor(tag_preds, B, N, h_s.device, active_mask)

        loss = None
        loss_e = None
        loss_r = None
        loss_dice = None

        # Compute joint loss at the final step T (Eq. 14-16)
        if target_labels is not None and target_rel_labels is not None:
            # 1. Entity Loss L_E (Eq. 14)
            mask_e = active_mask & (target_labels != -100)
            if self.use_crf:
                loss_e = self.crf(target_emissions, target_labels, mask=mask_e)
            else:
                ce = nn.CrossEntropyLoss(weight=self.bio_weights, ignore_index=-100)
                loss_e = ce(target_emissions.reshape(-1, NUM_BIO_LABELS), target_labels.reshape(-1))

            if self.dice_loss is not None and self.dice_loss_weight > 0.0:
                loss_dice = self.dice_loss(target_emissions, target_labels, mask=mask_e)
                loss_e = loss_e + self.dice_loss_weight * loss_dice

            # 2. Relation Loss L_R (Eq. 15): Class-weighted cross entropy on [1 - G, G]
            rel_ce = nn.CrossEntropyLoss(weight=self.relation_loss_weights, ignore_index=-100)
            r_flat = relation_score.reshape(-1, 1)
            r_probs = torch.cat([1.0 - r_flat, r_flat], dim=1)  # (B*N*N, 2)
            rel_labels_flat = target_rel_labels.reshape(-1).long()
            # Non-content token pairs are ignored
            mask2d_flat = mask2d.reshape(-1)
            rel_labels_flat = torch.where(mask2d_flat, rel_labels_flat, torch.full_like(rel_labels_flat, -100))
            loss_r = rel_ce(r_probs, rel_labels_flat)

            # 3. Total Loss (Eq. 16): L(theta) = L_E + L_R
            loss = self.span_loss_weight * loss_e + loss_r

        return {
            "loss": loss,
            "loss_e": loss_e,
            "loss_r": loss_r,
            "loss_dice": loss_dice,
            "logits": target_emissions,
            "relation_logits": relation_score,
            "word_mask": active_mask,
            "is_word_level": subword_to_word is not None,
            # Backward compatibility aliases:
            "aspect_logits": target_emissions,
            "opinion_logits": target_emissions,
        }