"""
Span-ASTE PyTorch Model Architecture
====================================
Implements Span-ASTE (Xu et al., ACL 2021; arXiv:2107.12214):
  1. Transformer Encoder (Afro-XLMR) + Subword Mean Pooling
  2. Span Enumeration & Width Feature Embeddings (Eq. 2)
  3. Dual-Channel Span Pruning via ATE/OTE Mention Module (Eq. 3-4)
  4. Span-Pair Distance Embeddings & Sentiment Relation Classification (Eq. 5-6)
  5. End-to-End Multi-Task Loss (Eq. 7)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from span_utils import (
    bucket_value,
    compute_span_distance,
    MENTION2ID,
    RELATION2ID,
)


class MLP(nn.Module):
    """2-layer Feed-Forward Network matching the paper (dim 150, ReLU, dropout 0.4)."""

    def __init__(self, in_dim: int, hidden_dim: int = 150, out_dim: int = 3, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpanASTEModel(nn.Module):
    """
    Span-ASTE model for Aspect-Opinion Pair and Sentiment Triplet Extraction.
    """

    def __init__(
        self,
        model_name: str = "Davlan/afro-xlmr-base",
        max_span_length: int = 8,
        pruning_ratio: float = 0.5,
        width_dim: int = 20,
        distance_dim: int = 128,
        hidden_dim: int = 150,
        dropout: float = 0.4,
        mention_loss_weight: float = 1.0,
        relation_loss_weight: float = 1.0,
        relation_loss_weights: list[float] | None = None,
    ):
        super().__init__()
        self.max_span_length = max_span_length
        self.pruning_ratio = pruning_ratio
        self.mention_loss_weight = mention_loss_weight
        self.relation_loss_weight = relation_loss_weight

        # 1. Transformer Encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        d_model = self.encoder.config.hidden_size

        # 2. Span Width and Distance Feature Embeddings
        # 10 buckets: [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+]
        self.width_embedding = nn.Embedding(10, width_dim)
        self.distance_embedding = nn.Embedding(10, distance_dim)

        # Span representation size: [start_word; end_word; width_emb] -> 2*d_model + width_dim
        span_rep_dim = 2 * d_model + width_dim

        # 3. Mention Module (ATE & OTE Supervision: Target, Opinion, Invalid)
        self.mention_classifier = MLP(
            in_dim=span_rep_dim,
            hidden_dim=hidden_dim,
            out_dim=3,
            dropout=dropout,
        )

        # 4. Triplet / Relation Module (Positive, Negative, Neutral, Invalid)
        # Pair representation: [target_span; opinion_span; distance_emb]
        pair_rep_dim = 2 * span_rep_dim + distance_dim
        self.relation_classifier = MLP(
            in_dim=pair_rep_dim,
            hidden_dim=hidden_dim,
            out_dim=4,
            dropout=dropout,
        )

        # Optional class weights for relation loss (e.g. to balance 90%+ Invalid pairs)
        if relation_loss_weights is not None:
            self.register_buffer(
                "relation_weights", torch.tensor(relation_loss_weights, dtype=torch.float32)
            )
        else:
            self.relation_weights = None

    def get_word_representations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        subword_to_word: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Passes inputs through Transformer and aggregates subwords into words via
        mean pooling matrix P: (B, M, L) @ (B, L, d) -> (B, M, d).
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sub_hidden = outputs.last_hidden_state  # (B, L, d)

        if subword_to_word is not None:
            # Word-level pooling
            word_hidden = torch.bmm(subword_to_word, sub_hidden)  # (B, M, d)
            return word_hidden
        return sub_hidden

    def build_span_representations(
        self,
        word_hidden: torch.Tensor,
        spans: list[tuple[int, int]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs span representations for a list of spans [s, e):
          s_{i, j} = [x_s; x_{e-1}; f_width(e - s)]
        Returns:
          span_reps: (K, 2*d + width_dim)
          width_bucket_ids: (K,)
        """
        if not spans:
            return torch.empty((0, 2 * word_hidden.size(-1) + self.width_embedding.embedding_dim), device=device), torch.empty((0,), dtype=torch.long, device=device)

        starts = torch.tensor([s for s, e in spans], dtype=torch.long, device=device)
        ends = torch.tensor([e - 1 for s, e in spans], dtype=torch.long, device=device)
        widths = torch.tensor([bucket_value(e - s) for s, e in spans], dtype=torch.long, device=device)

        start_reps = word_hidden[starts]  # (K, d)
        end_reps = word_hidden[ends]      # (K, d)
        width_reps = self.width_embedding(widths)  # (K, width_dim)

        span_reps = torch.cat([start_reps, end_reps, width_reps], dim=-1)  # (K, 2d + width_dim)
        return span_reps, widths

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        subword_to_word: torch.Tensor | None = None,
        seq_spans: list[list[tuple[int, int]]] | None = None,
        gold_mention_labels: list[torch.Tensor] | None = None,
        gold_pairs: list[list[tuple[tuple[int, int], tuple[int, int], int]]] | None = None,
        num_words: list[int] | None = None,
    ) -> dict:
        """
        Forward pass for a batch of sequences.
        Because each sentence has dynamic span counts, spans and pairs are handled per example.
        """
        device = input_ids.device
        bsz = input_ids.size(0)

        # 1. Contextual Word Representations
        word_hidden = self.get_word_representations(input_ids, attention_mask, subword_to_word)

        total_mention_loss = torch.tensor(0.0, device=device)
        total_relation_loss = torch.tensor(0.0, device=device)
        num_mention_examples = 0
        num_relation_examples = 0

        batch_outputs = []

        for b in range(bsz):
            n_w = num_words[b] if num_words is not None else word_hidden.size(1)
            spans = seq_spans[b] if seq_spans is not None else []
            num_spans = len(spans)

            if num_spans == 0:
                batch_outputs.append({
                    "target_candidates": [],
                    "opinion_candidates": [],
                    "pred_triplets": [],
                    "mention_logits": None,
                    "relation_logits": None,
                })
                continue

            # 2. Build Span Representations for all enumerated spans
            h_b = word_hidden[b]  # (M, d)
            span_reps, _ = self.build_span_representations(h_b, spans, device)  # (num_spans, span_dim)

            # 3. Mention Classification (Eq. 3)
            mention_logits = self.mention_classifier(span_reps)  # (num_spans, 3)
            mention_probs = F.softmax(mention_logits, dim=-1)     # (num_spans, 3)

            # Mention Loss
            if gold_mention_labels is not None and b < len(gold_mention_labels):
                m_labels = gold_mention_labels[b].to(device)
                m_loss = F.cross_entropy(mention_logits, m_labels)
                total_mention_loss = total_mention_loss + m_loss
                num_mention_examples += 1

            # 4. Dual-Channel Span Pruning (Eq. 4)
            # k = ceil(n * z)
            k = max(1, round(n_w * self.pruning_ratio))
            k = min(k, num_spans)

            target_scores = mention_probs[:, MENTION2ID["TARGET"]]   # (num_spans,)
            opinion_scores = mention_probs[:, MENTION2ID["OPINION"]] # (num_spans,)

            top_t_indices = torch.topk(target_scores, k=k).indices.tolist()
            top_o_indices = torch.topk(opinion_scores, k=k).indices.tolist()

            pruned_target_spans = [spans[idx] for idx in top_t_indices]
            pruned_opinion_spans = [spans[idx] for idx in top_o_indices]

            # In training, ensure any gold aspect/opinion spans are present in candidate pool
            if self.training and gold_pairs is not None and b < len(gold_pairs):
                gold_b = gold_pairs[b]
                gold_targets = {p[0] for p in gold_b}
                gold_opinions = {p[1] for p in gold_b}
                for gt in gold_targets:
                    if gt in spans and gt not in pruned_target_spans:
                        pruned_target_spans.append(gt)
                        top_t_indices.append(spans.index(gt))
                for go in gold_opinions:
                    if go in spans and go not in pruned_opinion_spans:
                        pruned_opinion_spans.append(go)
                        top_o_indices.append(spans.index(go))

            num_t = len(pruned_target_spans)
            num_o = len(pruned_opinion_spans)

            # 5. Triplet Module: Form Pair Representations for S^t x S^o
            if num_t > 0 and num_o > 0:
                t_reps = span_reps[torch.tensor(top_t_indices, dtype=torch.long, device=device)]  # (num_t, span_dim)
                o_reps = span_reps[torch.tensor(top_o_indices, dtype=torch.long, device=device)]  # (num_o, span_dim)

                # Pair expansion
                t_expanded = t_reps.unsqueeze(1).expand(num_t, num_o, -1)  # (num_t, num_o, span_dim)
                o_expanded = o_reps.unsqueeze(0).expand(num_t, num_o, -1)  # (num_t, num_o, span_dim)

                # Compute pairwise token distances
                dist_indices = torch.zeros((num_t, num_o), dtype=torch.long, device=device)
                for ti, t_span in enumerate(pruned_target_spans):
                    for oi, o_span in enumerate(pruned_opinion_spans):
                        raw_dist = compute_span_distance(t_span, o_span)
                        dist_indices[ti, oi] = bucket_value(raw_dist)

                dist_reps = self.distance_embedding(dist_indices)  # (num_t, num_o, dist_dim)

                pair_reps = torch.cat([t_expanded, o_expanded, dist_reps], dim=-1)  # (num_t, num_o, pair_dim)
                flat_pair_reps = pair_reps.view(num_t * num_o, -1)

                relation_logits = self.relation_classifier(flat_pair_reps)  # (num_t * num_o, 4)

                # Relation Loss
                if gold_pairs is not None and b < len(gold_pairs):
                    gold_dict = {(p[0], p[1]): p[2] for p in gold_pairs[b]}
                    pair_labels = torch.zeros(num_t * num_o, dtype=torch.long, device=device)
                    idx = 0
                    for t_span in pruned_target_spans:
                        for o_span in pruned_opinion_spans:
                            rel = gold_dict.get((t_span, o_span), RELATION2ID["INVALID"])
                            pair_labels[idx] = rel
                            idx += 1

                    r_loss = F.cross_entropy(
                        relation_logits,
                        pair_labels,
                        weight=self.relation_weights,
                    )
                    total_relation_loss = total_relation_loss + r_loss
                    num_relation_examples += 1

                # Decode Predictions: triplets with relation != INVALID (0)
                pred_triplets = []
                pred_rel_ids = relation_logits.argmax(dim=-1).view(num_t, num_o).tolist()
                for ti, t_span in enumerate(pruned_target_spans):
                    for oi, o_span in enumerate(pruned_opinion_spans):
                        rel_id = pred_rel_ids[ti][oi]
                        if rel_id != RELATION2ID["INVALID"]:
                            pred_triplets.append((t_span, o_span, rel_id))
            else:
                relation_logits = None
                pred_triplets = []

            batch_outputs.append({
                "target_candidates": pruned_target_spans,
                "opinion_candidates": pruned_opinion_spans,
                "pred_triplets": pred_triplets,
                "mention_logits": mention_logits,
                "relation_logits": relation_logits,
            })

        # Average losses over active batch examples
        mention_loss = total_mention_loss / max(1, num_mention_examples)
        relation_loss = total_relation_loss / max(1, num_relation_examples)
        total_loss = (
            self.mention_loss_weight * mention_loss
            + self.relation_loss_weight * relation_loss
        )

        return {
            "loss": total_loss,
            "mention_loss": mention_loss,
            "relation_loss": relation_loss,
            "batch_outputs": batch_outputs,
        }
