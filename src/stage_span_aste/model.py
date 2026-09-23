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
import torch.nn.functional as F
from span_utils import (
    MENTION2ID,
    RELATION2ID,
    bucket_value,
    compute_span_distance,
)
from torch import nn
from transformers import AutoConfig, AutoModel


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


class BiaffineSpanRelationClassifier(nn.Module):
    """
    Deep Biaffine Relation Classifier with Cross-Span Contextual Synchronization.
    Synthesizes:
      1. Dozat & Manning (ICLR 2017): Deep non-linear projections into relation subspaces + bilinear tensor scoring.
      2. Nguyen & Verspoor (2018): Second-order biaffine attention operator for multi-class relation extraction:
         s_{j,k} = Biaffine(h_j^(head), h_k^(tail)) = h_j^T U h_k + W [h_j; h_k] + b
      3. Chen et al. (ACL 2020, SDRN): Cross-span contextual synchronization between aspect spans and opinion spans.
    """

    def __init__(
        self,
        span_dim: int,
        biaffine_dim: int = 256,
        distance_dim: int = 20,
        out_dim: int = 4,
        use_span_sync: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.span_dim = span_dim
        self.biaffine_dim = biaffine_dim
        self.out_dim = out_dim
        self.use_span_sync = use_span_sync
        self.dropout = nn.Dropout(dropout)

        # 1. Deep non-linear projections into relation subspaces (Dozat & Manning 2017)
        self.proj_aspect = nn.Sequential(
            nn.Linear(span_dim, biaffine_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout),
        )
        self.proj_opinion = nn.Sequential(
            nn.Linear(span_dim, biaffine_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout),
        )

        # 2. Second-order bilinear interaction tensor U: (out_dim, biaffine_dim, biaffine_dim)
        # S_{t, o, c}^{bilinear} = h_t^T U_c h_o
        self.U = nn.Parameter(torch.empty(out_dim, biaffine_dim, biaffine_dim))
        nn.init.xavier_uniform_(self.U)

        # 3. Cross-span contextual synchronization (Chen et al. 2020)
        if self.use_span_sync:
            self.sync_aspect_ctx = nn.Linear(biaffine_dim, biaffine_dim, bias=False)
            self.sync_opinion_ctx = nn.Linear(biaffine_dim, biaffine_dim, bias=False)
            sync_feat_dim = 2 * biaffine_dim
        else:
            sync_feat_dim = 0

        # 4. Affine / multi-feature combination classifier
        # Input features: [h_t; h_o; (sync_t; sync_o); h_t * h_o; distance_emb]
        affine_in_dim = 2 * biaffine_dim + sync_feat_dim + biaffine_dim + distance_dim
        self.affine_classifier = nn.Sequential(
            nn.Linear(affine_in_dim, biaffine_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout),
            nn.Linear(biaffine_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        t_spans: torch.Tensor,
        o_spans: torch.Tensor,
        dist_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            t_spans: (N_t, span_dim) candidate aspect span representations
            o_spans: (N_o, span_dim) candidate opinion span representations
            dist_embeddings: (N_t, N_o, distance_dim) pairwise distance embeddings
        Returns:
            relation_logits: (N_t * N_o, out_dim)
        """
        num_t = t_spans.size(0)
        num_o = o_spans.size(0)
        if num_t == 0 or num_o == 0:
            return torch.empty((0, self.out_dim), device=t_spans.device, dtype=t_spans.dtype)

        # 1. Project into deep relation subspaces
        h_t = self.proj_aspect(t_spans)    # (N_t, biaffine_dim)
        h_o = self.proj_opinion(o_spans)  # (N_o, biaffine_dim)

        # 2. Second-order bilinear tensor score: (N_t, N_o, out_dim)
        # einsum: td,cde,oe -> toc (t: target, o: opinion, c: class, d/e: biaffine_dim)
        bilinear_logits = torch.einsum("td,cde,oe->toc", h_t, self.U, h_o)

        # 3. Cross-span contextual synchronization
        if self.use_span_sync:
            scale = 1.0 / (self.biaffine_dim ** 0.5)
            # Alignment affinity matrix: (N_t, N_o)
            affinity = torch.matmul(h_t, h_o.t()) * scale

            # Aspect attends over opinions: (N_t, N_o) @ (N_o, d) -> (N_t, d)
            attn_t = F.softmax(affinity, dim=1)
            ctx_t = torch.matmul(attn_t, self.sync_opinion_ctx(h_o))

            # Opinion attends over aspects: (N_o, N_t) @ (N_t, d) -> (N_o, d)
            attn_o = F.softmax(affinity.t(), dim=1)
            ctx_o = torch.matmul(attn_o, self.sync_aspect_ctx(h_t))

            # Pairwise broadcast
            ctx_t_exp = ctx_t.unsqueeze(1).expand(num_t, num_o, -1)
            ctx_o_exp = ctx_o.unsqueeze(0).expand(num_t, num_o, -1)

        # Pairwise broadcast of span projections
        h_t_exp = h_t.unsqueeze(1).expand(num_t, num_o, -1)
        h_o_exp = h_o.unsqueeze(0).expand(num_t, num_o, -1)
        elem_prod = h_t_exp * h_o_exp

        if self.use_span_sync:
            pair_features = torch.cat(
                [h_t_exp, h_o_exp, ctx_t_exp, ctx_o_exp, elem_prod, dist_embeddings], dim=-1
            )
        else:
            pair_features = torch.cat(
                [h_t_exp, h_o_exp, elem_prod, dist_embeddings], dim=-1
            )

        affine_logits = self.affine_classifier(pair_features)  # (N_t, N_o, out_dim)

        total_logits = (bilinear_logits + affine_logits).contiguous()  # (N_t, N_o, out_dim)
        return total_logits.reshape(num_t * num_o, self.out_dim)


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
        relation_threshold: float | None = None,
        use_biaffine: bool = True,
        biaffine_dim: int = 256,
        use_span_sync: bool = True,
        use_span_mean_pooling: bool = True,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.max_span_length = max_span_length
        self.pruning_ratio = pruning_ratio
        self.mention_loss_weight = mention_loss_weight
        self.relation_loss_weight = relation_loss_weight
        self.relation_threshold = relation_threshold
        self.use_biaffine = use_biaffine
        self.biaffine_dim = biaffine_dim
        self.use_span_sync = use_span_sync
        self.use_span_mean_pooling = use_span_mean_pooling
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

        # 1. Transformer Encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        d_model = self.encoder.config.hidden_size

        # 2. Span Width and Distance Feature Embeddings
        # 10 buckets: [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+]
        self.width_embedding = nn.Embedding(10, width_dim)
        self.distance_embedding = nn.Embedding(10, distance_dim)

        # Span representation size:
        # If use_span_mean_pooling: [start_word; end_word; mean_word; width_emb] -> 3*d_model + width_dim
        # Else: [start_word; end_word; width_emb] -> 2*d_model + width_dim
        span_rep_dim = (3 if self.use_span_mean_pooling else 2) * d_model + width_dim

        # 3. Mention Module (ATE & OTE Supervision: Target, Opinion, Invalid)
        self.mention_classifier = MLP(
            in_dim=span_rep_dim,
            hidden_dim=hidden_dim,
            out_dim=3,
            dropout=dropout,
        )

        # 4. Triplet / Relation Module (Positive, Negative, Neutral, Invalid)
        if self.use_biaffine:
            self.relation_classifier = BiaffineSpanRelationClassifier(
                span_dim=span_rep_dim,
                biaffine_dim=biaffine_dim,
                distance_dim=distance_dim,
                out_dim=4,
                use_span_sync=use_span_sync,
                dropout=dropout,
            )
        else:
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
          If use_span_mean_pooling:
            s_{i, j} = [x_s; x_{e-1}; x_mean; f_width(e - s)]
          Else:
            s_{i, j} = [x_s; x_{e-1}; f_width(e - s)]
        Returns:
          span_reps: (K, (3 or 2)*d + width_dim)
          width_bucket_ids: (K,)
        """
        d = word_hidden.size(-1)
        expected_dim = (3 if self.use_span_mean_pooling else 2) * d + self.width_embedding.embedding_dim
        if not spans:
            return torch.empty((0, expected_dim), device=device), torch.empty((0,), dtype=torch.long, device=device)

        num_words = word_hidden.size(0)
        max_idx = max(num_words - 1, 0)
        starts = torch.clamp(torch.tensor([s for s, e in spans], dtype=torch.long, device=device), 0, max_idx)
        ends = torch.clamp(torch.tensor([e - 1 for s, e in spans], dtype=torch.long, device=device), 0, max_idx)
        ends_excl = torch.clamp(torch.tensor([e for s, e in spans], dtype=torch.long, device=device), 0, num_words)
        widths = torch.clamp(torch.tensor([bucket_value(e - s) for s, e in spans], dtype=torch.long, device=device), 0, 9)

        start_reps = word_hidden[starts]  # (K, d)
        end_reps = word_hidden[ends]      # (K, d)
        width_reps = self.width_embedding(widths)  # (K, width_dim)

        if self.use_span_mean_pooling:
            # O(1) Prefix-sum vectorized span mean pooling across all candidate spans
            # prefix: (num_words + 1, d) where prefix[0] is zero
            prefix = torch.cat([
                torch.zeros((1, d), device=device, dtype=word_hidden.dtype),
                torch.cumsum(word_hidden, dim=0),
            ], dim=0)
            sum_reps = prefix[ends_excl] - prefix[starts]  # (K, d)
            span_lengths = torch.clamp(
                torch.tensor([e - s for s, e in spans], dtype=word_hidden.dtype, device=device),
                min=1.0,
            ).unsqueeze(-1)  # (K, 1)
            mean_reps = sum_reps / span_lengths  # (K, d)
            span_reps = torch.cat([start_reps, end_reps, mean_reps, width_reps], dim=-1)
        else:
            span_reps = torch.cat([start_reps, end_reps, width_reps], dim=-1)

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
        relation_threshold: float | None = None,
    ) -> dict:
        """
        Forward pass for a batch of sequences.
        Because each sentence has dynamic span counts, spans and pairs are handled per example.
        If relation_threshold is provided, predicts relations where P(Relation) >= relation_threshold
        instead of hard argmax != INVALID.
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
                    "candidate_pairs_with_scores": [],
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
                if m_labels.numel() > 0 and mention_logits.size(0) == m_labels.size(0):
                    m_labels = torch.clamp(m_labels, 0, 2)
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
                max_span_idx = max(span_reps.size(0) - 1, 0)
                t_idx_t = torch.clamp(torch.tensor(top_t_indices, dtype=torch.long, device=device), 0, max_span_idx)
                o_idx_t = torch.clamp(torch.tensor(top_o_indices, dtype=torch.long, device=device), 0, max_span_idx)

                t_reps = span_reps[t_idx_t]  # (num_t, span_dim)
                o_reps = span_reps[o_idx_t]  # (num_o, span_dim)

                # Compute pairwise token distances on CPU list then move to CUDA tensor once
                dist_matrix = [
                    [bucket_value(compute_span_distance(t_span, o_span)) for o_span in pruned_opinion_spans]
                    for t_span in pruned_target_spans
                ]
                dist_indices = torch.clamp(torch.tensor(dist_matrix, dtype=torch.long, device=device), 0, 9)

                dist_reps = self.distance_embedding(dist_indices)  # (num_t, num_o, dist_dim)

                if self.use_biaffine and isinstance(self.relation_classifier, BiaffineSpanRelationClassifier):
                    relation_logits = self.relation_classifier(t_reps, o_reps, dist_reps)  # (num_t * num_o, 4)
                else:
                    # Pair expansion for legacy MLP
                    t_expanded = t_reps.unsqueeze(1).expand(num_t, num_o, -1)  # (num_t, num_o, span_dim)
                    o_expanded = o_reps.unsqueeze(0).expand(num_t, num_o, -1)  # (num_t, num_o, span_dim)
                    pair_reps = torch.cat([t_expanded, o_expanded, dist_reps], dim=-1)  # (num_t, num_o, pair_dim)
                    flat_pair_reps = pair_reps.contiguous().reshape(num_t * num_o, -1)
                    relation_logits = self.relation_classifier(flat_pair_reps)  # (num_t * num_o, 4)

                # Relation Loss
                if gold_pairs is not None and b < len(gold_pairs):
                    gold_dict = {(p[0], p[1]): p[2] for p in gold_pairs[b]}
                    pair_labels_list = [
                        gold_dict.get((t_span, o_span), RELATION2ID["INVALID"])
                        for t_span in pruned_target_spans
                        for o_span in pruned_opinion_spans
                    ]
                    pair_labels = torch.clamp(torch.tensor(pair_labels_list, dtype=torch.long, device=device), 0, 3)

                    if self.use_focal_loss:
                        # Multi-Class Focal Loss (Lin et al., ICCV 2017):
                        # FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
                        ce_raw = F.cross_entropy(relation_logits, pair_labels, reduction="none")
                        pt = torch.exp(-ce_raw)  # True class probability p_t in [0, 1]
                        focal_modulator = (1.0 - pt) ** self.focal_gamma
                        if self.relation_weights is not None:
                            alpha = self.relation_weights[pair_labels]
                            r_loss = (alpha * focal_modulator * ce_raw).mean()
                        else:
                            r_loss = (focal_modulator * ce_raw).mean()
                    else:
                        r_loss = F.cross_entropy(
                            relation_logits,
                            pair_labels,
                            weight=self.relation_weights,
                        )
                    total_relation_loss = total_relation_loss + r_loss
                    num_relation_examples += 1

                # Decode Predictions & Collect Candidate Pairs with Scores
                pred_triplets = []
                candidate_pairs_with_scores = []

                rel_probs = F.softmax(relation_logits, dim=-1)  # (num_t * num_o, 4)
                # P(Relation) = 1.0 - P(INVALID)
                p_rel = (1.0 - rel_probs[:, RELATION2ID["INVALID"]]).contiguous().reshape(num_t, num_o).tolist()
                best_senti_ids = (rel_probs[:, 1:].argmax(dim=-1) + 1).contiguous().reshape(num_t, num_o).tolist()
                pred_rel_ids = relation_logits.argmax(dim=-1).contiguous().reshape(num_t, num_o).tolist()

                for ti, t_span in enumerate(pruned_target_spans):
                    for oi, o_span in enumerate(pruned_opinion_spans):
                        score = p_rel[ti][oi]
                        senti = best_senti_ids[ti][oi]
                        candidate_pairs_with_scores.append((t_span, o_span, senti, score))

                        if relation_threshold is None:
                            rel_id = pred_rel_ids[ti][oi]
                            if rel_id != RELATION2ID["INVALID"]:
                                pred_triplets.append((t_span, o_span, rel_id))
                        else:
                            if score >= relation_threshold:
                                pred_triplets.append((t_span, o_span, senti))
            else:
                relation_logits = None
                pred_triplets = []
                candidate_pairs_with_scores = []

            batch_outputs.append({
                "target_candidates": pruned_target_spans,
                "opinion_candidates": pruned_opinion_spans,
                "pred_triplets": pred_triplets,
                "candidate_pairs_with_scores": candidate_pairs_with_scores,
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
