"""
Linear-Chain Conditional Random Field (CRF) for sequence labeling in Joint AOPE (SDRN).
Self-contained pure PyTorch implementation with zero external library dependencies.

Provides:
  - Exact Negative Log-Likelihood (NLL) loss via the forward algorithm in log-space.
  - Vectorized Viterbi decoding for maximum-a-posteriori sequence inference.
  - Syntactic BIO transition constraints (forbids O -> I and START -> I).
  - Class-weight emission scaling (supports bio_class_weights for class imbalance).
  - Pure-Python reference decoder for dependency-free testing.
"""

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def viterbi_decode_reference(
    emissions: list[list[float]],
    transitions: list[list[float]],
    start_transitions: list[float],
    end_transitions: list[float],
) -> list[int]:
    """
    Pure-Python reference Viterbi decoder for a single sequence of length T.
    emissions: T x K
    transitions: K x K, transitions[from_tag][to_tag]
    start_transitions: K
    end_transitions: K
    """
    T = len(emissions)
    if T == 0:
        return []
    K = len(start_transitions)

    # viterbi_var at step 0
    viterbi_var = [start_transitions[k] + emissions[0][k] for k in range(K)]
    backpointers = []

    for t in range(1, T):
        next_var = []
        bptrs = []
        for j in range(K):
            # Best incoming tag i -> j
            best_score = -float("inf")
            best_i = 0
            for i in range(K):
                score = viterbi_var[i] + transitions[i][j] + emissions[t][j]
                if score > best_score:
                    best_score = score
                    best_i = i
            next_var.append(best_score)
            bptrs.append(best_i)
        viterbi_var = next_var
        backpointers.append(bptrs)

    # Terminal step
    best_last_tag = 0
    best_total_score = -float("inf")
    for k in range(K):
        score = viterbi_var[k] + end_transitions[k]
        if score > best_total_score:
            best_total_score = score
            best_last_tag = k

    # Backtrace
    path = [best_last_tag]
    cur_tag = best_last_tag
    for t in range(T - 2, -1, -1):
        cur_tag = backpointers[t][cur_tag]
        path.append(cur_tag)
    path.reverse()
    return path


def validate_bio_path_reference(path: list[int]) -> bool:
    """
    Pure-Python reference validator for BIO tag sequence constraints (3-tag).
    Returns True if valid, False if it starts with 'I' (2) or contains 'O' -> 'I' (0 -> 2).
    Empty paths are valid.
    """
    if not path:
        return True
    if path[0] == 2:
        return False
    for t in range(1, len(path)):
        if path[t - 1] == 0 and path[t] == 2:
            return False
    return True


def validate_5way_bio_path_reference(path: list[int]) -> bool:
    """
    Pure-Python reference validator for unified 5-way BIO tag constraints:
      0: 'O', 1: 'B-ASP', 2: 'I-ASP', 3: 'B-OPN', 4: 'I-OPN'
    Returns False if sequence starts with I-ASP (2) or I-OPN (4),
    or contains transitions: O -> I-*, ASP -> I-OPN, OPN -> I-ASP.
    """
    if not path:
        return True
    if path[0] in (2, 4):
        return False
    forbidden = {(0, 2), (0, 4), (1, 4), (2, 4), (3, 2), (4, 2)}
    for t in range(1, len(path)):
        if (path[t - 1], path[t]) in forbidden:
            return False
    return True


if TORCH_AVAILABLE:

    class LinearChainCRF(nn.Module):
        """
        Linear-Chain CRF supporting batch-first tensors and BIO constraints.
        Supports 3 tags (O, B, I) and 5 tags (O, B-ASP, I-ASP, B-OPN, I-OPN).
        """

        def __init__(
            self,
            num_tags: int = 5,
            bio_weights: torch.Tensor | list[float] | None = None,
            enforce_bio_constraints: bool = True,
        ):
            super().__init__()
            self.num_tags = num_tags
            self.enforce_bio_constraints = enforce_bio_constraints

            # transitions[i, j] = score of transitioning from tag i to tag j
            self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
            self.start_transitions = nn.Parameter(torch.empty(num_tags))
            self.end_transitions = nn.Parameter(torch.empty(num_tags))

            if bio_weights is not None:
                if not isinstance(bio_weights, torch.Tensor):
                    bio_w = torch.tensor(bio_weights, dtype=torch.float32)
                else:
                    bio_w = bio_weights.clone().detach().float()
            else:
                bio_w = torch.ones(num_tags, dtype=torch.float32)
            self.register_buffer("bio_weights", bio_w)

            if self.enforce_bio_constraints and self.num_tags == 3:
                # 0: O, 1: B, 2: I
                # Disallow START -> I and O -> I via fixed boolean mask buffers
                trans_mask = torch.zeros(num_tags, num_tags, dtype=torch.bool)
                trans_mask[0, 2] = True
                start_mask = torch.zeros(num_tags, dtype=torch.bool)
                start_mask[2] = True
                self.register_buffer("forbidden_trans_mask", trans_mask)
                self.register_buffer("forbidden_start_mask", start_mask)
            elif self.enforce_bio_constraints and self.num_tags == 5:
                # 0: O, 1: B-ASP, 2: I-ASP, 3: B-OPN, 4: I-OPN
                # Disallow START -> I-ASP, START -> I-OPN
                # Disallow O -> I-ASP, O -> I-OPN
                # Disallow B-ASP -> I-OPN, I-ASP -> I-OPN
                # Disallow B-OPN -> I-ASP, I-OPN -> I-ASP
                trans_mask = torch.zeros(num_tags, num_tags, dtype=torch.bool)
                trans_mask[0, 2] = True
                trans_mask[0, 4] = True
                trans_mask[1, 4] = True
                trans_mask[2, 4] = True
                trans_mask[3, 2] = True
                trans_mask[4, 2] = True
                start_mask = torch.zeros(num_tags, dtype=torch.bool)
                start_mask[2] = True
                start_mask[4] = True
                self.register_buffer("forbidden_trans_mask", trans_mask)
                self.register_buffer("forbidden_start_mask", start_mask)
            else:
                self.forbidden_trans_mask = None
                self.forbidden_start_mask = None

            self.reset_parameters()

        def reset_parameters(self):
            nn.init.uniform_(self.transitions, -0.1, 0.1)
            nn.init.uniform_(self.start_transitions, -0.1, 0.1)
            nn.init.uniform_(self.end_transitions, -0.1, 0.1)

        def _get_constrained_transitions(self, mask_value: float = -float("inf")):
            trans = self.transitions
            start = self.start_transitions
            if self.forbidden_trans_mask is not None:
                trans = trans.masked_fill(self.forbidden_trans_mask, mask_value)
            if self.forbidden_start_mask is not None:
                start = start.masked_fill(self.forbidden_start_mask, mask_value)
            return trans, start

        def _validate_gold_paths(
            self,
            packed_tags: torch.Tensor,
            packed_mask: torch.Tensor,
            lengths: torch.Tensor,
        ):
            """
            Validates that active gold sequences obey syntactic BIO constraints:
              - An active sequence cannot start with a forbidden tag (e.g. I-tag).
              - An active sequence cannot execute a forbidden transition.
            Empty/inactive sequences (lengths == 0) are ignored.
            Raises ValueError if any active gold path violates BIO syntax.
            """
            active = lengths > 0
            if not active.any():
                return

            # Check 1: START transitions
            if self.forbidden_start_mask is not None:
                starts_forbidden = active & self.forbidden_start_mask[packed_tags[:, 0]]
                if starts_forbidden.any():
                    bad_idx = int(torch.nonzero(starts_forbidden, as_tuple=True)[0][0].item())
                    bad_tag = int(packed_tags[bad_idx, 0].item())
                    raise ValueError(
                        f"Invalid BIO gold path for example {bad_idx}: active sequence starts with forbidden tag {bad_tag}."
                    )

            # Check 2: Forbidden step-to-step transitions
            if self.forbidden_trans_mask is not None and packed_tags.shape[1] > 1:
                valid_pairs = active.unsqueeze(1) & packed_mask[:, :-1] & packed_mask[:, 1:]
                from_tags = packed_tags[:, :-1]
                to_tags = packed_tags[:, 1:]
                is_forbidden = self.forbidden_trans_mask[from_tags, to_tags] & valid_pairs
                if is_forbidden.any():
                    match_indices = torch.nonzero(is_forbidden, as_tuple=True)
                    bad_idx = int(match_indices[0][0].item())
                    bad_pos = int(match_indices[1][0].item())
                    f_tag = int(from_tags[bad_idx, bad_pos].item())
                    t_tag = int(to_tags[bad_idx, bad_pos].item())
                    raise ValueError(
                        f"Invalid BIO gold path for example {bad_idx}: contains forbidden transition "
                        f"{f_tag} -> {t_tag} at step {bad_pos} -> {bad_pos + 1}."
                    )


        def _apply_bio_weights(self, emissions: torch.Tensor) -> torch.Tensor:
            """Scales emissions by class weight uniformly in both forward() and decode(),
            maintaining identical emission-to-transition calibration between training and inference."""
            if self.bio_weights is not None:
                return emissions * self.bio_weights.view(1, 1, -1)
            return emissions


        def _pack_valid_sequence(
            self,
            emissions: torch.Tensor,
            tags: torch.Tensor | None = None,
            mask: torch.Tensor | None = None,
        ):
            """
            Left-aligns valid subwords so that t=0 is the first valid subword
            for every example in the batch.
            """
            B, T, K = emissions.shape
            if mask is None:
                mask = torch.ones((B, T), dtype=torch.bool, device=emissions.device)
            else:
                mask = mask.bool()

            lengths = mask.sum(dim=1)  # (B,)
            max_len = int(lengths.max().item())
            if max_len == 0:
                max_len = 1

            packed_emissions = torch.zeros(
                (B, max_len, K), dtype=emissions.dtype, device=emissions.device
            )
            packed_tags = (
                torch.zeros((B, max_len), dtype=torch.long, device=emissions.device)
                if tags is not None
                else None
            )
            packed_mask = torch.zeros(
                (B, max_len), dtype=torch.bool, device=emissions.device
            )

            for b in range(B):
                l_b = int(lengths[b].item())
                if l_b > 0:
                    indices = torch.nonzero(mask[b], as_tuple=True)[0]
                    packed_emissions[b, :l_b] = emissions[b, indices]
                    if tags is not None:
                        packed_tags[b, :l_b] = tags[b, indices]
                    packed_mask[b, :l_b] = True

            return packed_emissions, packed_tags, packed_mask, lengths

        def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.Tensor,
            mask: torch.Tensor | None = None,
            reduction: str = "mean",
        ) -> torch.Tensor:
            """
            Computes the Negative Log-Likelihood (NLL) loss.
            emissions: (B, T, K)
            tags: (B, T) with tag indices in [0, K-1] (or -100 for ignored positions)
            mask: (B, T) boolean mask
            """
            emissions = self._apply_bio_weights(emissions)

            # If tags have -100, ensure mask excludes them
            if mask is None:
                mask = tags != -100
            else:
                mask = mask.bool() & (tags != -100)

            # Sanitize tags so -100 does not cause indexing errors during gather
            clean_tags = tags.clone()
            clean_tags[clean_tags == -100] = 0

            packed_emissions, packed_tags, packed_mask, lengths = (
                self._pack_valid_sequence(emissions, clean_tags, mask)
            )

            if self.enforce_bio_constraints and self.num_tags == 3:
                self._validate_gold_paths(packed_tags, packed_mask, lengths)

            # Transpose to (T_packed, B, K)
            feats = packed_emissions.transpose(0, 1)  # (T, B, K)
            tags_t = packed_tags.transpose(0, 1)  # (T, B)
            mask_t = packed_mask.transpose(0, 1)  # (T, B)
            T_packed = feats.shape[0]

            transitions, start_transitions = self._get_constrained_transitions()

            # 1. Forward algorithm: partition function log Z
            forward_var = start_transitions.unsqueeze(0) + feats[0]  # (B, K)

            for t in range(1, T_packed):
                emit_score = feats[t].unsqueeze(1)  # (B, 1, K)
                trans_score = transitions.unsqueeze(0)  # (1, K, K)
                next_var = forward_var.unsqueeze(2) + trans_score + emit_score  # (B, K, K)
                next_var = torch.logsumexp(next_var, dim=1)  # (B, K)
                forward_var = torch.where(mask_t[t].unsqueeze(1), next_var, forward_var)

            terminal_var = forward_var + self.end_transitions.unsqueeze(0)  # (B, K)
            log_Z = torch.logsumexp(terminal_var, dim=-1)  # (B,)

            # 2. Gold path score
            gold_score = start_transitions[tags_t[0]] + feats[0].gather(
                1, tags_t[0].unsqueeze(1)
            ).squeeze(1)

            for t in range(1, T_packed):
                trans = transitions[tags_t[t - 1], tags_t[t]]  # (B,)
                emit = feats[t].gather(1, tags_t[t].unsqueeze(1)).squeeze(1)  # (B,)
                gold_score = gold_score + (trans + emit) * mask_t[t].float()

            last_indices = (lengths - 1).clamp(min=0).unsqueeze(0)  # (1, B)
            last_tags = torch.gather(tags_t, 0, last_indices).squeeze(0)  # (B,)
            gold_score = gold_score + self.end_transitions[last_tags]

            nll_raw = log_Z - gold_score
            # Empty sequences contribute zero NLL
            nll = torch.where(lengths > 0, nll_raw, torch.zeros_like(nll_raw))

            if reduction == "mean":
                valid_count = (lengths > 0).sum().clamp(min=1)
                return nll.sum() / valid_count
            elif reduction == "sum":
                return nll.sum()
            return nll

        def decode(
            self, emissions: torch.Tensor, mask: torch.Tensor | None = None
        ) -> torch.Tensor:
            """
            Performs Viterbi decoding.
            emissions: (B, T, K)
            mask: (B, T) bool
            Returns:
              (B, T) torch.LongTensor with decoded tag IDs.
            """
            emissions = self._apply_bio_weights(emissions)
            B, T = emissions.shape[:2]


            if mask is None:
                mask = torch.ones((B, T), dtype=torch.bool, device=emissions.device)
            else:
                mask = mask.bool()

            packed_emissions, _, packed_mask, lengths = self._pack_valid_sequence(
                emissions, None, mask
            )

            feats = packed_emissions.transpose(0, 1)  # (T_packed, B, K)
            mask_t = packed_mask.transpose(0, 1)  # (T_packed, B)
            T_packed = feats.shape[0]

            transitions, start_transitions = self._get_constrained_transitions()

            viterbi_var = start_transitions.unsqueeze(0) + feats[0]  # (B, K)
            backpointers = []

            for t in range(1, T_packed):
                emit_score = feats[t].unsqueeze(1)  # (B, 1, K)
                trans_score = transitions.unsqueeze(0)  # (1, K, K)
                next_var = viterbi_var.unsqueeze(2) + trans_score + emit_score  # (B, K, K)
                max_var, bptr = torch.max(next_var, dim=1)  # (B, K)
                backpointers.append(bptr)
                viterbi_var = torch.where(mask_t[t].unsqueeze(1), max_var, viterbi_var)

            # Trace back per example and unpack into (B, T)
            pred_tensor = torch.zeros((B, T), dtype=torch.long, device=emissions.device)

            for b in range(B):
                l_b = int(lengths[b].item())
                if l_b <= 0:
                    continue

                score = viterbi_var[b] + self.end_transitions
                best_last_tag = int(torch.argmax(score).item())
                path = [best_last_tag]

                cur_tag = best_last_tag
                for t in range(l_b - 2, -1, -1):
                    cur_tag = int(backpointers[t][b, cur_tag].item())
                    path.append(cur_tag)
                path.reverse()

                indices = torch.nonzero(mask[b], as_tuple=True)[0]
                pred_tensor[b, indices] = torch.tensor(
                    path, dtype=torch.long, device=emissions.device
                )

            return pred_tensor