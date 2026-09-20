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


if TORCH_AVAILABLE:

    class LinearChainCRF(nn.Module):
        """
        Linear-Chain CRF supporting batch-first tensors and BIO constraints.
        Tags: 0: 'O', 1: 'B', 2: 'I'.
        """

        def __init__(
            self,
            num_tags: int = 3,
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

            self.reset_parameters()

        def reset_parameters(self):
            nn.init.uniform_(self.transitions, -0.1, 0.1)
            nn.init.uniform_(self.start_transitions, -0.1, 0.1)
            nn.init.uniform_(self.end_transitions, -0.1, 0.1)

            if self.enforce_bio_constraints and self.num_tags == 3:
                # 0: O, 1: B, 2: I
                # Disallow START -> I and O -> I
                with torch.no_grad():
                    self.start_transitions[2] = -10000.0
                    self.transitions[0, 2] = -10000.0

        def _apply_bio_weights(self, emissions: torch.Tensor) -> torch.Tensor:
            """Scales emissions by class weight -- used ONLY inside the NLL
            loss (forward()), never inside decode(). Applying this at
            decode time would bias every prediction toward whichever class
            has the larger weight (typically B/I, to counter O-dominance),
            regardless of actual model confidence -- that's not what class
            weighting during training is meant to do to inference."""
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
                else:
                    packed_mask[b, 0] = True

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

            # Transpose to (T_packed, B, K)
            feats = packed_emissions.transpose(0, 1)  # (T, B, K)
            tags_t = packed_tags.transpose(0, 1)  # (T, B)
            mask_t = packed_mask.transpose(0, 1)  # (T, B)
            T_packed = feats.shape[0]

            # 1. Forward algorithm: partition function log Z
            forward_var = self.start_transitions.unsqueeze(0) + feats[0]  # (B, K)

            for t in range(1, T_packed):
                emit_score = feats[t].unsqueeze(1)  # (B, 1, K)
                trans_score = self.transitions.unsqueeze(0)  # (1, K, K)
                next_var = forward_var.unsqueeze(2) + trans_score + emit_score  # (B, K, K)
                next_var = torch.logsumexp(next_var, dim=1)  # (B, K)
                forward_var = torch.where(mask_t[t].unsqueeze(1), next_var, forward_var)

            terminal_var = forward_var + self.end_transitions.unsqueeze(0)  # (B, K)
            log_Z = torch.logsumexp(terminal_var, dim=-1)  # (B,)

            # 2. Gold path score
            gold_score = self.start_transitions[tags_t[0]] + feats[0].gather(
                1, tags_t[0].unsqueeze(1)
            ).squeeze(1)

            for t in range(1, T_packed):
                trans = self.transitions[tags_t[t - 1], tags_t[t]]  # (B,)
                emit = feats[t].gather(1, tags_t[t].unsqueeze(1)).squeeze(1)  # (B,)
                gold_score = gold_score + (trans + emit) * mask_t[t].float()

            last_indices = (lengths - 1).clamp(min=0).unsqueeze(0)  # (1, B)
            last_tags = torch.gather(tags_t, 0, last_indices).squeeze(0)  # (B,)
            gold_score = gold_score + self.end_transitions[last_tags]

            nll = log_Z - gold_score

            if reduction == "mean":
                return nll.mean()
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

            viterbi_var = self.start_transitions.unsqueeze(0) + feats[0]  # (B, K)
            backpointers = []

            for t in range(1, T_packed):
                emit_score = feats[t].unsqueeze(1)  # (B, 1, K)
                trans_score = self.transitions.unsqueeze(0)  # (1, K, K)
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