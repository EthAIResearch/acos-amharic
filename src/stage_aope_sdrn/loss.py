"""
Loss functions for Joint AOPE SDRN.

Includes:
- MultiClassDiceLoss: Soft Sørensen-Dice loss for multi-class sequence labeling
  (Li et al., ACL 2020: "Dice Loss for Data-imbalanced NLP Tasks").
  Directly optimizes soft token-level F1 for minority entity classes (B-ASP, I-ASP, B-OPN, I-OPN)
  to counteract severe class imbalance (85%+ 'O' tokens).
"""
try:
    import torch
    from torch import nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class MultiClassDiceLoss(nn.Module):
        """
        Multi-Class Sørensen-Dice Loss for token-level sequence labeling.

        Computes:
            Dice_k = (2 * sum(p_{i,k} * y_{i,k}) + smooth) / (sum(p_{i,k}^2) + sum(y_{i,k}^2) + smooth)
            Loss_k = 1.0 - Dice_k

        Supports:
            - Class weighting (e.g. higher penalty on B-OPN, I-OPN).
            - ignore_index (-100) and sequence masking.
            - Float16 mixed precision stability.
        """

        def __init__(
            self,
            num_classes: int = 5,
            smooth: float = 1.0,
            square: bool = True,
            weight: torch.Tensor | list[float] | None = None,
            ignore_index: int = -100,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.smooth = smooth
            self.square = square
            self.ignore_index = ignore_index

            if weight is not None:
                if isinstance(weight, list):
                    weight = torch.tensor(weight, dtype=torch.float32)
                self.register_buffer("weight", weight.float())
            else:
                self.weight = None

        def forward(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """
            logits: (B, T, K) raw unnormalized emissions/logits.
            labels: (B, T) ground truth class indices in [0, K-1], or ignore_index (-100).
            mask: (B, T) boolean mask (True for content tokens, False for padding/special).
            """
            valid = labels != self.ignore_index
            if mask is not None:
                valid = valid & mask.bool()

            if not valid.any():
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

            probs = F.softmax(logits, dim=-1)  # (B, T, K)

            probs_valid = probs[valid]    # (M, K)
            labels_valid = labels[valid]  # (M,)

            # One-hot encode targets
            one_hot = F.one_hot(labels_valid, num_classes=self.num_classes).to(probs_valid.dtype)  # (M, K)

            # Intersection and denominator per class
            intersection = 2.0 * torch.sum(probs_valid * one_hot, dim=0)  # (K,)

            if self.square:
                denominator = torch.sum(probs_valid**2, dim=0) + torch.sum(one_hot**2, dim=0)  # (K,)
            else:
                denominator = torch.sum(probs_valid, dim=0) + torch.sum(one_hot, dim=0)

            dice_k = (intersection + self.smooth) / (denominator + self.smooth)
            loss_k = 1.0 - dice_k

            if self.weight is not None:
                w = self.weight.to(loss_k.device)
                return torch.sum(loss_k * w) / (torch.sum(w) + 1e-8)

            return torch.mean(loss_k)

else:
    # Fallback placeholder when PyTorch is not installed in the local environment
    class MultiClassDiceLoss:
        pass
