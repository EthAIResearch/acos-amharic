"""
Shared-encoder span-pair classifier -- used by Stage 3 (category, 22-way)
and Stage 4 (sentiment, 3-way). Pools the aspect span and opinion span
representations (masked mean over their subword tokens) plus the sentence's
pooled representation, concatenates, and classifies via a small MLP.
"""
import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class PairClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        class_weights: list | None = None,
        log_priors: list | None = None,
        tau: float = 1.0,
        dropout: float = 0.1,
    ):
        """
        class_weights: inverse-frequency weights for weighted CrossEntropyLoss
            (the simpler, weaker imbalance fix -- see pair_utils.compute_class_weights).
        log_priors: per-class log(count/total) for logit-adjusted loss (Menon et
            al. 2021) -- the more effective fix for severe long-tail imbalance.
            Pass at most ONE of class_weights / log_priors, not both -- combining
            them tends to over-correct.
        tau: logit adjustment strength (only used if log_priors is set). 1.0 is
            the standard default; try 0.5-2.0 if tuning.
        """
        super().__init__()
        assert not (class_weights is not None and log_priors is not None), \
            "Pass class_weights OR log_priors, not both -- combining tends to over-correct."

        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden),  # [aspect_pooled; opinion_pooled; sentence_pooled]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

        self.tau = tau
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float))
        else:
            self.class_weights = None

        if log_priors is not None:
            self.register_buffer("log_priors", torch.tensor(log_priors, dtype=torch.float))
        else:
            self.log_priors = None

    @staticmethod
    def _masked_mean_pool(hidden_states, mask):
        # hidden_states: (B, T, H), mask: (B, T)
        mask = mask.unsqueeze(-1).float()  # (B, T, 1)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(self, input_ids, attention_mask, aspect_mask, opinion_mask, label=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = self.dropout(out.last_hidden_state)  # (B, T, H)

        aspect_pooled = self._masked_mean_pool(seq_out, aspect_mask)
        opinion_pooled = self._masked_mean_pool(seq_out, opinion_mask)
        sentence_pooled = self._masked_mean_pool(seq_out, attention_mask)

        combined = torch.cat([aspect_pooled, opinion_pooled, sentence_pooled], dim=-1)
        logits = self.classifier(combined)  # raw logits -- always used for prediction/argmax

        loss = None
        if label is not None:
            if self.log_priors is not None:
                # Logit-adjusted cross-entropy (Menon et al. 2021): shift the
                # decision boundary toward minority classes during training
                # by adding tau*log_prior to each class's logit before the
                # softmax. The model's raw `logits` (returned above,
                # unadjusted) are still what's used for prediction -- the
                # adjustment only shapes what the model learns to output.
                adjusted_logits = logits + self.tau * self.log_priors.unsqueeze(0)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(adjusted_logits, label)
            else:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(logits, label)

        return {"loss": loss, "logits": logits}
