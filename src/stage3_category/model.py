"""
Shared-encoder span-pair classifier: pools the aspect span and opinion span
representations (masked mean over their subword tokens) plus the sentence's
[CLS]/pooled representation, concatenates, and classifies.

Reused as-is for Stage 4 (sentiment) -- only num_labels and the label field
in the dataset differ.
"""
import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class PairClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, class_weights: list | None = None, dropout: float = 0.1):
        super().__init__()
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

        self.class_weights = None
        if class_weights is not None:
            self.register_buffer("_class_weights", torch.tensor(class_weights, dtype=torch.float))
            self.class_weights = self._class_weights

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
        logits = self.classifier(combined)

        loss = None
        if label is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits, label)

        return {"loss": loss, "logits": logits}
