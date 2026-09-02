"""
Shared-encoder, dual-head token classification model for joint ATE + OTE.
"""
from torch import nn
from transformers import AutoConfig, AutoModel

NUM_BIO_LABELS = 3  # O, B, I


class JointTaggingModel(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.aspect_head = nn.Linear(hidden, NUM_BIO_LABELS)
        self.opinion_head = nn.Linear(hidden, NUM_BIO_LABELS)

    def forward(self, input_ids, attention_mask, aspect_labels=None, opinion_labels=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = self.dropout(out.last_hidden_state)  # (B, T, H)

        aspect_logits = self.aspect_head(seq_out)   # (B, T, 3)
        opinion_logits = self.opinion_head(seq_out)  # (B, T, 3)

        loss = None
        if aspect_labels is not None and opinion_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss_a = loss_fct(aspect_logits.view(-1, NUM_BIO_LABELS), aspect_labels.view(-1))
            loss_o = loss_fct(opinion_logits.view(-1, NUM_BIO_LABELS), opinion_labels.view(-1))
            loss = loss_a + loss_o

        return {
            "loss": loss,
            "aspect_logits": aspect_logits,
            "opinion_logits": opinion_logits,
        }
