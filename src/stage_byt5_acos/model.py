"""
ByT5 Seq2Seq Model Wrapper for Amharic ACOS Quadruple Extraction
================================================================
Wraps HuggingFace AutoModelForSeq2SeqLM for ByT5, providing generation
helpers, optimizer configurations (Adafactor / AdamW), and checkpoint loading.
"""
import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM


class ByT5ACOSModel(nn.Module):
    """
    ByT5 Seq2Seq Model for end-to-end ACOS quadruple extraction.
    """

    def __init__(self, model_name: str = "google/byt5-base"):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
    ):
        """
        Forward pass computing cross-entropy loss if labels are provided.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 384,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generates byte token IDs autoregressively.
        Default is greedy search (num_beams=1) for fast evaluation.
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )


def build_optimizer(
    model: nn.Module,
    optimizer_type: str = "adafactor",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
):
    """
    Builds optimizer. Adafactor is strongly recommended for ByT5 because factorized
    second moments drastically reduce GPU VRAM consumption.
    """
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    opt_lower = optimizer_type.lower()
    if opt_lower == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(
            optimizer_grouped_parameters,
            lr=lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    elif opt_lower == "adamw":
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Choose 'adafactor' or 'adamw'.")
