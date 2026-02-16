"""NLI expert model definitions.
Created by Moritz Rengert.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from transformers import AutoModel, AutoTokenizer

from losses import coral_expected_value, coral_loss
from coral_head import CoralHead


@dataclass
class NliExpertConfig:
    # Best settings from latest target-aware sweep (runs/sweep_target_aware_v3, trial 68).
    model_name: str = "roberta-large-mnli"
    num_classes: int = 5
    dropout: float = 0.15
    hidden_dim: int = 256
    max_length: int = 256
    pooling: str = "mean"  # "cls" or "mean"
    projector_dim: int = 256
    use_classifier: bool = True
    # Training defaults
    batch_size: int = 16
    epochs: int = 10
    lr: float = 1e-4
    encoder_lr: float = 0.0
    weight_decay: float = 0.03
    warmup_ratio: float = 0.1
    train_encoder_layers: int = 0
    coral_weight: float = 1.0
    mse_weight: float = 0.0
    soft_weight: float = 0.0
    max_grad_norm: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999
    early_stop_patience: int = 2
    early_stop_delta: float = 0.0
    overfit_patience: int = 2
    overfit_train_delta: float = 0.01
    overfit_dev_delta: float = 0.0
    expand_annotators: bool = False
    log_every: int = 20


class NliPlausibilityExpert(torch.nn.Module):
    """Frozen RoBERTa-MNLI encoder + CORAL head."""

    def __init__(self, config: NliExpertConfig):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.finetune = False
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        self.projector = None
        head_input_dim = hidden_size
        if self.config.projector_dim and self.config.projector_dim > 0:
            self.projector = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_size),
                torch.nn.Dropout(config.dropout),
                torch.nn.Linear(hidden_size, config.projector_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(config.dropout),
            )
            head_input_dim = config.projector_dim
        self.head = CoralHead(
            head_input_dim,
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.classifier = torch.nn.Linear(head_input_dim, config.num_classes) if self.config.use_classifier else None

    @property
    def num_classes(self) -> int:
        return self.config.num_classes

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.set_grad_enabled(self.finetune):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            if self.config.pooling == "mean":
                mask = attention_mask.unsqueeze(-1).float()
                summed = (outputs.last_hidden_state * mask).sum(dim=1)
                cls = summed / mask.sum(dim=1).clamp(min=1e-6)
            else:
                cls = outputs.last_hidden_state[:, 0, :]
        if self.projector is not None:
            cls = self.projector(cls)
        logits = self.head(cls)
        return {
            "ordinal_logits": logits,
            "cls": cls,
            "class_logits": self.classifier(cls) if self.classifier is not None else None,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = coral_loss(outputs["ordinal_logits"], batch["ordinal_labels"], num_classes=self.num_classes)
        return {"loss": loss, **outputs}

    @torch.no_grad()
    def predict_scores(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.forward(batch["input_ids"], batch["attention_mask"])["ordinal_logits"]
        return coral_expected_value(logits) + 1.0

    def unfreeze_last_layers(self, n_layers: int) -> None:
        if n_layers <= 0:
            return
        layers = getattr(self.encoder, "encoder").layer
        for layer in layers[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.finetune = True
        self.encoder.train()
