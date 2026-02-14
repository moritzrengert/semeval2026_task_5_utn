"""SBERT expert model definitions.
Created by Moritz Rengert.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from transformers import AutoModel, AutoTokenizer

from .losses import coral_expected_value, coral_loss
from .coral_head import CoralHead


@dataclass
class SbertExpertConfig:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    num_classes: int = 5
    dropout: float = 0.1
    hidden_dim: int = 512
    max_length: int = 256
    projector_dim: int = 256
    use_classifier: bool = True


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


class SbertSemanticMatchingExpert(torch.nn.Module):
    """Frozen all-mpnet-base-v2 encoder with pooled embeddings + CORAL head."""

    def __init__(self, config: SbertExpertConfig):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.finetune = False
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        concat_dim = hidden_size * 2
        self.projector = None
        head_input_dim = concat_dim
        if self.config.projector_dim and self.config.projector_dim > 0:
            self.projector = torch.nn.Sequential(
                torch.nn.LayerNorm(concat_dim),
                torch.nn.Dropout(config.dropout),
                torch.nn.Linear(concat_dim, config.projector_dim),
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

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        gloss_ids: torch.Tensor,
        gloss_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with torch.set_grad_enabled(self.finetune):
            ctx_out = self.encoder(input_ids=context_ids, attention_mask=context_mask)
            gloss_out = self.encoder(input_ids=gloss_ids, attention_mask=gloss_mask)
            ctx_emb = mean_pool(ctx_out.last_hidden_state, context_mask)
            gloss_emb = mean_pool(gloss_out.last_hidden_state, gloss_mask)
        features = torch.cat([ctx_emb, gloss_emb], dim=-1)
        if self.projector is not None:
            features = self.projector(features)
        logits = self.head(features)
        return {
            "ordinal_logits": logits,
            "context_emb": ctx_emb,
            "gloss_emb": gloss_emb,
            "features": features,
            "class_logits": self.classifier(features) if self.classifier is not None else None,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.forward(
            batch["context_ids"],
            batch["context_mask"],
            batch["gloss_ids"],
            batch["gloss_mask"],
        )
        loss = coral_loss(outputs["ordinal_logits"], batch["ordinal_labels"], num_classes=self.num_classes)
        return {"loss": loss, **outputs}

    @torch.no_grad()
    def predict_scores(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.forward(
            batch["context_ids"],
            batch["context_mask"],
            batch["gloss_ids"],
            batch["gloss_mask"],
        )["ordinal_logits"]
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
