from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from final_model.losses import coral_expected_value, coral_loss
from .coral_head import CoralHead


@dataclass
class NliExpertConfig:
    model_name: str = "roberta-large-mnli"
    num_classes: int = 5
    dropout: float = 0.1
    hidden_dim: int = 512
    max_length: int = 256
    pooling: str = "cls"  # "cls" or "mean"


def make_nli_collate_fn(tokenizer: PreTrainedTokenizerBase, max_length: int):
    """Tokenize premise/hypothesis pairs for the NLI expert."""

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        premises = [sample["premise"] for sample in batch]
        hypotheses = [sample["hypothesis"] for sample in batch]
        enc = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = torch.stack([sample["ordinal_label"] for sample in batch], dim=0)
        scores = torch.stack([sample["label"] for sample in batch], dim=0)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "ordinal_labels": labels,
            "scores": scores,
            "sample_ids": [sample["sample_id"] for sample in batch],
        }

    return collate


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
        self.head = CoralHead(
            hidden_size,
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.classifier = torch.nn.Linear(hidden_size, config.num_classes)

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
        logits = self.head(cls)
        return {
            "ordinal_logits": logits,
            "cls": cls,
            "class_logits": self.classifier(cls),
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
