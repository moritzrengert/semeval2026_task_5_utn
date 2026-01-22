from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


@dataclass
class EnsembleModelConfig:
    vocab_size: int
    num_classes: int = 5
    text_dim: int = 128
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ffn_dim: int = 256
    dropout: float = 0.1
    dense_feature_dims: Optional[Dict[str, int]] = None
    dense_projection_dim: int = 128
    hidden_dim: int = 256
    pad_idx: int = 0
    use_classification_head: bool = True


class TransformerTextEncoder(nn.Module):
    """Lightweight Transformer encoder for bagged task text."""

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        encoded = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        mask = attention_mask.unsqueeze(-1).float()
        masked_sum = (encoded * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        pooled = masked_sum / denom
        return self.layer_norm(pooled)


class DenseFeatureProjector(nn.Module):
    """Project arbitrary dense features into a shared space."""

    def __init__(
        self,
        feature_dims: Optional[Dict[str, int]],
        projection_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.projectors = nn.ModuleDict()
        feature_dims = feature_dims or {}
        for name, dim in feature_dims.items():
            self.projectors[name] = nn.Sequential(
                nn.Linear(dim, projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(projection_dim),
            )
        self.output_dim = len(self.projectors) * projection_dim

    def forward(self, features: Optional[Dict[str, torch.Tensor]]) -> Optional[torch.Tensor]:
        if not self.projectors:
            return None
        if features is None:
            raise ValueError("Dense features are required because dense projectors are configured.")
        projected = []
        for name, projector in self.projectors.items():
            if name not in features:
                raise KeyError(f"Dense features missing required key '{name}'")
            projected.append(projector(features[name]))
        return torch.cat(projected, dim=-1)


class CoralOrdinalHead(nn.Module):
    """Linear head that emits K-1 logits for CORAL loss."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes - 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class EnsembleRegressor(nn.Module):
    """Final-stage model combining text and optional dense ensemble signals."""

    def __init__(self, config: EnsembleModelConfig):
        super().__init__()
        self.text_encoder = TransformerTextEncoder(
            vocab_size=config.vocab_size,
            pad_idx=config.pad_idx,
            embed_dim=config.text_dim,
            num_layers=config.transformer_layers,
            num_heads=config.transformer_heads,
            ffn_dim=config.transformer_ffn_dim,
            dropout=config.dropout,
        )
        self.dense_projector = DenseFeatureProjector(
            feature_dims=config.dense_feature_dims,
            projection_dim=config.dense_projection_dim,
            dropout=config.dropout,
        )
        combined_dim = config.text_dim + self.dense_projector.output_dim
        self.feature_norm = nn.LayerNorm(combined_dim)
        self.reg_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(combined_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.ordinal_head = CoralOrdinalHead(combined_dim, config.num_classes)
        self.classifier = None
        if config.use_classification_head:
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(combined_dim, config.num_classes),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        dense_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        text_repr = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        dense_repr = self.dense_projector(dense_features)
        if dense_repr is None:
            features = text_repr
        else:
            features = torch.cat([text_repr, dense_repr], dim=-1)
        features = self.feature_norm(features)
        regression = self.reg_head(features).squeeze(-1)
        ordinal_logits = self.ordinal_head(features)
        class_logits = self.classifier(features) if self.classifier else None
        return {
            "regression": regression,
            "ordinal_logits": ordinal_logits,
            "class_logits": class_logits,
            "features": features,
        }
