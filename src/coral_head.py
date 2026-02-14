"""CORAL head modules for ordinal regression experts.
Created by Moritz Rengert.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class CoralHead(nn.Module):
    """Small MLP that outputs K-1 logits for CORAL."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        if hidden_dim and hidden_dim > 0:
            layers.extend(
                [
                    nn.LayerNorm(input_dim),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes - 1),
                ]
            )
        else:
            layers.extend(
                [
                    nn.LayerNorm(input_dim),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim, num_classes - 1),
                ]
            )
        self.net = nn.Sequential(*layers)
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
