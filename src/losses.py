"""CORAL loss utilities for ordinal regression.
Created by Moritz Rengert.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def coral_levels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Create level targets for CORAL."""
    device = labels.device
    levels = torch.arange(num_classes - 1, device=device).unsqueeze(0)
    return (labels.unsqueeze(1) > levels).float()


def coral_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 5,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary cross-entropy across ordinal thresholds."""
    targets = coral_levels(labels, num_classes)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)


def coral_expected_value(logits: torch.Tensor) -> torch.Tensor:
    """Map CORAL logits to an expected value."""
    probs = torch.sigmoid(logits)
    return probs.sum(dim=1)
