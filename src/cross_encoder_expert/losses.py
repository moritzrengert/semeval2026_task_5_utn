"""Regression losses (same as cross_encoder_adam)."""

import torch
from torch import nn


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred, target):
        return self.mse(pred, target)


class TranslatedReLULoss(nn.Module):
    def __init__(self, x0: float = 0.15, k: float = 1.0):
        super().__init__()
        self.x0 = float(x0)
        self.k = float(k)

    def forward(self, pred, target):
        x = (pred - target).abs()
        return torch.relu(self.k * (x - self.x0))


class SmoothK2Loss(nn.Module):
    def __init__(self, x0: float = 0.15, k: float = 1.0):
        super().__init__()
        self.x0 = float(x0)
        self.k = float(k)

    def forward(self, pred, target):
        x = (pred - target).abs()
        quad = self.k * (x * x - 2 * self.x0 * x + self.x0 * self.x0)
        return torch.where(x < self.x0, torch.zeros_like(x), quad)


def get_loss_function(loss_type: str, x0: float = 0.15, k: float = 1.0):
    loss_type = loss_type.lower()
    if loss_type == "mse":
        return MSELoss()
    if loss_type == "translated_relu":
        return TranslatedReLULoss(x0=x0, k=k)
    if loss_type == "smooth_k2":
        return SmoothK2Loss(x0=x0, k=k)
    raise ValueError(f"Unknown loss type: {loss_type}")
