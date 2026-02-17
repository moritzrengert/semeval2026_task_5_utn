"""Regression losses for cross-encoder training.
Created by Adam Jen Khai Lo."""

import torch
from torch import nn


class SmoothK2Loss(nn.Module):
    def __init__(self, x0: float = 0.15, k: float = 1.0):
        super().__init__()
        self.x0 = float(x0)
        self.k = float(k)

    def forward(self, pred, target):
        x = (pred - target).abs()
        quad = self.k * (x * x - 2 * self.x0 * x + self.x0 * self.x0)
        loss = torch.where(x < self.x0, torch.zeros_like(x), quad)
        return loss.mean()
