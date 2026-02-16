"""Custom Trainer with regression loss (same as cross_encoder_adam).
Created by Adam Jen Khai Lo."""

import torch
from transformers import Trainer
from .losses import get_loss_function


class RegressionLossTrainer(Trainer):
    def __init__(self, loss_type="mse", loss_x0=0.15, loss_k=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = get_loss_function(loss_type.lower(), x0=loss_x0, k=loss_k)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = (inputs.pop("labels") if "labels" in inputs else inputs.pop("label")).float()
        inputs.pop("stdev", None)
        out = model(**inputs)
        per_ex = self.loss_fn(out["logits"], labels)
        return (per_ex.mean(), out) if return_outputs else per_ex.mean()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.pop("labels") if "labels" in inputs else inputs.pop("label")
        inputs.pop("stdev", None)
        with torch.no_grad():
            out = model(**inputs)
            preds = out["logits"]
            loss = torch.nn.functional.mse_loss(preds, labels.float())
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, preds, labels)
