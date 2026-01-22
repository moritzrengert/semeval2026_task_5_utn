from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .losses import coral_expected_value, coral_loss
from .modules import EnsembleModelConfig, EnsembleRegressor


@dataclass
class LossConfig:
    """Different params for the loss"""

    regression_weight: float = 1.0
    coral_weight: float = 0.0
    cross_entropy_weight: float = 0.0
    num_classes: int = 5


class FinalEnsembleModel(nn.Module):
    """Combine all components into the final model with loss computation"""

    def __init__(
        self,
        model_config: EnsembleModelConfig,
        loss_config: Optional[LossConfig] = None,
    ):
        super().__init__()
        self.model = EnsembleRegressor(model_config)
        self.loss_config = loss_config or LossConfig()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        dense_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dense_features=dense_features,
        )

    def compute_loss(
        self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        total: Optional[torch.Tensor] = None
        if self.loss_config.regression_weight > 0:
            reg_loss = F.mse_loss(outputs["regression"], batch["labels"])
            losses["regression"] = reg_loss
            total = (
                reg_loss * self.loss_config.regression_weight
                if total is None
                else total + reg_loss * self.loss_config.regression_weight
            )
        if self.loss_config.coral_weight > 0:
            ord_loss = coral_loss(
                outputs["ordinal_logits"], batch["ordinal_labels"], num_classes=self.loss_config.num_classes
            )
            losses["coral"] = ord_loss
            total = (
                ord_loss * self.loss_config.coral_weight
                if total is None
                else total + ord_loss * self.loss_config.coral_weight
            )
        if (
            self.loss_config.cross_entropy_weight > 0
            and outputs.get("class_logits") is not None
        ):
            ce_loss = F.cross_entropy(
                outputs["class_logits"], batch["ordinal_labels"]
            )
            losses["cross_entropy"] = ce_loss
            total = (
                ce_loss * self.loss_config.cross_entropy_weight
                if total is None
                else total + ce_loss * self.loss_config.cross_entropy_weight
            )
        if total is None:
            raise ValueError(
                "No loss selected; set a non-zero weight in LossConfig."
            )
        losses["total"] = total
        return losses

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        dense_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dense_features=dense_features,
        )
        regression = outputs["regression"]
        ordinal_score = coral_expected_value(outputs["ordinal_logits"])
        class_logits = outputs.get("class_logits")
        class_pred = (
            class_logits.argmax(dim=-1) if class_logits is not None else None
        )
        return {
            "regression": regression,
            "ordinal_score": ordinal_score,
            "class_pred": class_pred,
        }
