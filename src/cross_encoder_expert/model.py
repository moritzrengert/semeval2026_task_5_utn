"""Cross-Encoder model (same as cross_encoder_adam)."""

import torch
from torch import nn
from transformers import AutoModel


def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    sum_h = (hidden_states * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)
    return sum_h / count


class CrossEncoderRegressor(nn.Module):
    def __init__(self, base_model_name: str, frozen_dim: int = 0, use_dual_path: bool = False, pooling: str = "cls"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.frozen_dim = frozen_dim
        self.use_dual_path = use_dual_path
        self.pooling = pooling.lower()
        if self.pooling not in ("cls", "mean", "mean_last_3", "attention"):
            raise ValueError(f"pooling must be one of cls, mean, mean_last_3, attention; got {pooling!r}")
        H = self.encoder.config.hidden_size
        if self.pooling == "attention":
            self.attn_query = nn.Linear(H, 1)
        if use_dual_path and frozen_dim > 0:
            self.mlp_cross = nn.Sequential(nn.Linear(H, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 128), nn.ReLU())
            self.mlp_frozen = nn.Sequential(
                nn.Linear(frozen_dim * 4, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 128), nn.ReLU()
            )
            self.final_layer = nn.Sequential(nn.Dropout(0.1), nn.Linear(256, 1))
        else:
            in_dim = H + (frozen_dim * 4 if frozen_dim > 0 else 0)
            self.regressor = nn.Sequential(
                nn.Linear(in_dim, 1024), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1),
            )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, frozen_a=None, frozen_b=None, **_):
        need_hidden_states = self.pooling == "mean_last_3"
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
            output_hidden_states=need_hidden_states,
        )
        hidden = out.last_hidden_state
        mask = attention_mask
        if self.pooling == "cls":
            pooled = hidden[:, 0, :]
        elif self.pooling == "mean":
            pooled = _mean_pool(hidden, mask)
        elif self.pooling == "mean_last_3":
            last_3 = torch.stack(out.hidden_states[-3:], dim=-1).mean(dim=-1)
            pooled = _mean_pool(last_3, mask)
        else:
            scores = self.attn_query(hidden).squeeze(-1)
            scores = scores.masked_fill(mask.eq(0), -1e9)
            attn = torch.softmax(scores, dim=1).unsqueeze(1)
            pooled = torch.bmm(attn, hidden).squeeze(1)
        if self.use_dual_path and self.frozen_dim > 0 and frozen_a is not None:
            frozen_feats = torch.cat([frozen_a, frozen_b, torch.abs(frozen_a - frozen_b), frozen_a * frozen_b], dim=-1)
            cross_out = self.mlp_cross(pooled)
            frozen_out = self.mlp_frozen(frozen_feats)
            pred = self.final_layer(torch.cat([cross_out, frozen_out], dim=-1)).squeeze(-1)
        else:
            if self.frozen_dim > 0 and frozen_a is not None:
                frozen_feats = torch.cat([frozen_a, frozen_b, torch.abs(frozen_a - frozen_b), frozen_a * frozen_b], dim=-1)
                pooled = torch.cat([pooled, frozen_feats], dim=-1)
            pred = self.regressor(pooled).squeeze(-1)
        return {"logits": pred}
