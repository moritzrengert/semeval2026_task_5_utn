"""Cross-Encoder model with mean-last-3 pooling.
Created by Adam Jen Khai Lo."""

import torch
from torch import nn
from transformers import AutoModel


class CrossEncoderRegressor(nn.Module):
    """Cross-Encoder: [text_a, text_b] → mean-last-3 pooling → MLP → score."""
    
    def __init__(self, base_model_name: str):
        super().__init__()
        # Load pre-trained transformer encoder (e.g., DeBERTa)
        self.encoder = AutoModel.from_pretrained(base_model_name)
        H = self.encoder.config.hidden_size
        # MLP head: hidden_size → 1024 → 256 → 1 (regression output)
        self.regressor = nn.Sequential(
            nn.Linear(H, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **_):
        # Encode [CLS] text_a [SEP] text_b [SEP] with cross-attention
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, 
                          token_type_ids=token_type_ids, output_hidden_states=True)
        # Average last 3 hidden layers for robust representation
        last_3 = torch.stack(out.hidden_states[-3:], dim=-1).mean(dim=-1)
        # Mean pooling over sequence (respecting attention mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last_3 * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        # Predict score via MLP regression head
        return {"logits": self.regressor(pooled).squeeze(-1)}
