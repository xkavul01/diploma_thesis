import torch
import torch.nn as nn

from ocr.model.laso_components.position_wise_ffn import PositionWiseFFN


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super(AttentionBlock, self).__init__()

        self._multihead_att = nn.MultiheadAttention(embed_dim=d_model, num_heads=8)
        self._dropout = nn.Dropout(p=0.25)
        self._layernorm = nn.LayerNorm(normalized_shape=d_model)
        self._position_wise_ffn = PositionWiseFFN(d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        x, _ = self._multihead_att(query, key, value)
        x = self._dropout(x)
        x = x + query
        x = self._layernorm(x)
        y = self._position_wise_ffn(x)
        y = self._dropout(y)
        y = y + x

        return y
