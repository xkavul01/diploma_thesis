import torch
import torch.nn as nn

from ocr.model.laso_components.attention_block import AttentionBlock


class LASODecoder(nn.Module):
    def __init__(self, d_model: int, n: int) -> None:
        super(LASODecoder, self).__init__()
        self._attention_block = AttentionBlock(d_model)
        self._layer_norm = nn.LayerNorm(d_model)
        self._n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self._n):
            x = self._layer_norm(x)
            x = self._attention_block(x, x, x)

        return x
