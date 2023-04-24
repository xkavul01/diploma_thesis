import torch
import torch.nn as nn

from ocr.model.laso_components.attention_block import AttentionBlock
from ocr.model.positional_encoding import PositionalEncoding


class PositionDependentSummarizer(nn.Module):
    def __init__(self, d_model: int, n: int) -> None:
        super(PositionDependentSummarizer, self).__init__()
        self._attention_block_1 = AttentionBlock(d_model)
        self._layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self._attention_block_2 = AttentionBlock(d_model)
        self._positional_encoder = PositionalEncoding(d_model, 0.25)
        self._n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positional_encoding = self._positional_encoder(x)
        result = self._attention_block_1(positional_encoding, x, x)

        for i in range(self._n - 1):
            result = self._layer_norm(result)
            result = self._attention_block_2(result, x, x)

        return result
