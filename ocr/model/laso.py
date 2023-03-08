from typing import List

import torch
import torch.nn as nn

from ocr.model.laso_components.position_dependent_summarizer import PositionDependentSummarizer
from ocr.model.laso_components.laso_decoder import LASODecoder


class LASO(nn.Module):
    def __init__(self, cnn: str, alphabet: List[str]) -> None:
        super(LASO, self).__init__()
        embedding_dim = 2048 if cnn == "resnet50" or cnn == "vgg16_bn" else 512
        self._pds = PositionDependentSummarizer(embedding_dim, 4)
        self._decoder = LASODecoder(embedding_dim, 6)
        self._linear = nn.Linear(in_features=embedding_dim, out_features=len(alphabet))
        self._layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._layer_norm(x)
        x = self._pds(x)
        x = self._decoder(x)
        x = self._linear(x)

        return x
