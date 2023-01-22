import torch
import torch.nn as nn

from model.autoregressive_decoder import AutoregressiveDecoder
from model.mp_ctc import MPCTC
from model.ctc_enhanced import CTCEnhanced


class OCRModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module = None) -> None:
        super(OCRModel, self).__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, x: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        if self._decoder is None:
            _, y = self._encoder(x)
            return y
        elif isinstance(self._decoder, AutoregressiveDecoder):
            x, y = self._encoder(x)
            y = self._decoder(x, target)
            return y
        elif isinstance(self._decoder, MPCTC) or isinstance(self._decoder, CTCEnhanced):
            x, y = self._encoder(x)
            y = self._decoder(x, y)
            return y
