import torch
import torch.nn as nn


class OCRModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, decoder_str: str) -> None:
        super(OCRModel, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._decoder_str = decoder_str

    def forward(self, x: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        x = self._encoder(x)

        if self._decoder_str == "base" or self._decoder_str == "mp_ctc" or self._decoder_str == "ctc_enhanced":
            x = self._decoder(x)
        elif self._decoder_str == "autoregressive":
            x = self._decoder(x, target)

        return x
