from typing import Union, Tuple

import torch
import torch.nn as nn

from ocr.model.autoregressive_decoder import AutoregressiveDecoder
from ocr.model.mp_ctc import MPCTC
from ocr.model.ctc_enhanced import CTCEnhanced
from ocr.model.laso import LASO
from ocr.model.ctc_lstm import CTCLSTM
from ocr.model.lstm_autoregressive import LSTMAutoregressive


class OCRModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module = None) -> None:
        super(OCRModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
                x: torch.Tensor,
                target: torch.Tensor = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, y = self.encoder(x)

        if self.decoder is None:
            return x, y

        elif isinstance(self.decoder, LSTMAutoregressive):
            return self.decoder(x, target), y

        elif isinstance(self.decoder, CTCLSTM):
            return self.decoder(y)

        elif isinstance(self.decoder, LASO):
            return self.decoder(x)

        elif isinstance(self.decoder, AutoregressiveDecoder):
            return self.decoder(x, target)

        elif isinstance(self.decoder, CTCEnhanced):
            if target is None:
                y = self.decoder(x, y)
            else:
                y = self.decoder(x, y, target)

            return y

        elif isinstance(self.decoder, MPCTC):
            return self.decoder(x, y)
