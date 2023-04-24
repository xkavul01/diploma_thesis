from typing import Union, Tuple, List

import torch
import torch.nn as nn

from ocr.model.mp_ctc_lstm import MPCTCLSTM
from ocr.model.mp_ctc_lstm_self import MPCTCLSTMSELF
from ocr.model.mp_ctc_lstm_cross import MPCTCLSTMCROSS
from ocr.model.mp_ctc import MPCTC
from ocr.model.ctc_lstm import CTCLSTM
from ocr.model.ctc_lstm_self import CTCLSTMSELF
from ocr.model.ctc_lstm_cross import CTCLSTMCROSS
from ocr.model.ctc_enhanced import CTCEnhanced


class OCRModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module = None) -> None:
        super(OCRModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
                x: torch.Tensor,
                target: torch.Tensor = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
        x, y = self.encoder(x)

        if self.decoder is None:
            return x, y

        elif isinstance(self.decoder, CTCLSTM) or \
                isinstance(self.decoder, CTCLSTMSELF) or \
                isinstance(self.decoder, CTCLSTMCROSS) or \
                isinstance(self.decoder, CTCEnhanced):
            return self.decoder(x, y, target), y

        elif isinstance(self.decoder, MPCTCLSTM) or \
                isinstance(self.decoder, MPCTCLSTMSELF) or \
                isinstance(self.decoder, MPCTCLSTMCROSS) or \
                isinstance(self.decoder, MPCTC):
            x, mask_idxs = self.decoder(x, y)
            return x, y, mask_idxs
