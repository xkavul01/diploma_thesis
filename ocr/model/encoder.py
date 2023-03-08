from typing import List, Tuple, Union

import torch
import torch.nn as nn

from ocr.model.vgg import vgg16_bn


cnns = {
    "vgg16_bn": vgg16_bn
}


class Encoder(nn.Module):
    def __init__(self,
                 cnn_model: str,
                 num_layers: int,
                 alphabet: List[str],
                 dropout: float,
                 decoder: str = None
                 ) -> None:
        super(Encoder, self).__init__()

        self._decoder = decoder
        self._dropout = nn.Dropout(p=dropout)
        self._tanh = nn.Tanh()
        if cnn_model == "vgg16_bn":
            self._cnn = cnns[cnn_model](weights="DEFAULT")
            self._max_pool = nn.MaxPool2d(kernel_size=(32, 1), stride=(32, 1))
            self._rnn = nn.LSTM(input_size=256,
                                hidden_size=128,
                                bidirectional=True,
                                num_layers=num_layers,
                                dropout=dropout)
            self._linear_scale = nn.Linear(in_features=256, out_features=256)
            self._linear_logit = nn.Linear(in_features=256, out_features=len(alphabet))
            self._ctc_shortcut = nn.Conv1d(in_channels=256, out_channels=len(alphabet), kernel_size=3, padding=1)

        else:
            raise ValueError("CNN is not available.")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._decoder is None:
            x = self._cnn(x)
            x = self._max_pool(x).squeeze(2)

            y = self._ctc_shortcut(x).permute(2, 0, 1)

            x = x.permute(2, 0, 1)
            x, _ = self._rnn(x)
            x = self._linear_scale(x)
            x = self._tanh(x)
            x = self._dropout(x)
            x = self._linear_logit(x)

            return x, y
        else:
            x = self._cnn(x)
            x = self._max_pool(x).squeeze(2).permute(2, 0, 1)
            x, _ = self._rnn(x)
            y = self._linear_scale(x)
            y = self._tanh(y)
            y = self._dropout(y)
            y = self._linear_logit(y)

            return x, y
