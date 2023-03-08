from typing import Union, Tuple

import torch
import torch.nn as nn

from sen.model.conv_block import ConvBlock


class StyleExtractorNetwork(nn.Module):
    def __init__(self, num_classes: int, batch_size: int, is_ocr: bool) -> None:
        super(StyleExtractorNetwork, self).__init__()
        self._conv_block_1 = ConvBlock(3, 16, 3, batch_norm=True)
        self._conv_block_2 = ConvBlock(16, 32, 3, batch_norm=True)
        self._conv_block_3 = ConvBlock(32, 64, 3, batch_norm=True)
        self._conv_block_4 = ConvBlock(64, 128, 3, batch_norm=True)
        self._max_pooling = nn.MaxPool2d(2, stride=2)
        self._dropout = nn.Dropout2d(p=0.25)

        self._rnn = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True)
        self._linear = nn.Linear(512, num_classes)

        self._batch_size = batch_size
        self._is_ocr = is_ocr

    def forward(self, x: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        x = self._conv_block_1(x)
        x = self._dropout(x)
        x = self._max_pooling(x)
        x = self._conv_block_2(x)
        x = self._dropout(x)
        x = self._max_pooling(x)
        x = self._conv_block_3(x)
        x = self._dropout(x)
        x = self._max_pooling(x)
        x = self._conv_block_4(x)
        x = self._dropout(x)

        x = x.permute(3, 0, 1, 2)
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x, _ = self._rnn(x)
        x = x.permute(1, 0, 2)

        if self._is_ocr:
            local_style_features = x.view(self._batch_size, -1)
            global_style_features = torch.mean(x, dim=1)

            return local_style_features, global_style_features
        else:
            x = torch.mean(x, dim=1)
            x = self._linear(x)

            return x
