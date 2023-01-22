from typing import List, Tuple

import torch
import torch.nn as nn

from model.resnet import resnet50, resnet18, resnet34
from model.vgg import vgg16_bn


cnns = {
    "vgg16_bn": vgg16_bn,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}


class Encoder(nn.Module):
    def __init__(self, cnn_model: str, num_layers: int, alphabet: List[str]) -> None:
        super(Encoder, self).__init__()

        if cnn_model == "resnet50" or cnn_model == "vgg16_bn":
            self._cnn = cnns[cnn_model](weights="DEFAULT")
            self._rnn = nn.LSTM(input_size=2048,
                                hidden_size=1024,
                                bidirectional=True,
                                num_layers=num_layers)
            self._linear_scale = nn.Linear(in_features=2048, out_features=2048)
            self._linear_logit = nn.Linear(in_features=2048, out_features=len(alphabet))

        elif cnn_model == "resnet18" or cnn_model == "resnet34":
            self._cnn = cnns[cnn_model](weights="DEFAULT")
            self._rnn = nn.LSTM(input_size=512,
                                hidden_size=256,
                                bidirectional=True,
                                num_layers=num_layers)
            self._linear_scale = nn.Linear(in_features=512, out_features=512)
            self._linear_logit = nn.Linear(in_features=512, out_features=len(alphabet))

        else:
            raise ValueError("CNN is not available.")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._cnn(x)
        x = x.permute(3, 0, 1, 2)
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x, _ = self._rnn(x)
        x = self._linear_scale(x)
        y = self._linear_logit(x)

        return x, y
