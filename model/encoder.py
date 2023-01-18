import torch
import torch.nn as nn

from model.resnet import resnet50, resnet18, resnet34


class Encoder(nn.Module):
    def __init__(self, cnn_model: str, num_layers: int):
        super(Encoder, self).__init__()

        if cnn_model == "resnet50":
            self._cnn = resnet50(pretrained=True)
            self._rnn = nn.LSTM(input_size=2048,
                                hidden_size=1024,
                                bidirectional=True,
                                num_layers=num_layers)
            self._linear = nn.Linear(in_features=2048, out_features=2048)

        elif cnn_model == "resnet18":
            self._cnn = resnet18(pretrained=True)
            self._rnn = nn.LSTM(input_size=512,
                                hidden_size=256,
                                bidirectional=True,
                                num_layers=num_layers)
            self._linear = nn.Linear(in_features=512, out_features=512)

        elif cnn_model == "resnet34":
            self._cnn = resnet34(pretrained=True)
            self._rnn = nn.LSTM(input_size=512,
                                hidden_size=256,
                                bidirectional=True,
                                num_layers=num_layers)
            self._linear = nn.Linear(in_features=512, out_features=512)

        else:
            raise ValueError("CNN is not available.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._cnn(x)
        x = x.permute(3, 0, 1, 2)
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x, _ = self._rnn(x)
        x = self._linear(x)

        return x
