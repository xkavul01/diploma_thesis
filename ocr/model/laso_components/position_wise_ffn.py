import torch
import torch.nn as nn


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model: int) -> None:
        super(PositionWiseFFN, self).__init__()
        self._linear1 = nn.Linear(in_features=d_model, out_features=d_model * 8)
        self._linear2 = nn.Linear(in_features=d_model * 4, out_features=d_model)
        self._glu = nn.GLU()
        self._dropout = nn.Dropout(p=0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear1(x)
        x = self._glu(x)
        x = self._dropout(x)
        x = self._linear2(x)

        return x
