from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLSTMSELF(nn.Module):
    def __init__(self, num_layers: int, dropout: float, alphabet: List[str]) -> None:
        super(CTCLSTMSELF, self).__init__()
        self._lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=num_layers, dropout=dropout)
        self._linear = nn.Linear(in_features=512, out_features=len(alphabet))
        self._embedding_layer = nn.Embedding(num_embeddings=len(alphabet), embedding_dim=256)
        self._multihead_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=dropout)
        self._layer_norm = nn.LayerNorm(256)
        self._alphabet = alphabet

    def forward(self, x: torch.Tensor, y: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        output = torch.full((x.shape[0], x.shape[1]), len(self._alphabet) - 1).long().to(next(self.parameters()).device)
        probs = F.softmax(y, dim=2)
        _, predictions = torch.max(probs, dim=2)

        ctc_predictions = torch.full((x.shape[0], x.shape[1]), len(self._alphabet) - 1).long()
        ctc_predictions[0, :] = 0
        ctc_predictions = ctc_predictions.to(next(self.parameters()).device)
        for i in range(y.shape[1]):
            unique_predictions = predictions[:, i].unique_consecutive()
            unique_predictions = unique_predictions[unique_predictions != len(self._alphabet) - 1]
            ctc_predictions[1:unique_predictions.shape[0] + 1, i] = unique_predictions

        if target is not None:
            for i in range(ctc_predictions.shape[1]):
                tmp = ctc_predictions[:, i]
                ctc_length = tmp[tmp != len(self._alphabet) - 1].shape[0]
                tmp = target[i]
                target_length = tmp[tmp != len(self._alphabet) - 1].shape[0]
                if abs(ctc_length - target_length) <= 2:
                    output[:, i] = ctc_predictions[:, i]
                else:
                    output[0, i] = 0
                    output[1:, i] = target[i, :-1]
        else:
            output = ctc_predictions

        x, _ = self._multihead_attn(x, x, x)
        x = self._layer_norm(x)

        output = self._embedding_layer(output)
        output = torch.concat((output, x), dim=-1)
        output, _ = self._lstm(output)
        output = self._linear(output)

        return output
