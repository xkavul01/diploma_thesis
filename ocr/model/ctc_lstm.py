from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLSTM(nn.Module):
    def __init__(self, alphabet: List[str]) -> None:
        super(CTCLSTM, self).__init__()
        self._lstm = nn.LSTM(input_size=256, hidden_size=256)
        self._linear = nn.Linear(in_features=256, out_features=len(alphabet))
        self._embedding_layer = nn.Embedding(num_embeddings=len(alphabet), embedding_dim=256)
        self._alphabet = alphabet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(x, dim=2)
        _, predictions = torch.max(probs, dim=2)

        ctc_predictions = torch.full((x.shape[0], x.shape[1]), len(self._alphabet) - 1).long()
        ctc_predictions = ctc_predictions.to(next(self.parameters()).device)
        for i in range(x.shape[1]):
            unique_predictions = predictions[:, i].unique_consecutive()
            unique_predictions = unique_predictions[unique_predictions != len(self._alphabet) - 1]
            ctc_predictions[:unique_predictions.shape[0], i] = unique_predictions

        output = self._embedding_layer(ctc_predictions)
        output, _ = self._lstm(output)
        output = self._linear(output)

        return output
