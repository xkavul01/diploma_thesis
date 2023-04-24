from typing import List, Tuple
from itertools import groupby

import torch
import torch.nn as nn
import torch.nn.functional as F


class MPCTCLSTMSELF(nn.Module):
    def __init__(self, num_layers: int, threshold: float, dropout: float, alphabet: List[str]):
        super(MPCTCLSTMSELF, self).__init__()

        self._lstm = nn.LSTM(input_size=512,
                             hidden_size=256,
                             num_layers=num_layers,
                             dropout=dropout,
                             bidirectional=True)
        self._embedding_layer = nn.Embedding(num_embeddings=len(alphabet), embedding_dim=256)
        self._linear = nn.Linear(in_features=512, out_features=len(alphabet))
        self._multihead_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=dropout)
        self._alphabet = alphabet
        self._threshold = threshold

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        ctc_probs, ctc_ids = F.softmax(y, dim=-1).max(dim=-1)
        predictions = [torch.stack([idx[0] for idx in groupby(ctc_ids[:, i])]) for i in range(y.shape[1])]
        predictions_idxs = [torch.nonzero(predictions[i] != len(self._alphabet) - 1).squeeze(-1)
                            for i in range(y.shape[1])]

        all_probs = []
        for i in range(y.shape[1]):
            probs = []
            count = 0
            for j, prediction in enumerate(predictions[i]):
                probs.append(-1)
                while count < ctc_ids.shape[0] and prediction == ctc_ids[count, i]:
                    if probs[j] < ctc_probs[count, i]:
                        probs[j] = ctc_probs[count, i]
                    count += 1
            probs = torch.Tensor(probs)
            all_probs.append(probs)

        mask_indexes_final = []
        mask_idxs = [torch.nonzero(all_probs[i] < self._threshold).squeeze(-1) for i in range(y.shape[1])]
        output = torch.full((y.shape[0], y.shape[1]),
                            fill_value=len(self._alphabet) - 1).long().to(next(self.parameters()).device)
        for i in range(y.shape[1]):
            mask_indexes_final.append([])
            for j in range(predictions_idxs[i].shape[0]):
                output[j, i] = predictions[i][int(predictions_idxs[i][j])]
                if (int(predictions_idxs[i][j]) == mask_idxs[i]).sum():
                    output[j, i] = len(self._alphabet) - 2
                    mask_indexes_final[-1].append(j)
            mask_indexes_final[-1] = torch.Tensor(mask_indexes_final[-1])

        x, _ = self._multihead_attn(x, x, x)

        output = self._embedding_layer(output)
        output = torch.concat((output, x), dim=-1)
        output, _ = self._lstm(output)
        output = self._linear(output)

        return output, mask_indexes_final
