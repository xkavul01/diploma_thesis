from typing import List
from itertools import groupby

import torch
import torch.nn as nn
import torch.nn.functional as F


class MPCTC(nn.Module):
    def __init__(self, num_layers: int, cnn: str, threshold: float, alphabet: List[str]):
        super(MPCTC, self).__init__()

        embedding_dim = 2048 if cnn == "resnet50" or cnn == "vgg16_bn" else 512
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8, dropout=0.25)
        layer_norm = nn.LayerNorm(embedding_dim)
        self._decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers, norm=layer_norm)
        self._embedding_layer = nn.Embedding(num_embeddings=len(alphabet), embedding_dim=embedding_dim)
        self._linear = nn.Linear(in_features=embedding_dim, out_features=len(alphabet))
        self._alphabet = alphabet
        self._threshold = threshold

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ctc_probs, ctc_ids = torch.exp(F.log_softmax(y, dim=-1)).max(dim=-1)
        predictions = [torch.stack([idx[0] for idx in groupby(ctc_ids[:, i])]) for i in range(y.shape[1])]
        predictions_idxs = [torch.nonzero(predictions[i] != len(self._alphabet) - 1).squeeze(-1)
                            for i in range(y.shape[1])]
        predictions_original_idxs = [torch.Tensor([j for j in range(y.shape[0])
                                                   if j == 0 or ctc_ids[j, i] != ctc_ids[j - 1, i]]).long()
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

        mask_idxs = [torch.nonzero(all_probs[i] < self._threshold).squeeze(-1) for i in range(y.shape[1])]
        confident_idxs = [torch.nonzero(all_probs[i] >= self._threshold).squeeze(-1) for i in range(y.shape[1])]

        output = torch.full((y.shape[0], y.shape[1]),
                            fill_value=len(self._alphabet) - 1).long().to(next(self.parameters()).device)
        for i in range(y.shape[1]):
            for j in range(predictions_idxs[i].shape[0]):
                output[j, i] = predictions[i][int(predictions_idxs[i][j])]
            output[mask_idxs[i], i] = len(self._alphabet) - 2

        output = self._embedding_layer(output)
        output = self._decoder(output, x)
        output = self._linear(output)

        for i in range(len(confident_idxs)):
            output[confident_idxs[i], i, :] = y[predictions_original_idxs[i][confident_idxs[i]], i, :]

        return output
