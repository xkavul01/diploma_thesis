from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ocr.model.positional_encoding import PositionalEncoding


class CTCEnhanced(nn.Module):
    def __init__(self, num_layers: int, dropout: float, alphabet: List[str]):
        super(CTCEnhanced, self).__init__()

        decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=8, dropout=dropout, dim_feedforward=1024)
        layer_norm = nn.LayerNorm(256)
        self._decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=6, norm=layer_norm)
        self._embedding_layer = nn.Embedding(num_embeddings=len(alphabet), embedding_dim=256)
        self._linear = nn.Linear(in_features=256, out_features=len(alphabet))
        self._positional_encoder = PositionalEncoding(d_model=256, dropout=dropout)
        self._alphabet = alphabet

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: str) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

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

        target_embedding = self._embedding_layer(output)
        target_embedding = self._positional_encoder(target_embedding)
        mask = self.generate_square_subsequent_mask(x.shape[0], next(self.parameters()).device)
        x = self._decoder(target_embedding, x, tgt_mask=mask)
        x = self._linear(x)

        return x
