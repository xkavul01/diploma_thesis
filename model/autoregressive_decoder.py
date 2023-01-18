from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.positional_encoding import PositionalEncoding


class AutoregressiveDecoder(nn.Module):
    def __init__(self, num_layers: int, cnn: str, alphabet: List[str]) -> None:
        super(AutoregressiveDecoder, self).__init__()

        embedding_dim = 2048 if cnn == "resnet50" else 512
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8)
        layer_norm = nn.LayerNorm(embedding_dim)
        self._decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers, norm=layer_norm)
        self._embedding_layer = nn.Embedding(num_embeddings=len(alphabet), embedding_dim=embedding_dim)
        self._positional_encoder = PositionalEncoding(d_model=embedding_dim)
        self._linear = nn.Linear(in_features=embedding_dim, out_features=len(alphabet))
        self._alphabet = alphabet

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: str) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, x: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        if target is None:
            output = torch.zeros(x.shape[0] + 1, x.shape[1]).long().to(next(self.parameters()).device)
            result_output = torch.zeros(x.shape[0], x.shape[1], len(self._alphabet)).to(next(self.parameters()).device)

            for t in range(1, output.shape[0]):
                target_embedding = self._embedding_layer(output[:t, :])
                positional_encoding = self._positional_encoder(target_embedding)
                target_embedding = target_embedding + positional_encoding

                mask = self.generate_square_subsequent_mask(t, next(self.parameters()).device)
                decoder_output = self._decoder(target_embedding, x, tgt_mask=mask)
                decoder_output = self._linear(decoder_output)
                result_output[t - 1, :, :] = decoder_output[t - 1, :, :]

                score_probs = F.softmax(decoder_output, dim=2)
                _, indices = torch.max(score_probs[t - 1, :, :], dim=1)
                output[t, :] = indices.long()

            return result_output
        else:
            target_embedding = self._embedding_layer(target)
            positional_encoding = self._positional_encoder(target_embedding)
            target_embedding = target_embedding + positional_encoding

            mask = self.generate_square_subsequent_mask(target_embedding.shape[0], next(self.parameters()).device)
            x = self._decoder(target_embedding, x, tgt_mask=mask)
            x = self._linear(x)

            return x
