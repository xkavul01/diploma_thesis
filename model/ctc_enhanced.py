from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCEnhanced(nn.Module):
    def __init__(self, num_layers: int, cnn: str, alphabet: List[str]):
        super(CTCEnhanced, self).__init__()

        embedding_dim = 2048 if cnn == "resnet50" or cnn == "vgg16_bn" else 512
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8)
        layer_norm = nn.LayerNorm(embedding_dim)
        self._decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers, norm=layer_norm)
        self._embedding_layer = nn.Embedding(num_embeddings=len(alphabet), embedding_dim=embedding_dim)
        self._linear = nn.Linear(in_features=embedding_dim, out_features=len(alphabet))
        self._alphabet = alphabet

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: str) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(x.shape[0] + 1, x.shape[1]).long().to(next(self.parameters()).device)
        logits = F.softmax(y, dim=2)
        _, indices = torch.max(logits, dim=2)
        output[1:, :] = indices

        target_embedding = self._embedding_layer(output)
        mask = self.generate_square_subsequent_mask(x.shape[0] + 1, next(self.parameters()).device)
        x = self._decoder(target_embedding, x, tgt_mask=mask)
        x = self._linear(x)

        return x
