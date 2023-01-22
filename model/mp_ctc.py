from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MPCTC(nn.Module):
    def __init__(self, num_layers: int, cnn: str, threshold: float, alphabet: List[str]):
        super(MPCTC, self).__init__()

        embedding_dim = 2048 if cnn == "resnet50" or cnn == "vgg16_bn" else 512
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8)
        layer_norm = nn.LayerNorm(embedding_dim)
        self._decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers, norm=layer_norm)
        self._embedding_layer = nn.Embedding(num_embeddings=len(alphabet), embedding_dim=embedding_dim)
        self._linear = nn.Linear(in_features=embedding_dim, out_features=len(alphabet))
        self._alphabet = alphabet
        self._threshold = threshold

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(y, dim=2)
        max_probs, indices = torch.max(probs, dim=2)
        masked_output = torch.where(max_probs > self._threshold, indices, len(self._alphabet) - 2)

        target_embedding = self._embedding_layer(masked_output)
        decoded_output = self._decoder(target_embedding, x)
        decoded_output = self._linear(decoded_output)
        high_confidence_indices = (masked_output != (len(self._alphabet) - 2)).nonzero()
        decoded_output[high_confidence_indices[:, 0],
                       high_confidence_indices[:, 1]] = y[high_confidence_indices[:, 0],
                                                          high_confidence_indices[:, 1]]

        return decoded_output
