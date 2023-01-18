from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MPCTC(nn.Module):
    def __init__(self, num_layers: int, cnn: str, threshold: float, alphabet: List[str]):
        super(MPCTC, self).__init__()

        embedding_dim = 2048 if cnn == "resnet50" else 512
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=1)
        layer_norm = nn.LayerNorm(embedding_dim)
        self._decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers, norm=layer_norm)
        self._embedding_layer = nn.Embedding(num_embeddings=len(alphabet), embedding_dim=embedding_dim)
        self._linear_ctc = nn.Linear(in_features=embedding_dim, out_features=len(alphabet))
        self._linear_decoder = nn.Linear(in_features=embedding_dim, out_features=len(alphabet))
        self._alphabet = alphabet
        self._threshold = threshold

    @staticmethod
    def _create_mask(sequence_length: int, batch_size: int, output: torch.Tensor, device: str) -> torch.Tensor:
        output = output.permute(1, 0)
        indices = (output != 0).nonzero()
        mask = torch.full(size=(batch_size, sequence_length, sequence_length), fill_value=float("-inf")).to(device)
        mask[indices[:, 0], :, indices[:, 1]] = 0.0

        for i in range(mask.shape[0]):
            mask[i, range(mask[i].shape[0]), range(mask[i].shape[1])] = 0.0

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._linear_ctc(x)
        output_softmax = F.softmax(output, dim=2)
        values, indices = torch.max(output_softmax, dim=2)
        masked_output = torch.where(values > self._threshold, indices, 0)

        mask = self._create_mask(x.shape[0], x.shape[1], masked_output, next(self.parameters()).device)
        target_embedding = self._embedding_layer(masked_output)
        decoded_output = self._decoder(target_embedding, x, tgt_mask=mask)
        decoded_output = self._linear_decoder(decoded_output)
        high_confidence_indices = (masked_output != 0).nonzero()
        decoded_output[high_confidence_indices[:, 0],
                       high_confidence_indices[:, 1]] = output[high_confidence_indices[:, 0],
                                                               high_confidence_indices[:, 1]]

        return decoded_output
