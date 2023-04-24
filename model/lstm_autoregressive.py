from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# Bahdanau + location attention
class LocationAttention(nn.Module):
    def __init__(self, hidden_size, decoder_layer):
        super(LocationAttention, self).__init__()
        k = 256  # the filters of location attention
        r = 7  # window size of the kernel
        self.hidden_size = hidden_size
        self.decoder_layer = decoder_layer
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.hidden_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.encoder_output_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.conv1d = nn.Conv1d(1, k, r, padding=3)
        self.prev_attn_proj = nn.Linear(k, self.hidden_size)
        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        self.sigma = self.softmax

    # hidden:         layers, b, f
    # encoder_output: t, b, f
    # prev_attention: b, t
    def forward(self, hidden, encoder_output, enc_len, prev_attention):
        encoder_output = encoder_output.transpose(0, 1)  # b, t, f
        attn_energy = self.score(hidden, encoder_output, prev_attention)

        attn_weight = torch.zeros(attn_energy.shape).to(next(self.parameters()))
        for i, le in enumerate(enc_len):
            attn_weight[i, :le] = self.sigma(attn_energy[i, :le])
        return attn_weight.unsqueeze(2)

    # encoder_output: b, t, f
    def score(self, hidden, encoder_output, prev_attention):
        hidden = hidden.permute(1, 2, 0)  # b, f, layers
        add_mask = torch.FloatTensor([1/self.decoder_layer] * self.decoder_layer).view(1, self.decoder_layer, 1)
        add_mask = torch.cat([add_mask] * hidden.shape[0], dim=0)
        add_mask = add_mask.to(next(self.parameters()))  # b, layers, 1
        hidden = torch.bmm(hidden, add_mask)  # b, f, 1
        hidden = hidden.permute(0, 2, 1)  # b, 1, f
        hidden_attn = self.hidden_proj(hidden)  # b, 1, f

        prev_attention = prev_attention.unsqueeze(1)  # b, 1, t
        conv_prev_attn = self.conv1d(prev_attention)  # b, k, t
        conv_prev_attn = conv_prev_attn.permute(0, 2, 1)  # b, t, k
        conv_prev_attn = self.prev_attn_proj(conv_prev_attn)  # b, t, f

        encoder_output_attn = self.encoder_output_proj(encoder_output)
        res_attn = self.tanh(encoder_output_attn + hidden_attn + conv_prev_attn)
        out_attn = self.out(res_attn)  # b, t, 1
        out_attn = out_attn.squeeze(2)  # b, t
        return out_attn


# Standard Bahdanau Attention
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, decoder_layer):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.decoder_layer = decoder_layer
        self.softmax = nn.Softmax(dim=0)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.hidden_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.encoder_output_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    # hidden: b, f  encoder_output: t, b, f  enc_len: numpy
    def forward(self, hidden, encoder_output, enc_len, prev_attention):
        encoder_output = encoder_output.transpose(0, 1)  # b, t, f
        attn_energy = self.score(hidden, encoder_output)  # b, t

        attn_weight = torch.zeros(attn_energy.shape).to(next(self.parameters()).device)
        for i, le in enumerate(enc_len):
            attn_weight[i, :le] = self.softmax(attn_energy[i, :le])
        return attn_weight.unsqueeze(2)

    # hidden: 1, batch, features
    # encoder_output: batch, time_step, features
    def score(self, hidden, encoder_output):
        hidden = hidden.permute(1, 2, 0) # batch, features, layers
        add_mask = torch.FloatTensor([1/self.decoder_layer] * self.decoder_layer).view(1, self.decoder_layer, 1)
        add_mask = torch.cat([add_mask] * hidden.shape[0], dim=0)
        add_mask = add_mask.to(next(self.parameters()).device)  # batch, layers, 1
        hidden = torch.bmm(hidden, add_mask)  # batch, feature, 1
        hidden = hidden.permute(0, 2, 1)  # batch, 1, features
        hidden_attn = self.hidden_proj(hidden)  # b, 1, f
        encoder_output_attn = self.encoder_output_proj(encoder_output)
        res_attn = self.tanh(encoder_output_attn + hidden_attn)  # b, t, f
        out_attn = self.out(res_attn)  # b, t, 1
        out_attn = out_attn.squeeze(2)  # b, t
        return out_attn


class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, n_layers, alphabet_size, attention, alphabet, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embedding_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(alphabet_size, self.embed_size)
        self.dropout = dropout
        self.attention = attention(self.hidden_size, self.n_layers)

        self.lstm = nn.LSTM(self.embed_size + self.hidden_size, self.hidden_size, self.n_layers, dropout=self.dropout)
        self.context_proj = nn.Linear(self.hidden_size * (self.n_layers + 1), self.hidden_size)
        self.out = nn.Linear(self.hidden_size, alphabet_size)
        self.alphabet = alphabet

    def forward(self, probs, hidden_state, cell_state, encoder_output, enc_len, prev_attention):
        attn_weights = self.attention(hidden_state, encoder_output, enc_len, prev_attention)

        encoder_output_b = encoder_output.permute(1, 2, 0)
        context = torch.bmm(encoder_output_b, attn_weights)
        tmp = hidden_state.permute(1, 2, 0)
        tmp = tmp.reshape(tmp.shape[0], -1, 1)
        tmp = torch.cat((context, tmp), 1).squeeze(2)
        context = self.context_proj(tmp)
        context = torch.tanh(context)

        _, char = torch.max(probs, dim=-1)
        embed_char = self.embedding(char.to(next(self.parameters()).device))

        in_dec = torch.cat((embed_char, context), 1)
        in_dec = in_dec.unsqueeze(0)
        output, (latest_hidden, latest_cell) = self.lstm(in_dec, (hidden_state, cell_state))
        output = output.squeeze(0)
        output = self.out(output)
        probs = F.softmax(output, dim=-1)

        return output, probs, latest_hidden, latest_cell, attn_weights.squeeze(2)


class LSTMAutoregressive(nn.Module):
    def __init__(self, alphabet: List[str]) -> None:
        super(LSTMAutoregressive, self).__init__()
        self.hidden_size = 256
        self.embedding_size = 256
        self.n_layers = 3
        self.dropout = 0.25
        self.decoder = Decoder(self.hidden_size,
                               self.embedding_size,
                               self.n_layers,
                               len(alphabet),
                               LocationAttention,
                               alphabet,
                               self.dropout)
        self.alphabet = alphabet

    def forward(self, x: torch.Tensor, ground_truths: torch.Tensor = None) -> torch.Tensor:
        # ground_truths: [batch_size, seq_len]
        outputs = torch.zeros(x.shape[0], x.shape[1], len(self.alphabet))
        outputs = outputs.to(next(self.parameters()).device)

        hidden_state = torch.zeros(self.n_layers, x.shape[1], self.hidden_size).to(next(self.parameters()).device)
        cell_state = torch.zeros(self.n_layers, x.shape[1], self.hidden_size).to(next(self.parameters()).device)
        attn_weights = torch.zeros(x.shape[1], x.shape[0]).to(next(self.parameters()))

        probs = torch.zeros(x.shape[1], len(self.alphabet))
        probs[:, 0] = 1.0  # <SOS> token
        for t in range(1, x.shape[0]):
            enc_len = [t for _ in range(x.shape[1])]
            output, probs, hidden_state, cell_state, attn_weights = self.decoder(probs,
                                                                                 hidden_state,
                                                                                 cell_state,
                                                                                 x,
                                                                                 enc_len,
                                                                                 attn_weights)
            if self.training:
                probs[:, :] = 0.0
                for i in range(x.shape[1]):
                    probs[i, ground_truths[i, t - 1]] = 1.0

            outputs[t - 1] = output.unsqueeze(0)

        return outputs
