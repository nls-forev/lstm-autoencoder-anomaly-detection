import torch
import torch.nn as nn
from torch import Tensor


class LSTMEncoder(nn.Module):
    def __init__(self, features: int, hidden: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=features, hidden_size=hidden, num_layers=2, batch_first=True
        )

    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x)
        return h_n[-1], c_n[-1]


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        h0: Tensor,
        c0: Tensor,
        T: int,
        features: int,
        hidden: int,
        num_layers: int,
    ):
        super().__init__()

        self.T = T
        self.h0 = h0
        self.c0 = c0

        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(in_features=hidden, out_features=features)

    def forward(self) -> Tensor:
        B = self.h0.size(0)

        decoder_input = self.h0.unsqueeze(1).repeat(1, self.T, 1)

        h0 = self.h0.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c0 = self.c0.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)

        o_t, (h_t, c_t) = self.lstm(decoder_input, (h0, c0))

        return self.fc(o_t)


class LSTMAutoencoder(nn.Module):
    def __init__(self, features, hidden, T, num_layers):
        super().__init__()
        self.encoder = LSTMEncoder(features, hidden, num_layers)
        self.decoder = LSTMDecoder(None, None, T, features, hidden, num_layers)

    def forward(self, x: Tensor) -> Tensor:
        h_T, c_T = self.encoder(x)
        self.decoder.h0 = h_T
        self.decoder.c0 = c_T
        return self.decoder()
