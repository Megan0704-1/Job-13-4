import torch
import torch.nn as nn


class RNNPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        rnn_hidden: int = 128,
        num_layers: int = 1,
        output_dim: int = 4,
    ):
        super().__init__()
        # or use nn.LSTM / nn.GRU
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.fc = nn.Linear(rnn_hidden, output_dim)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None):
        # If x is (batch, features), make it (batch, time=1, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # out: (batch, time, hidden); hidden: (num_layers, batch, hidden)
        out, hidden = self.rnn(x, hidden)
        logits = self.fc(out[:, -1, :])  # take the last time‐step
        return torch.softmax(logits, dim=-1)
