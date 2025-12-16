import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        output_dim: int,
    ):
        """
        Args:
        input_dim (int): dim of input observation
        hidden_dim (int): dim of the transformer
        num_layers (int): # of transformer encoder layers
        num_heads (int): # of attention heads per encoder layer
        output_dim (int): # of actions
        """
        super().__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            # if input is (batch, input_dim) add a sequence dim -> (batch, 1, input_dim)
            x = x.unsqueeze(1)

        x = self.embedding(x)
        # transformer expects (seq_len, batch, dim)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)

        x = x[0]
        logits = self.fc(x)

        return self.softmax(logits)
