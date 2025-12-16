import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x = (N, seq_len, feature_dim)
        attn_weight = self.attention(x)
        attn_weight = torch.softmax(attn_weight, dim=1)
        # aggregate over weighted feature dim
        pooled = (x * attn_weight).sum(dim=1)
        return pooled


class MLPPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden: list, output_dim: int):
        super().__init__()
        layers = []
        prv_dim = input_dim

        for neurons in hidden:
            layers.append(nn.Linear(prv_dim, neurons))
            layers.append(nn.ReLU())
            prv_dim = neurons

        layers.append(nn.Linear(prv_dim, output_dim))

        self.pooling = AttentionPooling(input_dim)
        self.fc = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x (torch.Tensor) : expect shapes (batch size, num_tokens, feature_dim)
        return : tensor
        """

        if x.dim() == 3:
            x = self.pooling(x)

        x = self.fc(x)
        return self.softmax(x)
