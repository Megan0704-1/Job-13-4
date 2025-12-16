import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (batch_size, num_tokens, feature_dim)
        attn_weight = self.attention(x)  # (batch_size, num_tokens, 1)
        attn_weight = F.softmax(attn_weight, dim=1)  # (batch_size, num_tokens, 1)
        pooled = torch.sum(x * attn_weight, dim=1)  # (batch_size, feature_dim)
        return pooled


class ActorCriticPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden: list, output_dim: int):
        super().__init__()
        self.pooling = AttentionPooling(input_dim, hidden[0])

        layers = []
        prv_dim = input_dim
        for neurons in hidden:
            layers.append(nn.Linear(prv_dim, neurons))
            layers.append(nn.ReLU())
            prv_dim = neurons

        self.net = nn.Sequential(*layers)

        self.actor = nn.Linear(prv_dim, output_dim)
        self.critic = nn.Linear(prv_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x (torch.Tensor) : expect shapes (batch size, num_tokens, feature_dim)
        return : tensor
        """
        if x.ndim == 3:
            x = self.pooling(x)
        features = self.net(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)  # shape: [batch]
        probs = F.softmax(logits, dim=-1)
        return probs, value
