import torch
import torch.nn as nn
import torch.nn.functional as F


class TransfomerPolicyAgent(Policy):
    def __init__(
        self, hidden_dim=128, num_layers=2, num_heads=4, gamma=0.99, *args, **kwargs
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.gamma = gamma

        super().__init__()

        self.set_name("transformer_policy_agent")
        self.update_method = "REINFORCE"
        self.distribution = None

    def _build_nn(self) -> torch.nn.Module:
        input_dim = self.preprocessor.output_shape[-1]
        output_dim = self.env.action_space.n
        return TransformerPolicy(
            input_dim, self.hidden_dim, self.num_layers, self.num_heads, output_dim
        )
