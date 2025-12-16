import torch
import numpy as np
from mlir_env.agents.policy_based.base_agent import PolicyBasedAgent
from mlir_env.agents.policy_based.policy import MLPPolicy


class MLPPolicyAgent(PolicyBasedAgent):
    """Use MLP to learn a policy"""

    def __init__(self, hidden_neurons=[512, 256], gamma=0.99, *args, **kwargs):
        self.hidden = hidden_neurons
        self.update_method = "REINFORCE"

        super().__init__(*args, **kwargs)

        self.set_name("mlp_policy_agent")
        self.gamma = gamma
        self.distribution = None

    def _build_nn(self) -> torch.nn.Module:
        input_dim = self.preprocessor.output_shape[
            -1
        ]  # use embedding dimension as features
        output_dim = self.env.action_space.n
        return MLPPolicy(input_dim, self.hidden, output_dim)
