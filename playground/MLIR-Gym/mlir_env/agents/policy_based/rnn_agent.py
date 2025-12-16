import torch
import numpy as np
from mlir_env.agents.policy_based.base_agent import PolicyBasedAgent
from mlir_env.agents.policy_based.policy import RNNPolicy


class RNNPolicyAgent(PolicyBasedAgent):
    """RL agent that uses an RNNPolicy under the hood."""

    def __init__(
        self,
        rnn_hidden: int = 128,
        num_layers: int = 1,
        gamma: float = 0.99,
        *args,
        **kwargs
    ):
        self.rnn_hidden = rnn_hidden
        self.num_layers = num_layers
        self.gamma = gamma
        self.update_method = "REINFORCE"
        super().__init__(*args, **kwargs)
        self.set_name("rnn_policy_agent")

    def _build_nn(self) -> torch.nn.Module:
        feat_dim = self.preprocessor.output_shape[-1]
        act_dim = self.env.action_space.n
        return RNNPolicy(
            input_dim=feat_dim,
            rnn_hidden=self.rnn_hidden,
            num_layers=self.num_layers,
            output_dim=act_dim,
        )
