import torch
import numpy as np
from mlir_env.agents.policy_based.base_agent import PolicyBasedAgent
from mlir_env.agents.policy_based.policy import ActorCriticPolicy


class ActorCriticPolicyAgent(PolicyBasedAgent):
    """Use a2c to learn a policy"""

    def __init__(self, hidden_neurons=[512, 256], gamma=0.99, *args, **kwargs):
        self.hidden = hidden_neurons
        self.update_method = "A2C"

        super().__init__(*args, **kwargs)

        self.set_name("actor_critic_policy_agent")
        self.gamma = gamma
        self.distribution = None

    def _build_nn(self) -> torch.nn.Module:
        input_dim = self.preprocessor.output_shape[
            -1
        ]  # use embedding dimension as features
        output_dim = self.env.action_space.n
        return ActorCriticPolicy(input_dim, self.hidden, output_dim)

    def get_action(self, obs):
        obs_tensor = self._obs_to_tensor(obs)
        probs, val = self.policy(obs_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

        if self.action_history:
            self.action_history.append(action)

        return action
