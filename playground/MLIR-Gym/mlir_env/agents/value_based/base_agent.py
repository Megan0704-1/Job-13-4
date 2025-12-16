import numpy as np
from collections import defaultdict
from mlir_env.agents.base_agent import RLAgent


class ValueBasedAgent(RLAgent):
    """For algorithm using a value based logic"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))

    def get_action(self, obs):
        """
        if entropy < epsilon: random choice of action
        else: use policy
        """
        entropy = np.random.rand()
        obs_tensor = self._obs_to_tensor(obs)

        if entropy < self.epsilon:
            print("by random")
            nxt_action = self.env.action_space.sample()
        else:
            print("by policy")
            with torch.no_grad():
                action_prob = self.policy.forward(obs_tensor)
            nxt_action = torch.argmax(action_prob[0], dim=-1).item()

        if self.action_history:
            self.action_history.append(nxt_action)
