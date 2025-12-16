import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod
from mlir_env.utils.config import get_common_config


class RLAgent(ABC):
    def __init__(
        self,
        env: gym.Env,
        lr: float = 1e-3,
        init_epsilon: float = 1,
        epsilon_decay: float = 0.001,
        final_epsilon: float = 0.1,
        baseline: float = 1,
        *args,
        **kwargs,
    ):
        """Base RL class for all agents."""
        self.env = env
        self.lr = lr
        self.epsilon = init_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.baseline = baseline
        self.name = None
        self.checkpoint_path = get_common_config()["checkpoint_path"]

    @abstractmethod
    def get_action(self, obs):
        """Algorithm for how to choose action"""
        pass

    @abstractmethod
    def update(self, obs, act, reward, term, nxt_obs):
        """Algorithm for learning rule"""
        pass

    @abstractmethod
    def set_name(self, name):
        pass

    @property
    def get_name(self):
        return self.name

    def decay_epsilon(self):
        """Optional: Shared epsilon decay"""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
