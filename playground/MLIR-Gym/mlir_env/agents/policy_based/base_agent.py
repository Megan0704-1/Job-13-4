import os
import torch
import numpy as np
from abc import abstractmethod

from mlir_env.utils import get_env_file
from mlir_env.actions import ActionHistoryVector
from mlir_env.agents.base_agent import RLAgent
from mlir_env.observations.preprocessor.base_preprocessor import ObservationPreprocessor


class PolicyBasedAgent(RLAgent):
    """For algorithms directly leraning a policy"""

    def __init__(self, preprocessor: ObservationPreprocessor, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # injectors
        self.action_history = kwargs.get("action_history", None)
        self.gradient_history = kwargs.get("gradient_history", None)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = preprocessor
        self.preprocessor._output_shape += self.action_history.output_dim
        self.policy = self._build_nn().to(self.device)

        # nn related setups
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    @abstractmethod
    def _build_nn(self) -> torch.nn.Module:
        """Subclasses must defined their network structuer"""
        pass

    def set_name(self, name):
        self.name = name

    def get_action(self, obs):
        obs_tensor = self._obs_to_tensor(obs)
        probs = self.policy(obs_tensor).squeeze(0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

        if self.action_history:
            self.action_history.append(action)

        return action

    def discount_reward(self, rewards):
        """
        Discount non-zero reward in the given list by gamma backward
        """
        returns = []
        look_back = 0

        for r in reversed(rewards):
            if r != 0:  # episode return
                look_back = 0
            look_back = r + self.gamma * look_back
            returns.append(look_back)

        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        returns = list(reversed(returns))

        return torch.tensor(returns, dtype=torch.float32)

    def load_checkpoint(self):
        checkpoint_path = self.checkpoint_path + self.get_name

        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}. Start from scratch")
            return 0, torch.inf
        else:
            print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"], checkpoint["loss"]

    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }
        # checkpoint/policy-name/benchmark-name
        save_path = self.checkpoint_path + self.get_name
        torch.save(checkpoint, save_path)

    def update_using_REINFORCE(self, trajectories):
        """Update using the REINFORCE algorithm."""
        obs, actions, rewards = trajectories
        obs_tensor = torch.stack([self._obs_to_tensor(o) for o in obs]).to(self.device)
        act_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        return_tensor = self.discount_reward(rewards).to(self.device)

        nxt_act_probs = self.policy(obs_tensor)
        self.distribution = torch.distributions.Categorical(nxt_act_probs)
        act_log_prob = self.distribution.log_prob(act_tensor)

        batch_return = return_tensor.mean()
        self.baseline = 0.9 * self.baseline + 0.1 * batch_return
        advantage = return_tensor - self.baseline
        entropy = self.distribution.entropy().mean()
        loss = -torch.mean(act_log_prob * advantage)
        loss -= 0.01 * entropy

        return loss

    def update_using_PPO(self, trajectories):
        """
        Update using a simplified PPO update.
        TODO: update this
        """
        obs, actions, rewards = trajectories

        obs_tensor = torch.stack([self._obs_to_tensor(o) for o in obs]).to(self.device)
        act_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        returns = self.discount_reward(rewards).to(self.device)

        new_probs, values = self.policy(obs_tensor)
        dist = torch.distributions.Categorical(new_probs)
        new_logp = dist.log_prob(act_tensor)

        if self.old_logp_tensor is None:
            # On the first batch, we don't have past log probabilities,
            # so we initialize them using the current policy's log probabilities.
            self.old_logp_tensor = new_logp.clone().detach()
        else:
            self.old_logp_tensor = torch.tensor(
                self.old_log_probs, dtype=torch.float
            ).to(self.device)

        # If values have an extra dimension (e.g., [batch, 1]), squeeze it.
        if values.dim() > 1 and values.size(1) == 1:
            values = values.squeeze(1)
        adv = returns - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO surrogate objective: compute the probability ratio and clip it
        ratio = torch.exp(new_logp - self.old_logp_tensor)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        actor_loss = -torch.min(surr1, surr2).mean()

        # Critic loss (mean squared error)
        critic_loss = adv.pow(2).mean()

        # Entropy bonus for exploration
        entropy = dist.entropy().mean()

        # Total PPO loss
        loss = actor_loss + critic_coef * critic_loss - entropy_coef * entropy

        return loss

    def update_using_A2C(self, trajectories):
        """
        Update using Actor-critic policy
        """
        obs, actions, rewards = trajectories
        obs_tensor = torch.stack([self._obs_to_tensor(o) for o in obs])
        act_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        return_tensor = self.discount_reward(rewards).to(self.device)

        nxt_act_probs, values = self.policy(obs_tensor)
        self.distribution = torch.distributions.Categorical(nxt_act_probs)
        act_log_prob = self.distribution.log_prob(act_tensor)

        advantages = return_tensor - values
        actor_loss = -(act_log_prob * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss
        entropy = self.distribution.entropy().mean()
        loss -= 0.01 * entropy

        return loss

    def update(self, trajectories):
        """General update method. It calls the chosen update method based on self.update_method."""
        result = None
        match self.update_method:
            case "REINFORCE":
                loss = self.update_using_REINFORCE(trajectories)
            case "PPO":
                loss = self.update_using_PPO(trajectories)
            case "A2C":
                loss = self.update_using_A2C(trajectories)
            case _:
                raise NotImplementedError(
                    f"Update method {self.update_method} not implemented."
                )

        self.optimizer.zero_grad()
        loss.backward()

        # gradient injector
        self.gradient_history.inject(self.policy.parameters())

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

        self.optimizer.step()

        self.decay_epsilon()

        # action injector
        if self.action_history:
            self.action_history.reset()

        return loss.item(), self.epsilon

    def _obs_to_tensor(self, obs):
        obs = self.preprocessor.preprocess(obs)

        if self.action_history:
            obs_array = self.action_history.inject(obs)

        return torch.tensor(obs_array, dtype=torch.float32).to(self.device)
