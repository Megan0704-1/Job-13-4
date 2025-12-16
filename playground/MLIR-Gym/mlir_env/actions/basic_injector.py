import numpy as np
import torch


class ActionHistoryVector:
    """Tracks a count histogram of actions taken in the episode."""

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.output_dim = num_actions
        self.total = 0
        self.reset()

    def reset(self):
        self.total = 0
        self.counts = np.zeros(self.num_actions, dtype=np.float32)

    def append(self, action: int):
        self.counts[action] += 1.0
        self.total += 1

    def get_vector(self):
        return (self.counts / max(1.0, self.total)).astype(np.float32)

    def inject(self, obs_vector: np.ndarray):
        hist_np = self.counts.astype(np.float32)
        if isinstance(obs_vector, torch.Tensor):
            obs_vector = obs_vector.squeeze(axis=0)
            hist_torch = torch.from_numpy(hist_np).to(obs_vector.device)
            return torch.cat((obs_vector, hist_torch), dim=-1)
        else:
            obs_vector = np.squeeze(obs_vector, axis=0)
            return np.concatenate((obs_vector, hist_np), axis=-1)


class GradientHistoryVector:
    """Track gradient updates"""

    def __init__(self):
        self.gradient_norms = []

    @property
    def get_gradients(self):
        return self.gradient_norms

    def inject(self, params):
        """
        param params: model.parameters
        """
        total_norm = 0
        for p in params:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm = param_norm.item() ** 2

        total_norm = total_norm**0.5
        self.gradient_norms.append(total_norm)
