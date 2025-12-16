# scripts/configs.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any

Algo = Literal["q_smoke", "ppo_gnn"]

@dataclass
class EnvConfig:
    cfg_path: str
    ctx_path: str
    state_cfg_path: Optional[str] = None
    rounds: int = 17  # episode rounds

@dataclass
class QConfig:
    episodes: int = 200
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay: int = 150
    alpha: float = 0.5
    gamma: float = 0.95
    seed: int = 21
    verbose: bool = True

@dataclass
class PPOGNNConfig:
    timesteps: int = 300_000
    n_steps: int = 1024
    batch_size: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    clip_range: float = 0.2
    # GNN
    gcn_hidden: int = 128
    gcn_out: int = 128
    dropout: float = 0.10
    # output path
    save_path: str = "out/ppo_gnn_model.zip"


@dataclass
class TrainConfig:
    algo: Algo
    env: EnvConfig
    out_path: str = "out/train_log.txt"
    q: QSmokeConfig = field(default_factory=QConfig)
    ppo: PPOGNNConfig = field(default_factory=PPOGNNConfig)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TrainConfig":
        env = EnvConfig(**d["env"])
        q = QConfig(**d.get("q", {}))
        ppo = PPOGNNConfig(**d.get("ppo", {}))
        return TrainConfig(algo=d["algo"], env=env, out_path=d.get("out_path", "out/train_log.txt"), q=q, ppo=ppo)

