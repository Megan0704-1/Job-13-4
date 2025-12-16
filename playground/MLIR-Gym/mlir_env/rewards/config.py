from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any
import yaml


##############################
# A reward config data class
# Expect to be initialize in mlir_world
##############################
@dataclass
class RewardConfig:
    final: str = "log_speedup"
    potential: str = "none"

    final_params: dict[str, Any] = field(default_factory=dict)
    potential_params: dict[str, Any] = field(default_factory=dict)
    step_params: dict[str, Any] = field(default_factory=dict)

    eval_on_truncate_penalty: float = 0.1
    gamma: float = 0.99

    @staticmethod
    def from_yaml(path: str) -> RewardConfig:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return RewardConfig(**data)
