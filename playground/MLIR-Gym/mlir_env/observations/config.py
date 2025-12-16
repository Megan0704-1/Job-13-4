from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any
import yaml


##############################
# A state config data class
# Expect to be initialize in mlir_world
##############################
@dataclass
class StateConfig:
    state: str = "structured_vector"
    state_params_yml: str = None

    @classmethod
    def from_yaml(cls, path: str) -> StateConfig:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return StateConfig(**data)

    @property
    def name(self):
        return self.state
