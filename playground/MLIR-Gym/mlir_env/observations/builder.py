# observations/builder.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from gymnasium import spaces

from mlir_env.core.context import EpisodeContext
from mlir_env.observations.interface import *


@dataclass
class StateBuilder:
    """
    This builder has a state class that implements `interface.py`
    Driver is expects to construct the builder by `from_config` method. This internally calls the `from_yaml` class method implements in each state class
    User is expects to call `get_observation`, `reset` using this builder
    """

    cfg: Config
    ctx: EpisodeContext
    state_cfg: StateConfig
    state_class: State | None = None

    def __post_init__(self):
        """Factory if state_class is not injected"""
        # construct state_class
        if self.state_class is None:
            cls_state = STATE_REGISTRY.get(self.state_cfg.state)
            if cls_state is None:
                raise KeyError(
                    f"Unknown state class '{self.state_cfg.state}'. "
                    f"Available: {list(STATE_REGISTRY.keys())}"
                )
            state_params_path = self.state_cfg.state_params_yml or ""
            self.state_class = cls_state.from_yaml(self.cfg, self.ctx, state_params_path)

    @classmethod
    def from_config(cls, cfg: Config, ctx: EpisodeContext, state_cfg: StateConfig) -> StateBuilder:
        # Triggers __post_init__ automatically
        return cls(cfg=cfg, ctx=ctx, state_cfg=state_cfg, state_class=None)

    def get_space(self) -> spaces.Space:
        return self.state_class.build_space()

    def get_observation(self, ir_text: str) -> spaces.Space:
        return self.state_class.observe(ir_text)

    def reset(self) -> None:
        self.state_class.reset()
