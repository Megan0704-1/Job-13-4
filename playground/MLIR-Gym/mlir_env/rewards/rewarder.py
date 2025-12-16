# rewards/rewarder.py
# This file implements the interface for mlir_world to interact with reward classes.
# 2 methods to init a rewarder.
# 1) pass in final and potential directly -> for testing
# 2) init using the registry -> for easier APIs

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from mlir_env.core.immutable import StepReturn, Status
from mlir_env.core.context import EpisodeContext
from mlir_env.rewards.config import RewardConfig
from mlir_env.rewards.interface import *
from mlir_env.utils.common_utils import *


@dataclass
class Rewarder:
    """
    final, potential fields are left for testing
    User can specify which reward class to use directly in reward config.yml file (string),
    they are then post init by rewarder factory.
    """

    cfg: RewardConfig
    final: FinalReward | None = None
    potential: Potential | None = None
    masked_penalty = 0.1
    eval_on_truncate_penalty = 0.05

    def __post_init__(self):
        """Factory if final/potential is not injected"""
        # make final
        if self.final is None:
            cls_final = FINAL_REWARD_REGISTRY.get(self.cfg.final)
            if cls_final is None:
                raise KeyError(
                    f"Unknown final reward '{self.cfg.final}'. "
                    f"Available: {list(FINAL_REWARD_REGISTRY.keys())}"
                )
            params = self.cfg.final_params or {}
            self.final = cls_final(**params)

        # make potential
        if self.potential is None and self.cfg.potential != "none":
            cls_pot = POTENTIAL_REGISTRY.get(self.cfg.potential)
            if cls_pot is None:
                raise KeyError(
                    f"Unknown potential '{self.cfg.potential}'. "
                    f"Available: {list(POTENTIAL_REGISTRY.keys())}"
                )
            params = self.cfg.potential_params or {}
            self.potential = cls_pot(**params)

        # get step params
        sp = self.cfg.step_params or {}
        self.masked_penalty = float(sp.get("masked_penalty", 0.05))
        self.eval_on_truncate_penalty = float(sp.get("eval_on_truncate_penalty", 0.05))

    @classmethod
    def from_config(cls, cfg: RewardConfig) -> Rewarder:
        # Triggers __post_init__ automatically
        return cls(cfg=cfg, final=None, potential=None)

    # TODO(megan.kuo) PBRS
    def get_reward(self, ctx: EpisodeContext, sr: StepReturn) -> float:
        """
        Refer to ADR
        """
        if sr.result.status is Status.ERROR:
            if action_masked_out.check(sr.result.e_code):
                return -self.masked_penalty
            c = getattr(self.final, "c_penalty", 10.0)  # explicit
            return -float(c)

        assert action_run_success.check(
            sr.result.e_code
        ), "E: Rewarder::get_reward, unexpected e_code."

        # successful non-terminal action
        if not sr.term and not sr.trun:
            return 0  # self.potential.value({})

        r = self.final.compute(ctx, sr.result.metrics)
        if sr.trun:
            r -= self.eval_on_truncate_penalty

        # success term
        return r
