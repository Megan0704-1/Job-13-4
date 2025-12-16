# rewards/interface.py
# This file declares the interface for FinalReward and Potential class
# Goal is to declare fields for different rewarding mechanism and expose APIs for caller
########################################
# To be more specific
#   xxx_xxx_REGISTRY is a dict for rewarder to call.
#   ABC classes are declared for reward classes to implement.
########################################
# Note. classes need to initialize with ctx to get baseline numbers.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Type, Any
from mlir_env.core.immutable import Metrics
from mlir_env.core.context import EpisodeContext


# Interface for finalrewards
class FinalReward(ABC):
    @abstractmethod
    def compute(self, ctx: EpisodeContext, metrics: Metrics) -> float: ...


# TODO(megan.kuo): to be done in phase-2
class Potential(ABC):
    def value(self, obs: dict, feats: dict | None = None) -> float: ...
    def on_terminal(self, obs: dict, metrics: Metrics, ctx: EpisodeContext) -> None: ...


# exposing registries
FINAL_REWARD_REGISTRY: Dict[str, Type[FinalReward]] = {}
POTENTIAL_REGISTRY: Dict[str, Type[Potential]] = {}


# decorator to be put on reward classes.
def register_final(name: str):
    def wrapper(cls_final: Type[FinalReward]):
        FINAL_REWARD_REGISTRY[name] = cls_final
        return cls_final

    return wrapper


def register_potential(name: str):
    def wrapper(cls_pot: Type[Potential]):
        POTENTIAL_REGISTRY[name] = cls_pot
        return cls_pot

    return wrapper
