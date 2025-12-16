# observations/interface.py
# This file declares the interface for observer and state spaces
# Goal is to declare methods and fields for different observation classes.
########################################
# To be more specific
#   xxx_xxx_REGISTRY is a dict of driver to call.
#   ABC classes are declared for builer classes to implement.
########################################

from abc import ABC, abstractmethod
from typing import Optional, Dict, Type, Any
import numpy as np
from gymnasium import spaces


# Interface for State classes
class State(ABC):
    """
    Produces observations, every state class inherit from here.
    """

    @classmethod
    @abstractmethod
    def from_yaml(cls, config_path: str) -> "State": ...

    @abstractmethod
    def build_space(self) -> spaces.Space: ...

    @abstractmethod
    def observe(
        self, ir_text: str
    ) -> Any: ...  # the return type must belong to gym.spaces

    @abstractmethod
    def schema_id(self) -> str: ...

    @abstractmethod
    def schema_meta(self) -> Dict[str, Any]: ...  # JSON-serializable class information

    @abstractmethod
    def reset(self) -> None: ...


# exposing registries
STATE_REGISTRY: Dict[str, Type[State]] = {}


# decorator to be put on state classes.
def register_state(name: str):
    def wrapper(cls_state: Type[State]):
        STATE_REGISTRY[name] = cls_state
        return cls_state

    return wrapper
