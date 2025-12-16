# views/utils.py
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCaps:
    version: str
    oneshot: bool
    affine: bool
