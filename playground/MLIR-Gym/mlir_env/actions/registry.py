# actions/registry.py
# A class that keep records of past actions in 1 episode, pass use counts.
# Note. Env owns the class, external user read-only
# 1 Episode, 1 ActionRegistry

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Deque, Tuple, Union
import collections

from mlir_env.core.immutable import Phase
from mlir_env.core.config import Config
from mlir_env.utils.ir_utils import get_ir_hash, detect_phase_from_ir

from .action import Action
from .space import get_action_space_default


@dataclass
class ActionRegistry:
    cfg: Config
    action_space: Optional[List[Action]] = None
    past_actions: List[Action] = field(default_factory=list)
    used_action_counts: Dict[str, int] = field(default_factory=dict)
    noop_window: Deque[Tuple[str, str]] = field(init=False)

    def __post_init__(self):
        # initialize action space (from default action space)
        if not self.action_space:
            self.action_space = get_action_space_default()

        # initialize used action counts (a dict of 0s)
        self.used_action_counts = {action.scope_name: 0 for action in self.action_space}

        # maks sure eval exist in action space
        assert any(
            action.is_eval() for action in self.action_space
        ), "E: Action space must have an eval action"

        # initialize noop hash window (len: cfg.hash_window_len, entry: {ir_hash, action})
        self.noop_window = collections.deque(maxlen=self.cfg.hash_window_len)

    def add_entry(self, action: Action) -> None:
        self.past_actions.append(action)
        self.used_action_counts[action.scope_name] = (
            self.used_action_counts.get(action.scope_name, 0) + 1
        )

    def record_noop(self, ir_text: str, action: Action) -> None:
        """Add a noop hash in self.noop_window"""
        h = get_ir_hash(ir_text)
        self.noop_window.append((h, action.scope_name))

    def is_noop_recent(self, ir_text: str, action: Action) -> bool:
        """Check if a noop appear in recent window"""
        h = get_ir_hash(ir_text)
        return (h, action.scope_name) in self.noop_window

    def mask_actions(self, _from: Union[Phase, str]) -> np.ndarray:
        """
        Return a list of 0 and 1, 1 means it is actionable for this IR , 0 otherwise
        Definition of an action is actionable:
        1. it is an eval action
        2. use_counts of this pass hasn't yet pass the limit
        3. this IR (phase) satisfy the precond of the action.
        4. if this IR-action pair appear in noop_window recently
        """
        phase = _from if isinstance(_from, Phase) else detect_phase_from_ir(_from)
        ir_text = _from if isinstance(_from, str) else None

        mask = np.zeros(len(self.action_space), dtype=np.int8)
        limit = self.cfg.pass_limits
        for idx, action in enumerate(self.action_space):
            if action.is_eval():
                # never mask eval
                mask[idx] = 1
                continue

            capped = (
                limit > 0 and self.used_action_counts.get(action.scope_name, 0) >= limit
            )
            if capped:
                mask[idx] = 0
                continue

            if ir_text and self.is_noop_recent(ir_text, action):
                mask[idx] = 0
                continue

            mask[idx] = 1 if action.precond(phase) else 0

        return mask

    def legal_indices(self, phase: Phase) -> np.ndarray:
        """Returns a list of legal indices in action space."""
        mask = self.mask_actions(phase)
        return (mask.astype(bool)).nonzero()[0]

    def legal_actions(self, phase: Phase) -> List[Action]:
        """Returns a list of legal action name."""
        indices = self.legal_indices(phase)
        return [self.action_space[idx] for idx in indices]

    def get_last_k(self) -> List[Action]:
        k = self.cfg.k_actions
        assert k >= 0
        return self.past_actions[-k:]

    def get_all_action_names(self) -> List[str]:
        return [action.scope_name for action in self.action_space]

    def get_past_action_names(self) -> List[str]:
        return [action.scope_name for action in self.past_actions]

    def get_pipeline_command(self) -> str:
        """
        Get shell command from past actions
        e.g.,
        past actions = ['module::print-memref', 'func::tile-loops(tile-size=5)']
        -> command = builtin.module(print-memref,func.func(tile-loops(tile-size=5)))
        """
        bodies = [
            action.pipeline_body()
            for action in self.past_actions
            if not action.is_eval()
        ]
        command = ",".join(bodies)
        return f"builtin.module({command})" if command else ""
