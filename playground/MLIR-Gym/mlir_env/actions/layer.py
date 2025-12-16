# actions/layer.py

from __future__ import annotations
import numpy as np
from gymnasium import spaces

from mlir_env.utils.common_utils import *

from mlir_env.core.config import Config
from mlir_env.core.context import EpisodeContext
from mlir_env.core.immutable import StepResult, StepReturn, Info, Status, ErrorBits

from .action import Action
from .registry import ActionRegistry
from .runner import PassRunner, EvalRunner


class ActionLayer:
    """
    A thin adaptor that wires actions/* (action, registry, runners) into the existing system.
    MLIRGymEnv owns this instance. ActionLayer does not mutate env state
    """

    def __init__(self, cfg: Config, ctx: EpisodeContext, eval_runs: int = 1):
        self.cfg = cfg
        self.ctx = ctx
        self.registry = ActionRegistry(cfg=self.cfg)
        self.runner = PassRunner(cfg=self.cfg)
        self.evaluator = EvalRunner(ctx=self.ctx, eval_runs=eval_runs)

    @property
    def space(self):
        return spaces.Discrete(len(self.registry.action_space))

    def mask(self, ir_text: str) -> np.ndarray:
        return self.registry.mask_actions(ir_text)

    def idx_to_action(self, idx: int) -> Action:
        if idx < 0 or idx >= len(self.registry.action_space):
            raise IndexError("Invalid access to action space")
        return self.registry.action_space[idx]

    def apply_action(self, ir_text: str, idx: int) -> StepReturn:
        """
        returns: (
            result: StepResult,
            term: bool,
            trun: bool,
            info: Info
        )
        """
        # get current action from idx
        action = self.idx_to_action(idx)

        # get action mask
        action_mask = self.mask(ir_text)

        # construct Info data
        info = Info(index=idx, mask=action_mask)

        # if the agent chose an action that is masked, i.e., an invalid action
        if action_mask[idx] == 0:
            e_code = ErrorBits.W_NO_CHANGE | ErrorBits.E_MASKED
            return StepReturn(
                result=StepResult(
                    new_ir=ir_text,
                    status=Status.ERROR,
                    metrics=None,
                    e_code=e_code,
                    error_msg="E: This action is masked",
                ),
                trun=False,
                term=False,
                info=info,
            )

        # if this is an eval action: use evaluator and get metrics (stored in result field)
        if action.is_eval():
            result = self.evaluator.apply(ir_text)
            return StepReturn(result=result, trun=False, term=True, info=info)
        # else this is an actionable action, use runner to
        else:
            result = self.runner.apply(ir_text, action)

            # update past action history
            if result.status is Status.OK:
                self.registry.add_entry(action)

                # update noop_window if IR + action pair has no effect
                # req: w_no_change and none -> apply successfully but no changes of IR
                if ir_not_changed.check(result.e_code):
                    self.registry.record_noop(ir_text, action)

            return StepReturn(result=result, trun=False, term=False, info=info)
