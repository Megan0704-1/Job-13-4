# actions/action.py
# this file implement Action Data class
# scope name is used majorly for debugging purpose

from dataclasses import dataclass
from typing import Optional
from mlir_env.core.immutable import Scope, Phase, Requirement


@dataclass(frozen=True, slots=True)
class Action:
    name: str
    scope: Scope
    req: Requirement
    options: Optional[str] = None

    @property
    def scope_name(self) -> str:
        return f"{self.scope}::{self.name}{f'({self.options})' if self.options else ''}"

    def pipeline_atom(self) -> str:
        if self.is_eval():
            return ""
        return f"{self.name}{f'({self.options})' if self.options else ''}"

    def pipeline_body(self) -> str:
        """Return substring that can be directly included in builtin.module"""
        if self.is_eval():
            raise ValueError("pipeline for eval is defined in the context.")
        atom = self.pipeline_atom()
        return f"func.func({atom})" if self.scope == "func" else atom

    def to_pipeline_flag(self, container: str = "builtin.module") -> str:
        """Helper function, cleaner for single pipeline execution"""
        return f"--pass-pipeline={container}({self.pipeline_body()})"

    def is_eval(self):
        return self.scope == "env" and self.name == "EVAL"

    # Read this function as: can this action be selected based on the phase of the IR?
    def precond(self, phase: Phase) -> bool:
        return True if self.is_eval() else self.req.check(phase)
