# mlir_env/core/immutable.py
from dataclasses import dataclass
from typing import Literal, NewType
from enum import IntFlag, auto, Enum
import numpy as np

Scope = Literal["func", "module", "env"]
ActionID = NewType("ActionID", int)


# --- Measurement dataclass --- #
# includes: latency, compile time, object bytes
@dataclass(frozen=True, slots=True)
class Metrics:
    latency_ms: float
    compile_ms: float
    size_kb: float


# --- Step dataclass --- #
@dataclass(frozen=True, slots=True)
class Info:
    index: int
    mask: np.ndarray


# --- Action dataclass (Errors) --- #
class Status(Enum):
    OK = True
    ERROR = False


# error bit class is designed with ADR table in mind.
# see [ADR: reward-change](https://github.com/MPSLab-ASU/MLIR-Gym/blob/eba40d91a41c9de03ea028a9e5b2e98df425d47a/docs/adr/reward-change.md)
class ErrorBits(IntFlag):
    NONE = 0
    # no change of IR
    W_NO_CHANGE = auto()
    # did not satisfied precond
    W_FAILED_PRECOND = auto()
    # chose a masked action
    E_MASKED = auto()
    # an action result in timeout
    E_TIMEDOUT = auto()
    # an action failed
    E_FAILED = auto()
    # evaluation failed
    E_EVAL_FAILED = auto()


# --- Actions dataclass --- #
# result return by runners
@dataclass(frozen=True, slots=True)
class StepResult:
    new_ir: str
    status: Status
    metrics: Metrics | None
    e_code: ErrorBits
    error_msg: str


# --- Actions dataclass --- #
# result return by action layer
@dataclass(frozen=True, slots=True)
class StepReturn:
    result: StepResult
    trun: bool
    term: bool
    info: Info


# --- State dataclass --- #
# IR phase, this class is use for masking possible next actions
# IntFlag make this enum class combinable
class Phase(IntFlag):
    NONE = 0
    TENSOR = auto()
    LINALG = auto()
    AFFINE = auto()
    MATH = auto()
    ARITH = auto()
    MEMREF = auto()
    VECTOR = auto()
    SCF = auto()
    CF = auto()
    FUNC = auto()
    LLVM = auto()


# --- Actions dataclass --- #
# precond class : requirement
# Every pass before adding as an action must
# have a requirement for masking
# TODO(megan.kuo) add field for finer preconditions.
@dataclass(frozen=True, slots=True)
class Requirement:
    # require that ALL bits in all_of are present
    all_of: int = 0
    # forbid ANY of these bits from being present
    none_of: int = 0
    # require that AT LEAST ONE of these bits is present
    any_of: int = 0

    def check(self, phase: Phase) -> bool:
        if (phase & self.all_of) != self.all_of:
            return False
        if self.none_of and (phase & self.none_of):
            return False
        if self.any_of and (phase & self.any_of) == 0:
            return False
        return True
