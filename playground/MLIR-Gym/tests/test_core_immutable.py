# tests/test_core_immutable.py
import pytest
from mlir_env.core.immutable import Phase, Requirement


def test_requirement_check_all_any_none():
    req = Requirement(
        all_of=Phase.TENSOR | Phase.LINALG,
        none_of=Phase.LLVM,
        any_of=Phase.VECTOR | Phase.SCF,
    )

    # missing LINALG -> fail
    assert not req.check(Phase.TENSOR | Phase.VECTOR)
    # has all_of and any_of (VECTOR), none_of not present -> pass
    assert req.check(Phase.TENSOR | Phase.LINALG | Phase.VECTOR)
    # none_of present (LLVM) -> fail
    assert not req.check(Phase.TENSOR | Phase.LINALG | Phase.VECTOR | Phase.LLVM)
    # any_of not present -> fail
    assert not req.check(Phase.TENSOR | Phase.LINALG)
