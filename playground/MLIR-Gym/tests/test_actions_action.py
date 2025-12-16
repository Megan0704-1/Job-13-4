# tests/test_actions.py
import pytest
from mlir_env.core.immutable import Phase, Requirement
from mlir_env.actions.action import Action


def test_action_eval_and_pipeline_atom():
    action = Action("EVAL", "env", Requirement())
    assert action.is_eval()
    assert action.precond(Phase.TENSOR)  # eval always legal
    assert action.pipeline_atom() == ""  # never appears inside pipeline


def test_action_with_options_scope_name_and_atom():
    a = Action("tile-loops", "func", Requirement(), options="tile-sizes=4,4")
    assert a.scope_name == "func::tile-loops(tile-sizes=4,4)"
    assert a.pipeline_atom() == "tile-loops(tile-sizes=4,4)"
    assert a.precond(Phase.TENSOR)  # no req limits → True


def test_action_pipeline_strings():
    a = Action("tile-loops", "func", Requirement(), options="tile-sizes=4,4")
    b = Action("cse", "module", Requirement())
    assert a.pipeline_body() == "func.func(tile-loops(tile-sizes=4,4))"
    assert (
        a.to_pipeline_flag()
        == "--pass-pipeline=builtin.module(func.func(tile-loops(tile-sizes=4,4)))"
    )
    assert b.pipeline_body() == "cse"
    assert b.to_pipeline_flag() == "--pass-pipeline=builtin.module(cse)"
