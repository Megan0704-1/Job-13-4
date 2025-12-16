import os
import numpy as np
import pytest

from mlir_env.core.config import Config
from mlir_env.core.immutable import StepResult, Requirement, Status, ErrorBits
from mlir_env.actions.action import Action
from mlir_env.actions.registry import ActionRegistry
from mlir_env.actions.layer import ActionLayer
from mlir_env.utils.common_utils import *


class DummyCfg:
    cpu_core = 0
    compile_option = "affine"
    pass_limits = 10
    hash_window_len = 8
    k_actions = 3
    eval_timeout_s = 12
    eval_runs = 5
    llvm_build_dir = os.environ.get("LLVM_BUILD_DIR")
    benchmark_sh = os.environ.get("MLIR_GYM_ROOT") + "/benchmarks/benchmark.sh"


class DummyCtx:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline_content = "builtin.module(cse,func.func(canonicalize))"


class FakePassRunner:
    def __init__(self, result: StepResult):
        self.result = result
        self.calls = 0

    def apply(self, ir_text, action):
        self.calls += 1
        return self.result


class FakeEvalRunner:
    def __init__(self, result: StepResult):
        self.result = result
        self.calls = 0

    def apply(self, ir_text):
        self.calls += 1
        return self.result


def make_space():
    return [
        Action("cse", "module", Requirement()),
        Action("EVAL", "env", Requirement()),
    ]


def make_layer_with_custom_space():
    cfg = DummyCfg()
    ctx = DummyCtx(cfg)
    layer = ActionLayer(cfg=cfg, ctx=ctx, eval_runs=1)
    layer.registry = ActionRegistry(cfg=cfg, action_space=make_space())
    return layer


def test_idx_to_action_out_of_range_raises():
    layer = make_layer_with_custom_space()
    with pytest.raises(IndexError):
        layer.idx_to_action(999)


# This function tests when layer applies a masked-out action
def test_invalid_masked_action_gets_penalty(monkeypatch):
    layer = make_layer_with_custom_space()

    # masked-out all actions
    mask = np.zeros(len(layer.registry.action_space), dtype=np.int8)
    monkeypatch.setattr(layer, "mask", lambda _ir: mask)

    # apply a masked-out action
    step = layer.apply_action("ir", 0)
    assert step.result.status is Status.ERROR
    assert action_masked_out.check(step.result.e_code)


# This function tests layer.evaluator
# 1. get eval action index
# 2. call layer.apply_action (internally calls evaluator)
# -> expects term is True
def test_eval_action_terminates_and_uses_evaluator(monkeypatch):
    layer = make_layer_with_custom_space()
    eval_idx = [i for i, a in enumerate(layer.registry.action_space) if a.is_eval()][0]

    fake_eval_result = StepResult(
        new_ir="ir",
        status=Status.OK,
        metrics={"ms": 1.0},
        e_code=ErrorBits.NONE,
        error_msg="",
    )
    layer.evaluator = FakeEvalRunner(fake_eval_result)

    sr = layer.apply_action("ir", eval_idx)
    assert sr.term is True
    assert sr.result.metrics == {"ms": 1.0}
    assert action_run_success.check(sr.result.e_code)
    assert layer.evaluator.calls == 1


# This function tests add_entry and noop_window update logic
def test_non_eval_action_updates_registry_and_noop_on_unchanged(monkeypatch):
    layer = make_layer_with_custom_space()
    non_eval_idx = 0
    non_eval_action = layer.registry.action_space[non_eval_idx]

    e_code = ErrorBits.NONE
    changed_res = StepResult(
        new_ir="IR2", status=Status.OK, metrics=None, e_code=e_code, error_msg=""
    )
    layer.runner = FakePassRunner(changed_res)

    calls = {"add": 0, "noop": 0}

    def add_entry_spy(a):
        assert a == non_eval_action
        calls["add"] += 1

    def noop_spy(ir, a):
        calls["noop"] += 1

    layer.registry.add_entry = add_entry_spy
    layer.registry.record_noop = noop_spy

    # changed=True -> add_entry, no updates to noop_window
    sr1 = layer.apply_action("IR1", non_eval_idx)
    assert sr1.term is False and action_run_success.check(sr1.result.e_code)
    assert calls["add"] == 1 and calls["noop"] == 0

    # changed=False -> add_entry and update noop_window
    e_code = ErrorBits.W_NO_CHANGE | ErrorBits.NONE
    unchanged_res = StepResult(
        new_ir="IR1", status=Status.OK, metrics=None, e_code=e_code, error_msg=""
    )
    layer.runner = FakePassRunner(unchanged_res)
    sr2 = layer.apply_action("IR1", non_eval_idx)
    assert ir_not_changed.check(sr2.result.e_code)
    assert calls["add"] == 2 and calls["noop"] == 1
