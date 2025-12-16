# tests/test_rewards_rewarder.py

import math
import numpy as np
import pytest
from mlir_env.rewards.config import RewardConfig
from mlir_env.rewards.rewarder import Rewarder
from mlir_env.rewards.interface import FinalReward, FINAL_REWARD_REGISTRY
from mlir_env.rewards.finals.log_speedup import LogSpeedup
from mlir_env.core.immutable import (
    Metrics,
    StepResult,
    StepReturn,
    Info,
    Status,
    ErrorBits,
)
from mlir_env.utils.common_utils import *


# ---------- helpers ----------
class DummyCtx:
    def __init__(self, L0=100.0, C0=50.0, B0=200.0):
        self.L0, self.C0, self.B0 = L0, C0, B0

    def get_baseline(self):
        return (self.L0, self.C0, self.B0)


def mk_step_return(
    *,
    status=Status.OK,
    term=False,
    trun=False,
    latency_ms=None,
    compile_ms=None,
    size_kb=None,
    e_code=ErrorBits.NONE,
    error_msg="",
    idx=0,
    mask_len=8
):
    m = None
    if (latency_ms is not None) or (compile_ms is not None) or (size_kb is not None):
        m = Metrics(latency_ms=latency_ms, compile_ms=compile_ms, size_kb=size_kb)
    res = StepResult(
        new_ir="", status=status, metrics=m, e_code=e_code, error_msg=error_msg
    )
    info = Info(index=idx, mask=([1] * mask_len))
    return StepReturn(result=res, trun=trun, term=term, info=info)


# ---------- tests ----------


def test_rewarder_factory_from_config_and_terminal_success():
    cfg = RewardConfig(final="log_speedup", potential="none")
    rw = Rewarder.from_config(cfg)
    assert isinstance(rw.final, FinalReward)

    ctx = DummyCtx(L0=100.0)
    sr = mk_step_return(status=Status.OK, term=True, trun=False, latency_ms=50.0)
    r = rw.get_reward(ctx, sr)
    assert np.allclose(r, -math.log(0.5))


def test_rewarder_unknown_final_raises_keyerror():
    cfg = RewardConfig(final="__unknown__", potential="none")
    with pytest.raises(KeyError):
        Rewarder.from_config(cfg)


def test_nonterminal_valid_step_returns_d_reward_passthrough():
    cfg = RewardConfig(final="log_speedup", potential="none")
    rw = Rewarder.from_config(cfg)
    ctx = DummyCtx()
    sr = mk_step_return(status=Status.OK, term=False, trun=False)
    r = rw.get_reward(ctx, sr)
    assert np.allclose(
        r, 0.0
    )  # FIXME(megan.kuo) this is a temporal value returned by the rewarder for successsful non-terminal step.


def test_masked_action_nonterminal_small_negative():
    cfg = RewardConfig(final="log_speedup", potential="none")
    rw = Rewarder.from_config(cfg)
    ctx = DummyCtx()
    # imitate masked：status=Status.ERROR, term=False
    e_code = ErrorBits.E_MASKED | ErrorBits.W_NO_CHANGE
    sr = mk_step_return(
        status=Status.ERROR, term=False, trun=False, error_msg="E:MASKED", e_code=e_code
    )
    r = rw.get_reward(ctx, sr)
    assert action_masked_out.check(sr.result.e_code)
    assert (
        r < 0.0
    )  # FIXME(megan.kuo) this is a temporal value returned by the rewarder for masked action.


def test_terminal_timeout_or_failed_uses_c_penalty():
    cfg = RewardConfig(final="log_speedup", potential="none")
    rw = Rewarder.from_config(cfg)
    ctx = DummyCtx()
    sr_timeout = mk_step_return(
        status=Status.ERROR, term=True, trun=False, error_msg="E:TIMEOUT"
    )
    sr_failed = mk_step_return(
        status=Status.ERROR, term=True, trun=False, error_msg="E:FAILED"
    )
    assert rw.get_reward(ctx, sr_timeout) == -rw.final.c_penalty
    assert rw.get_reward(ctx, sr_failed) == -rw.final.c_penalty


# imitate truncation behavior
def test_truncated_applies_eval_on_truncate_penalty(monkeypatch):
    # set eval_on_truncate_penalty field
    cfg = RewardConfig(
        final="log_speedup", potential="none", eval_on_truncate_penalty=0.3
    )
    rw = Rewarder.from_config(cfg)
    ctx = DummyCtx(L0=100.0)

    # success term and trun=True e.g., reach max step and apply success
    sr = mk_step_return(status=Status.OK, term=True, trun=True, latency_ms=100.0)
    r = rw.get_reward(ctx, sr)
    # base reward - eval_on_truncate cost
    assert np.allclose(r, -rw.eval_on_truncate_penalty)
