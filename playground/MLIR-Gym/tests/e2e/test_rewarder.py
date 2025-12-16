# tests/e2e/test_rewarder.py
import math
import numpy as np
import textwrap
import pytest

from mlir_env.actions.layer import ActionLayer
from mlir_env.rewards.config import RewardConfig
from mlir_env.rewards.rewarder import Rewarder
from mlir_env.rewards.finals.log_speedup import LogSpeedup
from mlir_env.core.immutable import StepReturn, StepResult, Info, Metrics, Status
from mlir_env.utils.common_utils import *


# ---------- helpers ----------
def minimal_ir():
    return textwrap.dedent(
        """\
    module {
      func.func @main() {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %s = arith.addi %c0, %c1 : i32
        return
      }
    }
    """
    )


def build_rewarder(c_penalty=10.0, lambda_ct=0.0, beta_kb=0.0):
    cfg = RewardConfig(
        final="log_speedup",
        potential="none",
        final_params={
            "c_penalty": c_penalty,
            "lambda_ct": lambda_ct,
            "alpha_ct": 1.2,
            "beta_kb": beta_kb,
            "gamma_kb": 1.0,
            "corrections": 1e-6,
        },
        potential_params={},
    )
    return Rewarder.from_config(cfg)


# ---------- tests ----------


@pytest.mark.e2e
def test_invalid_action_penalized_by_rewarder(
    cfg_from_yaml, ctx_from_yaml, monkeypatch
):
    """
    目標：invalid（masked）動作不由 ActionLayer 給 -1，
    而是由 Rewarder 集中計分（例如 -invalid_penalty）。
    條件：選一個在 minimal_ir 下必然 masked 的動作（比如需要 TENSOR/LINALG 的 pass）。
    """
    ir = minimal_ir()
    layer = ActionLayer(cfg=cfg_from_yaml, ctx=ctx_from_yaml, eval_runs=1)

    # Find an invalid action that is bound to be MASKED.
    # for minimal IR (e.g., linalg that needs tensor type)
    names = [a.scope_name for a in layer.registry.action_space]
    try:
        bad_idx = next(
            i for i, n in enumerate(names) if "linalg-fuse-elementwise-ops" in n
        )
    except StopIteration:
        pytest.skip(
            "No linalg fuse action in action space; adjust test to another masked action"
        )

    # make sure it is masked
    mask = layer.mask(ir)
    assert mask[bad_idx] == 0, "Expected the chosen action to be masked on this IR"

    # apply the masked action
    sr = layer.apply_action(ir, bad_idx)
    assert sr.term is False
    assert sr.result.status is Status.ERROR
    assert action_masked_out.check(sr.result.e_code)

    rwd = build_rewarder()
    setattr(rwd, "masked_penalty", 0.3)
    reward = rwd.get_reward(ctx_from_yaml, sr)
    assert np.allclose(
        reward, -0.3
    )  # FIXME(megan.kuo) masked-penalty bounds to be changed


@pytest.mark.e2e
def test_final_reward_log_speedup_success(cfg_from_yaml, ctx_from_yaml):
    """
    目標：終局成功時，reward = -log(lat/L0)（再加可選軟約束）。
    用假的 StepReturn 構造終局，避免依賴系統 benchmark。
    """
    rwd = build_rewarder(c_penalty=7.0)

    # baseline 取自 ctx
    L0, C0, B0 = ctx_from_yaml.get_baseline()
    assert L0 >= 0.0

    # Case1: lat == L0 -> reward ~ 0
    sr_equal = StepReturn(
        result=StepResult(
            new_ir="",
            status=Status.OK,
            metrics=Metrics(
                latency_ms=L0, compile_ms=0.8 * C0 if C0 else 1.0, size_kb=B0
            ),
            e_code=ErrorBits.W_NO_CHANGE,
            error_msg="",
        ),
        trun=False,
        term=True,
        info=Info(index=0, mask=None),
    )
    r0 = rwd.get_reward(ctx_from_yaml, sr_equal)
    assert np.allclose(r0, 0.0)

    # Case2: 2x speedup -> -log(0.5) > 0
    sr_fast = StepReturn(
        result=StepResult(
            new_ir="",
            status=Status.OK,
            metrics=Metrics(
                latency_ms=max(L0 * 0.5, 1e-6), compile_ms=0.8 * (C0 or 1.0), size_kb=B0
            ),
            e_code=ErrorBits.W_NO_CHANGE,
            error_msg="",
        ),
        trun=False,
        term=True,
        info=Info(index=0, mask=None),
    )
    r1 = rwd.get_reward(ctx_from_yaml, sr_fast)
    assert r1 > 0.0
    assert np.allclose(r1, -math.log(0.5))


@pytest.mark.e2e
def test_final_reward_failure_penalty(cfg_from_yaml, ctx_from_yaml):
    """
    目標：終局失敗（不可編譯/timeout）回 -C_penalty。
    """
    rwd = build_rewarder(c_penalty=9.0)

    sr_fail = StepReturn(
        result=StepResult(
            new_ir="",
            status=Status.ERROR,
            metrics=None,
            e_code=ErrorBits.W_NO_CHANGE | ErrorBits.E_TIMEDOUT,
            error_msg="timeout",
        ),
        trun=False,
        term=True,
        info=Info(index=0, mask=None),
    )
    rF = rwd.get_reward(ctx_from_yaml, sr_fail)
    assert np.allclose(rF, -rwd.final.c_penalty)


@pytest.mark.e2e
def test_truncate_auto_eval_path(cfg_from_yaml, ctx_from_yaml, monkeypatch):
    """
    Reach max step -> force eval (term and trun)
    """
    rwd = build_rewarder(c_penalty=10.0)

    # slower metrics : latency
    L0, C0, B0 = ctx_from_yaml.get_baseline()
    forced_metrics = Metrics(latency_ms=max(0.8 * L0, 1e-6), compile_ms=C0, size_kb=B0)

    sr_forced_term = StepReturn(
        result=StepResult(
            new_ir="",
            status=Status.OK,
            metrics=forced_metrics,
            e_code=ErrorBits.NONE,
            error_msg="",
        ),
        trun=True,
        term=True,
        info=Info(index=-1, mask=None),
    )

    # monkeypatch
    setattr(rwd, "eval_on_truncate_penalty", 0.3)

    # reward is discount by rwd.eval_on_truncate_penalty
    r_final = rwd.get_reward(ctx_from_yaml, sr_forced_term)
    expect = -math.log(0.8) - 0.3
    np.allclose(r_final, expect)
