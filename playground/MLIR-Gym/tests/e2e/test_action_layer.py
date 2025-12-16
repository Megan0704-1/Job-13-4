# tests/e2e/test_action_layer.py
import os
import shutil
import tempfile
import textwrap

import pytest

from mlir_env.core.config import Config
from mlir_env.core.context import EpisodeContext
from mlir_env.core.immutable import Status, ErrorBits
from mlir_env.actions.layer import ActionLayer
from mlir_env.utils.common_utils import *

# --------- helpers ---------


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


# --------- tests ---------


@pytest.mark.e2e
def test_passrunner_apply_canonicalize_real(cfg_from_yaml, ctx_from_yaml):
    """
    E2E：用真實 mlir-opt 跑 canonicalize，一次或兩次。
    若第二次成為 NO-OP，確認 mask 會遮掉；
    否則至少確認不崩
    """
    ir = minimal_ir()
    cfg_from_yaml
    ctx_from_yaml
    layer = ActionLayer(cfg=cfg_from_yaml, ctx=ctx_from_yaml, eval_runs=1)

    # find canon index
    names = [a.scope_name for a in layer.registry.action_space]
    try:
        idx = next(i for i, n in enumerate(names) if n.startswith("func::canonicalize"))
    except StopIteration:
        pytest.fail("canonicalize not found in action space")

    # 1st time action
    sr1 = layer.apply_action(ir, idx)
    assert sr1.result.status is Status.OK, f"first pass failed: {sr1.result.error}"
    assert sr1.term is False

    # update IR
    ir2 = sr1.result.new_ir

    # 2nd time action -> noop
    sr2 = layer.apply_action(ir2, idx)
    assert sr2.result.status is Status.OK, f"second pass failed: {sr2.result.error}"

    # Expect: mask-out same IR-action pair
    if ir_not_changed.check(sr2.result.e_code):
        mask = layer.mask(sr2.result.new_ir)
        assert mask[idx] == 0, "NO-OP should be masked for this (ir_hash, action)"


@pytest.mark.e2e
def test_eval_action_if_available(cfg_from_yaml, ctx_from_yaml):
    """
    可選：若 benchmark.sh 存在就測 EVAL。
    只驗證能終局，不檢查數值；reward 之後在 reward-change 再驗。
    """
    ir = minimal_ir()
    layer = ActionLayer(cfg=cfg_from_yaml, ctx=ctx_from_yaml, eval_runs=1)

    # 找 EVAL 動作
    try:
        eval_idx = next(
            i for i, a in enumerate(layer.registry.action_space) if a.is_eval()
        )
    except StopIteration:
        pytest.skip("No EVAL action in action space")

    sr = layer.apply_action(ir, eval_idx)
    # 能結束且有 metrics（ok 不一定為真；但 step 要能走完）
    assert sr.term is True
    # metrics 可能因樣本 IR 不可跑而為 None；這裏只驗流程，不驗數值
