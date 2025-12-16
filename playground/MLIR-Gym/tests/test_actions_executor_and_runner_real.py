# tests/test_actions_executor_and_runner_real.py

import os
import shutil
import yaml
import pytest
from pathlib import Path

from mlir_env.core.config import Config
from mlir_env.core.context import EpisodeContext
from mlir_env.core.immutable import Status, ErrorBits

from mlir_env.actions.registry import ActionRegistry
from mlir_env.actions.runner import PassRunner

from mlir_env.utils.ir_utils import detect_phase_from_ir
from mlir_env.utils.common_utils import *


def test_config_and_context_load(cfg_from_yaml, ctx_from_yaml):
    assert isinstance(cfg_from_yaml.mlir_opt_bin, str)
    assert Path(ctx_from_yaml.mlir_path).exists()
    assert len(ctx_from_yaml.mlir_content) > 0
    assert ctx_from_yaml.get_baseline() == (
        ctx_from_yaml.L0,
        ctx_from_yaml.C0,
        ctx_from_yaml.B0,
    )


def test_action_mask_and_eval_always_legal(cfg_from_yaml, ctx_from_yaml):
    reg = ActionRegistry(cfg=cfg_from_yaml)
    phase = detect_phase_from_ir(ctx_from_yaml.mlir_content)
    mask = reg.mask_actions(phase)

    # eval is always legal
    eval_idx = [i for i, a in enumerate(reg.action_space) if a.is_eval()][0]
    assert mask[eval_idx] == 1

    # at least one pass is legal at the first phase of gemm.mlir
    legal_idxs = (mask.astype(bool)).nonzero()[0]
    assert len(legal_idxs) > 0


def test_runner_apply_canonicalize_real_mlir_opt(cfg_from_yaml, ctx_from_yaml):
    binname = cfg_from_yaml.mlir_opt_bin
    if shutil.which(binname) is None:
        pytest.skip(f"'{binname}' not found in PATH; skipping real mlir-opt run")

    reg = ActionRegistry(cfg=cfg_from_yaml)
    runner = PassRunner(cfg_from_yaml)

    phase = detect_phase_from_ir(ctx_from_yaml.mlir_content)
    mask = reg.mask_actions(phase)

    idx = reg.legal_indices(phase)[0]
    assert mask[idx] == 1

    action = reg.action_space[idx]
    step = runner.apply(ctx_from_yaml.mlir_content, action)

    timed_out = action_timed_out.check(step.e_code)
    no_changed = ir_not_changed.check(step.e_code)

    assert not timed_out
    assert not no_changed
    assert step.status is Status.OK
    assert isinstance(step.new_ir, str) and len(step.new_ir) > 0
