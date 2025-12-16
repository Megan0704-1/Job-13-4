# tests/test_actions_space_and_registry.py
import numpy as np
from mlir_env.core.config import Config
from mlir_env.core.immutable import Phase
from mlir_env.actions.registry import ActionRegistry
from mlir_env.actions.space import get_action_space_default
from mlir_env.actions.action import Action


def make_registry(pass_limits=2, hash_window_len=3):
    cfg = Config(pass_limits=pass_limits, hash_window_len=hash_window_len)
    return ActionRegistry(cfg=cfg, action_space=get_action_space_default())


def phase_of(*bits):
    p = Phase(0)
    for b in bits:
        p |= b
    return p


# This function tests initialization of used_action_counts in registry
def test_action_space_contains_eval_and_counts_initialized():
    reg = make_registry()
    names = reg.get_all_action_names()
    assert any("env::EVAL" in n for n in names)
    # all counts start at 0
    for a in reg.action_space:
        assert reg.used_action_counts[a.scope_name] == 0


# this function tests the functionality of registry.mask and action.precond
# Note. registry.mask should never masked-out eval action.
def test_mask_includes_eval_and_respects_preconditions():
    reg = make_registry()
    # pick a phase that allows tensor+linalg passes but not llvm lowers
    p = phase_of(Phase.TENSOR, Phase.LINALG)
    mask = reg.mask_actions(p)
    assert mask.shape == (len(reg.action_space),)

    # eval is always 1
    eval_idx = [i for i, a in enumerate(reg.action_space) if a.is_eval()][0]
    assert mask[eval_idx] == 1

    # linalg-fuse-elementwise-ops should be legal
    fuse_idx = [
        i
        for i, a in enumerate(reg.action_space)
        if "linalg-fuse-elementwise-ops" in a.name
    ][0]
    assert mask[fuse_idx] == 1

    # a LLVM-only finalize should be masked at tensor stage
    fin_idx = [
        i for i, a in enumerate(reg.action_space) if a.name == "finalize-memref-to-llvm"
    ][0]
    assert mask[fin_idx] == 0


# This function tests the cap check during registry masking
# 1. set pass_limits in config to 1
# 2. apply the action once
# -> this action is expected be masked-out.
def test_caps_enforced_by_masking_attempt_counts():
    reg = make_registry(pass_limits=1)

    # find a non-eval action that is legal at tensor+linalg
    p = phase_of(Phase.TENSOR, Phase.LINALG)
    legal_idxs = reg.legal_indices(p)
    non_eval_idx = [i for i in legal_idxs if not reg.action_space[i].is_eval()][0]
    act = reg.action_space[non_eval_idx]

    # before use: legal
    mask0 = reg.mask_actions(p)
    assert mask0[non_eval_idx] == 1

    # simulate attempt -> add_entry
    reg.add_entry(act)

    # after one attempt, with cap=1: masked out
    mask1 = reg.mask_actions(p)
    assert mask1[non_eval_idx] == 0

    # eval still legal
    eval_idx = [i for i, a in enumerate(reg.action_space) if a.is_eval()][0]
    assert mask1[eval_idx] == 1


# This function tests pipeline command
def test_get_pipeline_command_builds_module_and_func_atoms_in_order():
    reg = make_registry()

    # add a module-level then func-level action
    mod = [a for a in reg.action_space if a.scope == "module" and not a.is_eval()][0]
    fun = [a for a in reg.action_space if a.scope == "func" and not a.is_eval()][0]
    reg.add_entry(mod)
    reg.add_entry(fun)
    reg.add_entry([a for a in reg.action_space if a.is_eval()][0])

    cmd = reg.get_pipeline_command()
    # order follows past_actions: module atom first, then func.func(...)
    assert cmd.startswith("builtin.module(")
    assert "func.func(" in cmd
    assert cmd.endswith(")")  # closes the module

    # sanity: eval not present
    assert "eval" not in cmd


# This function tests the noop_window in the registry
# 1. set window size (n)
# 2. record noop entries 2*n times
# -> registry.noop_window is expected to record at most n noops
def test_used_action_counts_init_and_increment():
    window_len = 8
    reg = make_registry(hash_window_len=window_len)

    ir_text = "hahaha"
    act = reg.action_space[0]

    for i in range(window_len * 2):
        reg.record_noop(ir_text, act)

    assert len(reg.noop_window) == window_len


# This function tests registry.noop_window functionality
# Only mask when IR does not changed recently (window_len)
def test_noop_window_masks_only_for_same_ir_and_action(monkeypatch):
    reg = make_registry(pass_limits=10, hash_window_len=4)
    action_id = 0

    ir1 = "builtin.module { func.func @main() {} }"
    ir2 = "builtin.module { func.func @other() {} }"
    mod_action = reg.action_space[action_id]  # cse, module scope

    # mask 1st time without any action
    # cse should not be masked-out.
    ir1_mask1 = reg.mask_actions(ir1)
    assert ir1_mask1[action_id] == 1

    # manual record noop: cse does not change ir1
    # in code, record_noop would be called by action_layer.apply automatically
    reg.record_noop(ir1, mod_action)

    # same IR, same action is expected to be masked-out
    ir1_mask2 = reg.mask_actions(ir1)
    assert ir1_mask2[action_id] == 0

    # different IR is not affected by noop_window
    ir2_mask1 = reg.mask_actions(ir2)
    assert ir2_mask1[action_id] == 1

    # noop_window would not be triggered by masking phase
    mask3 = reg.mask_actions(Phase.FUNC)
    assert mask3[action_id] == 1
