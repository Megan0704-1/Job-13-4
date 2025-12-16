# tests/test_actions_runner.py

from mlir_env.core.config import Config
from mlir_env.actions.action import Action
from mlir_env.core.immutable import Requirement, Status, ErrorBits
from mlir_env.actions.runner import PassRunner
from mlir_env.actions.executor import ExecResult
from mlir_env.utils.common_utils import *

DUMMY_IR = "module { func.func @f() { return } }"
CHANGED_IR = "module { // changed\n func.func @f() { return } }"


def test_runner_apply_success(monkeypatch):
    cfg = Config()
    r = PassRunner(cfg)
    a = Action("canonicalize", "func", Requirement())

    def fake_run(spec, input_bytes):
        assert "--pass-pipeline=builtin.module(func.func(canonicalize))" in spec.argv[1]
        return ExecResult(
            returncode=0,
            stdout=CHANGED_IR.encode(),
            stderr=b"",
            wall_ms=0.3,
            timed_out=False,
        )

    monkeypatch.setattr(r.exec, "run", fake_run)

    res = r.apply(DUMMY_IR, a)
    no_changed = ir_not_changed.check(res.e_code)
    timed_out = action_timed_out.check(res.e_code)
    assert res.status and (not no_changed) and (not timed_out)
    assert res.new_ir == CHANGED_IR


def test_runner_apply_verify_fail(monkeypatch):
    cfg = Config()
    r = PassRunner(cfg)
    a = Action("convert-vector-to-scf", "func", Requirement())

    def fake_run(spec, input_bytes):
        return ExecResult(
            returncode=1,
            stdout=b"",
            stderr=b"verify error",
            wall_ms=1.0,
            timed_out=False,
        )

    monkeypatch.setattr(r.exec, "run", fake_run)

    res = r.apply(DUMMY_IR, a)

    no_changed = ir_not_changed.check(res.e_code)
    run_failed = action_run_failed.check(res.e_code)
    assert (
        (res.status == Status.ERROR)
        and no_changed
        and run_failed
        and "verify" in (res.error_msg or "")
    )


def test_runner_apply_timeout(monkeypatch):
    cfg = Config()
    r = PassRunner(cfg)
    a = Action("cse", "func", Requirement())

    def fake_run(spec, input_bytes):
        return ExecResult(
            returncode=-1, stdout=b"", stderr=b"", wall_ms=5.0, timed_out=True
        )

    monkeypatch.setattr(r.exec, "run", fake_run)

    res = r.apply(DUMMY_IR, a)
    no_changed = ir_not_changed.check(res.e_code)
    timed_out = action_timed_out.check(res.e_code)
    assert (res.status is Status.ERROR) and no_changed and timed_out
