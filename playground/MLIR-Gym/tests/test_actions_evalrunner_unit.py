import os
import contextlib
from types import SimpleNamespace

from mlir_env.core.immutable import Status, ErrorBits
from mlir_env.actions.runner import EvalRunner
from mlir_env.utils.common_utils import *


class DummyCfg:
    cpu_core = 0
    compile_option = "affine"
    eval_timeout_s = 12
    eval_runs = 5
    llvm_build_dir = os.environ.get("LLVM_BUILD_DIR")
    benchmark_sh = os.environ.get("MLIR_GYM_ROOT") + "/benchmarks/benchmark.sh"


class DummyCtx:
    def __init__(self):
        self.cfg = DummyCfg()
        self.pipeline_content = "builtin.module(cse,func.func(canonicalize))"


class FakeExecResult:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = False


class FakeExecutor:
    def __init__(self, result):
        self._result = result
        self.last_spec = None

    def run(self, spec, input_bytes=None):
        self.last_spec = spec
        return self._result


@contextlib.contextmanager
def fake_mktemp(_content):
    yield "/tmp/fake.mlir"


# This function tests the benchmark flag from EvalRunner
def test_evalrunner_flag_and_build_spec(monkeypatch):
    ctx = DummyCtx()
    ev = EvalRunner(ctx, executor=FakeExecutor(FakeExecResult()), eval_runs=3)

    # check commands that is suppose to be run by the executor
    s = ev.get_benchmark_flag("/tmp/f.mlir")
    assert "--mlir /tmp/f.mlir" in s
    assert f"--build {ctx.cfg.llvm_build_dir}" in s
    assert "--entry main" in s
    assert "--runs 3" in s
    assert "--pipeline --pass-pipeline=builtin.module(cse,func.func(canonicalize))" in s

    # check spec: [bin, flags]
    spec = ev.build_spec("/tmp/f.mlir")
    argv = spec.argv
    assert argv[0] == ctx.cfg.benchmark_sh
    assert "--mlir" in argv and "/tmp/f.mlir" in argv


# This function tests apply method in EvalRunner.
def test_evalrunner_apply_success_and_failure(monkeypatch):
    ctx = DummyCtx()

    # monkeypatch mask benchmark_utils to return fix dict
    import mlir_env.actions.runner as runner_mod

    monkeypatch.setattr(
        runner_mod, "parse_baseline_metrics", lambda s: {"ms": 1.23, "bytes": 456}
    )
    monkeypatch.setattr(runner_mod, "mktemp", fake_mktemp)

    # success: returncode == 0
    ok_exec = FakeExecutor(FakeExecResult(returncode=0, stdout=b"FAKE METRICS LOG"))
    ev_ok = EvalRunner(ctx, executor=ok_exec, eval_runs=2)
    s_ok = ev_ok.apply("dummy_ir")
    assert s_ok.status is Status.OK
    assert s_ok.metrics == {"ms": 1.23, "bytes": 456}
    assert ir_has_changed.check(s_ok.e_code)
    assert s_ok.error_msg == ""

    # failed: returncode != 0
    bad_exec = FakeExecutor(FakeExecResult(returncode=1, stdout=b"", stderr=b"oops"))
    ev_bad = EvalRunner(ctx, executor=bad_exec, eval_runs=2)
    s_bad = ev_bad.apply("dummy_ir")
    assert s_bad.status is Status.ERROR
    assert s_bad.metrics is None
    assert action_run_failed.check(s_bad.e_code)
    assert s_bad.error_msg == "oops"
