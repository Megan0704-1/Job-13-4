# actions/runner.py
from __future__ import annotations
import numpy as np
from typing import List
from pathlib import Path
from mlir_env.core.config import Config
from mlir_env.core.context import EpisodeContext
from mlir_env.core.immutable import StepResult, Status, ErrorBits
from mlir_env.actions.action import Action
from mlir_env.actions.executor import CommandSpec, SubprocessExecutor
from mlir_env.utils.benchmark_utils import parse_baseline_metrics, mktemp
from mlir_env.utils.common_utils import *


class PassRunner:
    """
    A runner class responsible for executing ONE action on a state
    Returns StepResult with state change information
    """

    def __init__(self, cfg: Config, executor: SubprocessExecutor | None = None):
        self.cfg = cfg
        self.exec = executor or SubprocessExecutor()

    def build_spec(self, action: Action) -> CommandSpec:
        argv = [self.cfg.mlir_opt_bin, action.to_pipeline_flag()]
        t = self.cfg.apply_timeout_s
        timeout = t if (t and t > 0) else None
        return CommandSpec(argv=argv, timeout_s=timeout)

    def apply(self, ir_text: str, action: Action) -> StepResult:
        """Apply an action to the IR"""
        if action.is_eval():
            raise ValueError("eval should be handled by Env")
        spec = self.build_spec(action)
        result = self.exec.run(spec, ir_text.encode("utf-8"))

        # check for timeout
        if result.timed_out:
            e_code = ErrorBits.W_NO_CHANGE | ErrorBits.E_TIMEDOUT
            return StepResult(
                new_ir=ir_text,
                status=Status.ERROR,
                metrics=None,
                e_code=e_code,
                error_msg="run single pass timeout, no IR change",
            )

        # check return code
        success = result.returncode == 0
        out = result.stdout.decode("utf-8", errors="replace")
        err_msg = (result.stderr.decode("utf-8", errors="replace")) or "failed"
        if not success:
            e_code = ErrorBits.W_NO_CHANGE | ErrorBits.E_FAILED
            return StepResult(
                new_ir=ir_text,
                status=Status.ERROR,
                metrics=None,
                e_code=e_code,
                error_msg=err_msg,
            )

        # success
        e_code = ErrorBits.NONE
        if out == ir_text:
            e_code |= ErrorBits.W_NO_CHANGE

        return StepResult(
            new_ir=out, status=Status.OK, metrics=None, e_code=e_code, error_msg=""
        )


class EvalRunner:
    """
    A runner class responsible for running EVAL action
    Includes method for lowering (by pipeline_content), for measurement
    Returns a StepResult with Metrics.
    """

    def __init__(
        self,
        ctx: EpisodeContext,
        executor: SubprocessExecutor | None = None,
        eval_runs: int = 1,
    ):
        self.ctx = ctx
        self.exec = executor or SubprocessExecutor()
        self.eval_timeout_s = self.ctx.cfg.eval_timeout_s
        self.eval_runs = eval_runs if eval_runs else self.ctx.cfg.eval_runs
        self.backend = "aot"  # or "jit"
        self.cpu_core = self.ctx.cfg.cpu_core
        self.compile_option=self.ctx.cfg.compile_option

    def get_benchmark_flag(self, new_ir_path: str) -> str:
        pipeline = f"--pass-pipeline={self.ctx.pipeline_content}"
        return f"--mlir {new_ir_path} --build {self.ctx.cfg.llvm_build_dir} --entry main --runs {self.eval_runs} --pipeline {pipeline} --backend {self.compile_option} --cpu {self.cpu_core} --json"

    def build_spec(self, new_ir_path: str) -> CommandSpec:
        argv = [
            self.ctx.cfg.benchmark_sh,
            *(self.get_benchmark_flag(new_ir_path).split()),
        ]
        return CommandSpec(argv=argv, timeout_s=self.eval_timeout_s)

    def apply(self, ir_text: str) -> StepResult:
        """Apply lowering and jit evalutation on the ir"""
        with mktemp(ir_text) as ir_path:
            spec = self.build_spec(ir_path)
            result = self.exec.run(spec, input_bytes=b"")
            print(f"execution path: {ir_path}")

        # check return code
        success = result.returncode == 0
        err_msg = (result.stderr.decode("utf-8", errors="replace")) or "eval failed"
        if not success:
            e_code = ErrorBits.W_NO_CHANGE | ErrorBits.E_EVAL_FAILED
            return StepResult(
                new_ir=ir_text,
                status=Status.ERROR,
                metrics=None,
                e_code=e_code,
                error_msg=err_msg,
            )

        # success
        out = result.stdout.decode("utf-8", errors="replace")
        metrics = parse_baseline_metrics(out)
        return StepResult(
            new_ir=ir_text,
            status=Status.OK,
            metrics=metrics,
            e_code=ErrorBits.NONE,
            error_msg="",
        )
