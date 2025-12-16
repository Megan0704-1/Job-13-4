# actions/executor.py
# This file contains the helper function for launching mlir-opt flag subprocess

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import subprocess, time, os


@dataclass(frozen=True, slots=True)
class CommandSpec:
    argv: list[str]
    timeout_s: int | None = None
    cwd: str | None = None
    env: dict | None = None


@dataclass(frozen=True, slots=True)
class ExecResult:
    returncode: int = field(default_factory=lambda: -1)
    stdout: bytes = field(default_factory=lambda: b"")
    stderr: bytes = field(default_factory=lambda: b"")
    wall_ms: float = 0.0
    timed_out: bool = False


class SubprocessExecutor:
    def run(self, spec: CommandSpec, input_bytes: bytes) -> ExecResult:
        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                spec.argv,
                input=input_bytes,
                capture_output=True,
                check=False,
                timeout=spec.timeout_s,
                cwd=spec.cwd,
                env=(spec.env or os.environ.copy()),
            )

            return ExecResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                wall_ms=(time.perf_counter() - t0) * 1000.0,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                wall_ms=(time.perf_counter() - t0) * 1000.0, timed_out=True
            )
