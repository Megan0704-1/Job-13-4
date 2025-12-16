# mlir_env/core/config.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import yaml, hashlib

import os
import sys
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# immutable episode-wise
@dataclass(frozen=True, slots=True)
class Config:
    backends: list[str] = field(default_factory=lambda: ["cpu"])
    seed: int = 42

    dim_feature: int = 256
    dim_action: int = 64
    k_actions: int = 8
    hash_window_len: int = 4
    pass_limits: int = 3

    # reward weights
    lambda_L: float = 1.0
    lambda_C: float = 0.0
    lambda_B: float = 0.0

    mlir_opt_bin: str = "mlir-opt"
    benchmark_sh: str = "benchmark.sh"
    llvm_build_dir: str = "build"

    apply_timeout_s: int = 30
    eval_timeout_s: int | None = None

    eval_runs: int = 1
    compile_option: str = "aot"
    num_cores: int = 32
    cpu_core: int = 0

    viewer_cache_enabled: bool = True
    viewer_cache_cap: int = 128

    _hash_id: str | None = field(default=None, init=False, repr=False, compare=False)

    @staticmethod
    def get_environ_vars() -> Dict[str, str]:
        # get llvm build directory
        llvm_build_dir = os.environ.get("LLVM_BUILD_DIR")

        # check if folder exist
        if not os.path.isdir(llvm_build_dir):
            logging.warning(
                f"Unable to locate llvm-project/build directory at {llvm_build_dir}"
            )
            sys.exit(2)

        # get project root directory
        mlir_gym_root = os.environ.get("MLIR_GYM_ROOT")

        # check if folder exist
        if not os.path.isdir(mlir_gym_root):
            logging.warning(
                f"Unable to locate project root directory at {mlir_gym_root}"
            )
            sys.exit(2)

        cpu_core = int(os.environ.get("CPU_CORE", 0))

        return {"llvm_build_dir": llvm_build_dir, "mlir_gym_root": mlir_gym_root, "cpu_core": cpu_core}

    @staticmethod
    def from_yaml(path: str) -> Config:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}

        os_vars = Config.get_environ_vars()

        # check if paths exist
        mlir_opt_bin = Path(os_vars.get("llvm_build_dir")) / "bin/mlir-opt"
        benchmark_sh = Path(os_vars.get("mlir_gym_root")) / "benchmarks/benchmark.sh"
        llvm_build_dir = Path(os_vars.get("llvm_build_dir"))

        if not (os.path.isfile(mlir_opt_bin) and os.access(mlir_opt_bin, os.X_OK)):
            logging.warning(f"Unable to locate mlir-opt at {mlir_opt_bin}")
            sys.exit(2)

        if not os.path.isfile(benchmark_sh):
            logging.warning(f"Unable to locate benchmark.sh at {benchmark_sh}")
            sys.exit(2)

        data.update(
            {
                "mlir_opt_bin": str(mlir_opt_bin),
                "benchmark_sh": str(benchmark_sh),
                "llvm_build_dir": str(llvm_build_dir),
                "cpu_core": os_vars.get("cpu_core") % int(data.get("num_cores", 32))
            }
        )

        return Config(**data)

    def hash(self) -> str:
        if self._hash_id is None:
            s = repr(asdict(self) | {"_hash_id": None})
            h = hashlib.sha256(s.encode()).hexdigest()[:8]
            object.__setattr__(self, "_hash_id", h)
        return self._hash_id  # type: ignore[return-value]
