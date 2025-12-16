# mlir_env/core/context.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
from .config import Config
import yaml

import os
import sys
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class EpisodeContext:
    cfg: Config
    backend: str
    mlir_path: str
    pipeline_path: str
    mlir_id: str
    mlir_content: str
    pipeline_content: str
    L0: float  # program baseline latency
    C0: float  # program baseline compile time
    B0: float  # program baseline byte size
    max_steps: int
    steps_so_far: int = 0
    view_option: str = "affine" # "scf"

    @staticmethod
    def from_yaml(cfg: Config, path: str) -> EpisodeContext:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}

        # get mlir path
        mlir_path = data["mlir_path"]

        # check if path exist
        if not os.path.exists(os.path.expandvars(mlir_path)):
            logging.warning(f"Unable to locate your mlir file at {mlir_path}")
            sys.exit(2)

        # load mlir content
        mlir_content = Path(mlir_path).read_text()

        # get pipeline path
        pipeline_path = data["pipeline_path"]

        # check if path exist
        if not os.path.exists(os.path.expandvars(pipeline_path)):
            logging.warning(
                f"Unable to locate your baseline lowering pipeline file at {pipeline_path}"
            )
            sys.exit(2)

        # load basic lowering content
        pipeline_content = Path(pipeline_path).read_text()

        return EpisodeContext(
            cfg=cfg,
            backend=data.get("backend", "cpu"),
            mlir_path=mlir_path,
            pipeline_path=pipeline_path,
            mlir_id=data.get("mlir_id", Path(mlir_path).name),
            mlir_content=mlir_content,
            pipeline_content=pipeline_content,
            L0=float(data.get("L0", 0.0)),  # ms
            C0=float(data.get("C0", 0.0)),  # ms
            B0=float(data.get("B0", 0.0)),  # kB
            max_steps=int(data.get("max_steps", 20)),
            view_option=data.get("view_option", "affine")
        )

    def reset_for(self, new_path: str | None = None) -> EpisodeContext:
        """Return a new context"""
        if new_path:
            return EpisodeContext.from_yaml(self.cfg, new_path)
        return EpisodeContext(
            cfg=self.cfg,
            backend=self.backend,
            mlir_path=self.mlir_path,
            pipeline_path=self.pipeline_path,
            mlir_id=self.mlir_id,
            mlir_content=Path(self.mlir_path).read_text(),
            pipeline_content=Path(self.pipeline_path).read_text(),
            L0=self.L0,
            C0=self.C0,
            B0=self.B0,
            max_steps=self.max_steps,
            steps_so_far=0,
            view_option=self.view_option
        )

    def get_baseline(self) -> tuple[float, float, float]:
        return (self.L0, self.C0, self.B0)
