# tests/test_context.py

from mlir_env.core.config import Config
from mlir_env.core.context import EpisodeContext
import pytest


def test_context_from_yaml(globalVar):
    cfg = Config.from_yaml(globalVar.get("config.yaml"))
    ctx = EpisodeContext.from_yaml(cfg, globalVar.get("context.yaml"))

    assert ctx.cfg is cfg
    assert ctx.backend == "cpu"
    assert ctx.mlir_id == "gemm"
    assert ctx.mlir_content != None
    assert ctx.pipeline_content != None
    assert ctx.L0 == pytest.approx(34530.0)
    assert ctx.C0 == pytest.approx(10.0)
    assert ctx.B0 == pytest.approx(1.92)
    assert ctx.max_steps == 20
    assert ctx.steps_so_far == 0


def test_context_reset_for(globalVar):
    cfg = Config.from_yaml(globalVar.get("config.yaml"))
    ctx = EpisodeContext.from_yaml(cfg, globalVar.get("context.yaml"))

    a = ctx.reset_for()
    b = ctx.reset_for(globalVar.get("context.yaml"))

    assert a == b
    assert a.cfg.hash() == b.cfg.hash()
