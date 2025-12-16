# tests/test_view_runner.py
import os, pytest
from mlir_env.views.viewer import AnalysisIntent, pick_view
from mlir_env.views.view_runner import ViewRunner
from mlir_env.views.spec import ViewKind

GEMM = r"""
module {
  func.func @gemm(%A: tensor<4x4xf32>, %W: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<4x4xf32>
    %m = linalg.matmul ins(%A, %W : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %m : tensor<4x4xf32>
  }
}
"""


def test_affine_view_contains_affine_for(cfg_from_yaml):
    runner = ViewRunner(cfg_from_yaml)
    out = runner.view(GEMM, ViewKind.AFFINE)
    assert "affine.for" in out


def test_scf_view_contains_scf_for(cfg_from_yaml):
    runner = ViewRunner(cfg_from_yaml)
    out = runner.view(GEMM, ViewKind.SCF)
    assert "scf.for" in out


def test_memref_view_contains_scf_for(cfg_from_yaml):
    runner = ViewRunner(cfg_from_yaml)
    out = runner.view(GEMM, ViewKind.MEMREF)
    assert "memref" in out


def test_analysis_intent(cfg_from_yaml):
    runner = ViewRunner(cfg_from_yaml)
    pick_by_manual = runner.view(GEMM, ViewKind.MEMREF)

    intent = AnalysisIntent(
        need_polyhedral_access=False, need_loops=False, inspect_vector_ops=False
    )
    view_mode = pick_view(intent)
    pick_by_intent = runner.view(GEMM, view_mode)
    assert pick_by_manual == pick_by_intent
