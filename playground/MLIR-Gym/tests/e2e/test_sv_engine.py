# tests/e2e/test_sv_engine.py

import textwrap
import yaml
from gymnasium import spaces

from mlir_env.observations.classes.SV.engine import SVEngine


def partial_gemm():
    return textwrap.dedent(
        """
    module {
      func.func @gemm(%arg0: tensor<2048x2048xf32>, %arg1: tensor<2048x2048xf32>, %arg2: tensor<2048xf32>) {
        %c1 = arith.constant 1 : index
        %c2048 = arith.constant 2048 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %alloc = memref.alloc() {alignment = 64 : i64} : memref<2048x2048xf32>
        scf.for %arg3 = %c0 to %c2048 step %c1 {
          scf.for %arg4 = %c0 to %c2048 step %c1 {
            memref.store %cst, %alloc[%arg3, %arg4] : memref<2048x2048xf32>
          }
        }
        scf.for %arg3 = %c0 to %c2048 step %c1 {
          scf.for %arg4 = %c0 to %c2048 step %c1 {
            scf.for %arg5 = %c0 to %c2048 step %c1 {
              %extracted = tensor.extract %arg0[%arg3, %arg5] : tensor<2048x2048xf32>
              %extracted_1 = tensor.extract %arg1[%arg5, %arg4] : tensor<2048x2048xf32>
              %1 = memref.load %alloc[%arg3, %arg4] : memref<2048x2048xf32>
              %2 = arith.mulf %extracted, %extracted_1 : f32
              %3 = arith.addf %1, %2 : f32
              memref.store %3, %alloc[%arg3, %arg4] : memref<2048x2048xf32>
            }
          }
        }
        return
    }
   }"""
    )


def test_sv_engine_observation_contains(tmp_path, globalVar, cfg_from_yaml, ctx_from_yaml):
    # write detailed state YAML
    yml = globalVar.get("state_sv_config.yaml")

    # construct engine from YAML
    engine = SVEngine.from_yaml(cfg_from_yaml, ctx_from_yaml, str(yml))
    space = engine.build_space()
    assert isinstance(space, spaces.Dict)

    obs = engine.observe(partial_gemm())
    assert space.contains(obs)
    assert obs["loop_mask"].sum() == 4
    assert (obs["loop_feats"][:, 0] != 0).any()
