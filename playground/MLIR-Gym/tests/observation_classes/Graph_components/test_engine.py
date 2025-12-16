import numpy as np
from mlir_env.observations.classes.Graph.immutable import GraphParams, GraphOutParams
from mlir_env.observations.classes.Graph.engine import GraphEngine

IR = r"""
module {
  func.func @f(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c0 = arith.constant 0.0 : f32
    %e  = tensor.empty() : tensor<4x4xf32>
    %i0 = linalg.fill ins(%c0 : f32) outs(%e : tensor<4x4xf32>) -> tensor<4x4xf32>
    %y  = linalg.matmul ins(%A,%B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%i0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %y : tensor<4x4xf32>
  }
}
"""

def test_engine_outputs_have_expected_fields_and_shapes(cfg_from_yaml, ctx_from_yaml):
    p = GraphParams(max_nodes=32, feature_dim=7, emit_ids=True)
    eng = GraphEngine(cfg=cfg_from_yaml, ctx=ctx_from_yaml, params=p)

    obs_dict = eng.observe(IR)
    obs = GraphOutParams(**obs_dict)

    assert obs.node_features.shape == (32, 7)
    assert obs.adj.shape == (32, 32)
    assert obs.node_mask.dtype == np.bool_
    assert obs.op_type_ids.shape == (32,)
    assert int(obs.adj.sum()) > 0, "Should have at least 1 edge exist"

