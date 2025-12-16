import pytest
from mlir_env.observations.classes.Graph.immutable import GraphParams
from mlir_env.observations.classes.Graph.extractor import GraphExtractor

IR = r"""
module {
  func.func @f(%A: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c0 = arith.constant 0.0 : f32
    %e  = tensor.empty() : tensor<4x4xf32>
    %i0 = linalg.fill ins(%c0 : f32) outs(%e : tensor<4x4xf32>) -> tensor<4x4xf32>
    %y  = linalg.generic
      { indexing_maps = [ affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)> ],
        iterator_types = ["parallel","parallel"] }
      ins(%i0 : tensor<4x4xf32>) outs(%e : tensor<4x4xf32>) {
      ^bb0(%in:f32, %out:f32):
        %0 = arith.maximumf %in, %c0 : f32
        linalg.yield %0 : f32
    } -> tensor<4x4xf32>
    return %y : tensor<4x4xf32>
  }
}
"""

def test_op_and_dialect_ids_are_stable_across_instances():
    p = GraphParams(max_nodes=64, feature_dim=7, emit_ids=True)

    ex1 = GraphExtractor(p)
    feats1, edges1, type_ids1 = ex1.extract(IR)

    ex2 = GraphExtractor(p)
    feats2, edges2, type_ids2 = ex2.extract(IR)

    assert type_ids1[:len(type_ids2)] == type_ids2[:len(type_ids1)]
    assert len(feats1) == len(type_ids1)
1
