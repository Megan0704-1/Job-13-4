#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @col_reduce_max(%arg0: tensor<102400x102400xf32>) -> tensor<102400xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<102400xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<102400xf32>) -> tensor<102400xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<102400x102400xf32>) outs(%1 : tensor<102400xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.maximumf %out, %in : f32
      linalg.yield %3 : f32
    } -> tensor<102400xf32>
    return %2 : tensor<102400xf32>
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<102400x102400xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<102400x102400xf32>) -> tensor<102400x102400xf32>
    %2 = call @col_reduce_max(%1) : (tensor<102400x102400xf32>) -> tensor<102400xf32>
    return
  }
}

