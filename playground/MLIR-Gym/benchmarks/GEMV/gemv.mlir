#map = affine_map<(d0) -> (d0)>
module {
  func.func @gemv_bias_relu(%arg0: tensor<8192x16384xf32>, %arg1: tensor<16384xf32>, %arg2: tensor<8192xf32>) -> tensor<8192xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8192xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8192xf32>) -> tensor<8192xf32>
    %2 = linalg.matvec ins(%arg0, %arg1 : tensor<8192x16384xf32>, tensor<16384xf32>) outs(%1 : tensor<8192xf32>) -> tensor<8192xf32>
    %3 = tensor.empty() : tensor<8192xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8192xf32>) -> tensor<8192xf32>

    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%2, %arg2 : tensor<8192xf32>, tensor<8192xf32>) outs(%4 : tensor<8192xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %9 = arith.addf %in, %in_0 : f32
      linalg.yield %9 : f32
    } -> tensor<8192xf32>

    %6 = tensor.empty() : tensor<8192xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<8192xf32>) -> tensor<8192xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%5 : tensor<8192xf32>) outs(%7 : tensor<8192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.maximumf %in, %cst : f32
      linalg.yield %9 : f32
    } -> tensor<8192xf32>
    return %8 : tensor<8192xf32>
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 3.14 : f32
    %0 = tensor.empty() : tensor<8192x16384xf32>
    %1 = tensor.empty() : tensor<16384xf32>
    %2 = tensor.empty() : tensor<8192xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%1 : tensor<16384xf32>) -> tensor<16384xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%2 : tensor<8192xf32>) -> tensor<8192xf32>
    %6 = call @gemv_bias_relu(%3, %4, %5) : (tensor<8192x16384xf32>, tensor<16384xf32>, tensor<8192xf32>) -> tensor<8192xf32>
    return
  }
}

