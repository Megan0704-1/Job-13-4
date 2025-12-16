#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> ()>
#map4 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @rmsnorm_tensor(%arg0: tensor<32768x32768xf32>, %arg1: tensor<32768xf32>) -> tensor<32768x32768xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 9.99999974E-6 : f32
    // 1 / N
    %cst_2 = arith.constant 3.051757E-5 : f32
    %0 = tensor.empty() : tensor<32768xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32768xf32>) -> tensor<32768xf32>

    // sum of X^2
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<32768x32768xf32>) outs(%1 : tensor<32768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.mulf %in, %in : f32
      %10 = arith.addf %out, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<32768xf32>

    // 1/N * sum of X^2
    %3 = tensor.empty() : tensor<32768xf32>
    %4 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel"]} ins(%2, %cst_2 : tensor<32768xf32>, f32) outs(%3 : tensor<32768xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %9 = arith.mulf %in, %in_3 : f32
      linalg.yield %9 : f32
    } -> tensor<32768xf32>

    // term = 1 / [ sqrt ( 1/N * sum of X^2 + ep ) ]
    %5 = tensor.empty() : tensor<32768xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel"]} ins(%4, %cst_1 : tensor<32768xf32>, f32) outs(%5 : tensor<32768xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %9 = arith.addf %in, %in_3 : f32
      %10 = math.sqrt %9 : f32
      %11 = arith.divf %cst_0, %10 : f32
      linalg.yield %11 : f32
    } -> tensor<32768xf32>

    // X * term * gamma
    %7 = tensor.empty() : tensor<32768x32768xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map4, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %6, %arg1 : tensor<32768x32768xf32>, tensor<32768xf32>, tensor<32768xf32>) outs(%7 : tensor<32768x32768xf32>) {
    ^bb0(%in: f32, %in_3: f32, %in_4: f32, %out: f32):
      %9 = arith.mulf %in, %in_3 : f32
      %10 = arith.mulf %9, %in_4 : f32
      linalg.yield %10 : f32
    } -> tensor<32768x32768xf32>
    return %8 : tensor<32768x32768xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<32768x32768xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32768x32768xf32>) -> tensor<32768x32768xf32>
    %2 = tensor.empty() : tensor<32768xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32768xf32>) -> tensor<32768xf32>
    %4 = call @rmsnorm_tensor(%1, %3) : (tensor<32768x32768xf32>, tensor<32768xf32>) -> tensor<32768x32768xf32>
    return
  }
}

