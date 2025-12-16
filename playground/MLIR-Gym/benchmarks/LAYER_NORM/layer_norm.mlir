#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> ()>
#map4 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @row_layernorm_tensor(%arg0: tensor<32768x32768xf32>, %arg1: tensor<32768xf32>, %arg2: tensor<32768xf32>) -> tensor<32768x32768xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 9.99999974E-6 : f32
    %cst_2 = arith.constant 3.05175781E-5 : f32
    %0 = tensor.empty() : tensor<32768xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32768xf32>) -> tensor<32768xf32>

    // row-wise add
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<32768x32768xf32>) outs(%1 : tensor<32768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = arith.addf %in, %out : f32
      linalg.yield %14 : f32
    } -> tensor<32768xf32>

    // calculate mean from add
    %3 = tensor.empty() : tensor<32768xf32>
    %4 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel"]} ins(%2, %cst_2 : tensor<32768xf32>, f32) outs(%3 : tensor<32768xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %14 = arith.mulf %in, %in_3 : f32
      linalg.yield %14 : f32
    } -> tensor<32768xf32>


    // sum of squared deviations: sum (x-mu) ** 2
    %5 = tensor.empty() : tensor<32768xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<32768xf32>) -> tensor<32768xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0, %4 : tensor<32768x32768xf32>, tensor<32768xf32>) outs(%6 : tensor<32768xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %14 = arith.subf %in, %in_3 : f32
      %15 = arith.mulf %14, %14 : f32
      %16 = arith.addf %out, %15 : f32
      linalg.yield %16 : f32
    } -> tensor<32768xf32>

    // variance
    %8 = tensor.empty() : tensor<32768xf32>
    %9 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel"]} ins(%7, %cst_2 : tensor<32768xf32>, f32) outs(%8 : tensor<32768xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %14 = arith.mulf %in, %in_3 : f32
      linalg.yield %14 : f32
    } -> tensor<32768xf32>

    // inverse std: 1 / sqrt(variance + epsilon)
    %10 = tensor.empty() : tensor<32768xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel"]} ins(%9, %cst_1 : tensor<32768xf32>, f32) outs(%10 : tensor<32768xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %14 = arith.addf %in, %in_3 : f32
      %15 = math.sqrt %14 : f32
      %16 = arith.divf %cst_0, %15 : f32
      linalg.yield %16 : f32
    } -> tensor<32768xf32>

    // Y = (X - mu) * inverse std * gamma + B
    %12 = tensor.empty() : tensor<32768x32768xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map4, #map4, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %4, %11, %arg1, %arg2 : tensor<32768x32768xf32>, tensor<32768xf32>, tensor<32768xf32>, tensor<32768xf32>, tensor<32768xf32>) outs(%12 : tensor<32768x32768xf32>) {
    ^bb0(%in: f32, %in_3: f32, %in_4: f32, %in_5: f32, %in_6: f32, %out: f32):
      %14 = arith.subf %in, %in_3 : f32
      %15 = arith.mulf %14, %in_4 : f32
      %16 = arith.mulf %15, %in_5 : f32
      %17 = arith.addf %16, %in_6 : f32
      linalg.yield %17 : f32
    } -> tensor<32768x32768xf32>
    return %13 : tensor<32768x32768xf32>
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<32768x32768xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32768x32768xf32>) -> tensor<32768x32768xf32>
    %2 = tensor.empty() : tensor<32768xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32768xf32>) -> tensor<32768xf32>
    %4 = tensor.empty() : tensor<32768xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<32768xf32>) -> tensor<32768xf32>
    %6 = call @row_layernorm_tensor(%1, %3, %5) : (tensor<32768x32768xf32>, tensor<32768xf32>, tensor<32768xf32>) -> tensor<32768x32768xf32>
    return
  }
}

