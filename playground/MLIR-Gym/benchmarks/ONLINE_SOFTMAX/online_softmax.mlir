#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @row_softmax_tensor_online(%arg0: tensor<32768x32768xf32>) -> tensor<32768x32768xf32> {
    %cst = arith.constant -3.40282347E+38 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<32768xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32768xf32>) -> tensor<32768xf32>
    %2 = tensor.empty() : tensor<32768xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<32768xf32>) -> tensor<32768xf32>
    %4:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<32768x32768xf32>) outs(%1, %3 : tensor<32768xf32>, tensor<32768xf32>) {
    ^bb0(%in: f32, %out: f32, %out_1: f32):
      %7 = arith.maximumf %in, %out : f32
      %8 = arith.subf %in, %7 : f32
      %9 = math.exp %8 : f32
      %10 = arith.subf %out, %7 : f32
      %11 = math.exp %10 : f32
      %12 = arith.mulf %out_1, %11 : f32
      %13 = arith.addf %12, %9 : f32
      linalg.yield %7, %13 : f32, f32
    } -> (tensor<32768xf32>, tensor<32768xf32>)
    %5 = tensor.empty() : tensor<32768x32768xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %4#0, %4#1 : tensor<32768x32768xf32>, tensor<32768xf32>, tensor<32768xf32>) outs(%5 : tensor<32768x32768xf32>) {
    ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32):
      %7 = arith.subf %in, %in_1 : f32
      %8 = math.exp %7 : f32
      %9 = arith.divf %8, %in_2 : f32
      linalg.yield %9 : f32
    } -> tensor<32768x32768xf32>
    return %6 : tensor<32768x32768xf32>
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<32768x32768xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32768x32768xf32>) -> tensor<32768x32768xf32>
    %2 = call @row_softmax_tensor_online(%1) : (tensor<32768x32768xf32>) -> tensor<32768x32768xf32>
    return
  }
}

