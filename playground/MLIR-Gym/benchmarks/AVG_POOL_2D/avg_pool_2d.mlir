#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @avgpool2d_nhwc_valid(%arg0: tensor<1x1024x1024x16384xf32>) -> tensor<1x512x512x16384xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.500000e-01 : f32
    %0 = tensor.empty() : tensor<1x512x512x16384xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x512x512x16384xf32>) -> tensor<1x512x512x16384xf32>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    %2 = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %cst_1 : tensor<1x1024x1024x16384xf32>, tensor<2x2xf32>) outs(%1 : tensor<1x512x512x16384xf32>) -> tensor<1x512x512x16384xf32>
    %3 = tensor.empty() : tensor<1x512x512x16384xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1x512x512x16384xf32>) outs(%3 : tensor<1x512x512x16384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.mulf %in, %cst_0 : f32
      linalg.yield %5 : f32
    } -> tensor<1x512x512x16384xf32>
    return %4 : tensor<1x512x512x16384xf32>
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 3.14 : f32
    %0 = tensor.empty() : tensor<1x1024x1024x16384xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1024x1024x16384xf32>) -> tensor<1x1024x1024x16384xf32>
    %6 = call @avgpool2d_nhwc_valid(%3) : (tensor<1x1024x1024x16384xf32>) -> tensor<1x512x512x16384xf32>
    return
  }
}

