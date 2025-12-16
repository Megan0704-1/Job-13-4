module {
  func.func @pointwise_1x1_nhwc(%arg0: tensor<1x64x64x64xf32>, %arg1: tensor<1x1x64x32768xf32>) -> tensor<1x64x64x32768xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x64x64x32768xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x64x64x32768xf32>) -> tensor<1x64x64x32768xf32>
    %2 = linalg.conv_2d_nhwc_hwcf ins(%arg0, %arg1 : tensor<1x64x64x64xf32>, tensor<1x1x64x32768xf32>) outs(%1 : tensor<1x64x64x32768xf32>) -> tensor<1x64x64x32768xf32>
    return %2 : tensor<1x64x64x32768xf32>
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x64x64x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x64x64x64xf32>) -> tensor<1x64x64x64xf32>
    %2 = tensor.empty() : tensor<1x1x64x32768xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x1x64x32768xf32>) -> tensor<1x1x64x32768xf32>
    %4 = call @pointwise_1x1_nhwc(%1, %3) : (tensor<1x64x64x64xf32>, tensor<1x1x64x32768xf32>) -> tensor<1x64x64x32768xf32>
    return
  }
}

