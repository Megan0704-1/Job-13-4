module {
  func.func @dwconv2d_nhwc_hwc_valid(%arg0: tensor<1x256x256x16384xf32>, %arg1: tensor<3x3x16384xf32>) -> tensor<1x254x254x16384xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x254x254x16384xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x254x254x16384xf32>) -> tensor<1x254x254x16384xf32>
    %2 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x256x256x16384xf32>, tensor<3x3x16384xf32>) outs(%1 : tensor<1x254x254x16384xf32>) -> tensor<1x254x254x16384xf32>
    return %2 : tensor<1x254x254x16384xf32>
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 3.140000e+00 : f32
    %cst_0 = arith.constant 2.000000e-01 : f32
    %0 = tensor.empty() : tensor<1x256x256x16384xf32>
    %1 = tensor.empty() : tensor<3x3x16384xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x256x256x16384xf32>) -> tensor<1x256x256x16384xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<3x3x16384xf32>) -> tensor<3x3x16384xf32>
    %4 = call @dwconv2d_nhwc_hwc_valid(%2, %3) : (tensor<1x256x256x16384xf32>, tensor<3x3x16384xf32>) -> tensor<1x254x254x16384xf32>
    return
  }
}

