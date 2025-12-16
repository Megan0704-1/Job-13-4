module {
  func.func @conv2d_nchw_fchw(%arg0: tensor<1x128x128x128xf32>, %arg1: tensor<1024x128x1x1xf32>) -> tensor<1x1024x128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1024x128x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1024x128x128xf32>) -> tensor<1x1024x128x128xf32>
    %2 = linalg.conv_2d_nchw_fchw ins(%arg0, %arg1 : tensor<1x128x128x128xf32>, tensor<1024x128x1x1xf32>) outs(%1 : tensor<1x1024x128x128xf32>) -> tensor<1x1024x128x128xf32>
    return %2 : tensor<1x1024x128x128xf32>
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x128x128x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
    %2 = tensor.empty() : tensor<1024x128x1x1xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1024x128x1x1xf32>) -> tensor<1024x128x1x1xf32>
    %4 = call @conv2d_nchw_fchw(%1, %3) : (tensor<1x128x128x128xf32>, tensor<1024x128x1x1xf32>) -> tensor<1x1024x128x128xf32>
    return
  }
}

