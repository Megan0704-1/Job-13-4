module {
  func.func @bmm_bias_relu(%A : tensor<8x1024x1024xf32>,
                           %B : tensor<8x1024x1024xf32>,
                           %Bias : tensor<1024xf32>) -> tensor<8x1024x1024xf32> {
    %c0 = arith.constant 0.0 : f32

    // Batch matmul
    %empty0 = tensor.empty() : tensor<8x1024x1024xf32>
    %init0  = linalg.fill ins(%c0 : f32) outs(%empty0 : tensor<8x1024x1024xf32>) -> tensor<8x1024x1024xf32>
    %bmm = linalg.batch_matmul
      ins(%A, %B : tensor<8x1024x1024xf32>, tensor<8x1024x1024xf32>)
      outs(%init0 : tensor<8x1024x1024xf32>) -> tensor<8x1024x1024xf32>

    // Add bias (broadcast over batch and M): (b,m,n) + (n) -> (b,m,n)
    %empty1 = tensor.empty() : tensor<8x1024x1024xf32>
    %init1  = linalg.fill ins(%c0 : f32) outs(%empty1 : tensor<8x1024x1024xf32>) -> tensor<8x1024x1024xf32>
    %bmm_bias = linalg.generic
      {
        indexing_maps = [
          affine_map<(b,m,n) -> (b,m,n)>,  // bmm
          affine_map<(b,m,n) -> (n)>,      // Bias
          affine_map<(b,m,n) -> (b,m,n)>   // out
        ],
        iterator_types = ["parallel", "parallel", "parallel"]
      }
      ins(%bmm, %Bias : tensor<8x1024x1024xf32>, tensor<1024xf32>)
      outs(%init1 : tensor<8x1024x1024xf32>) {
        ^bb0(%x : f32, %b : f32, %o : f32):
          %sum = arith.addf %x, %b : f32
          linalg.yield %sum : f32
      } -> tensor<8x1024x1024xf32>

    // ReLU
    %empty2 = tensor.empty() : tensor<8x1024x1024xf32>
    %init2  = linalg.fill ins(%c0 : f32) outs(%empty2 : tensor<8x1024x1024xf32>) -> tensor<8x1024x1024xf32>
    %relu = linalg.generic
      {
        indexing_maps = [
          affine_map<(b,m,n) -> (b,m,n)>,
          affine_map<(b,m,n) -> (b,m,n)>
        ],
        iterator_types = ["parallel", "parallel", "parallel"]
      }
      ins(%bmm_bias : tensor<8x1024x1024xf32>)
      outs(%init2 : tensor<8x1024x1024xf32>) {
        ^bb0(%in : f32, %out : f32):
          %max = arith.maximumf %in, %c0 : f32
          linalg.yield %max : f32
      } -> tensor<8x1024x1024xf32>

    return %relu : tensor<8x1024x1024xf32>
  }

  // Simple main that fills inputs with 1.0 and calls the op.
  func.func @main() attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1.0 : f32

    %Ae = tensor.empty() : tensor<8x1024x1024xf32>
    %Be = tensor.empty() : tensor<8x1024x1024xf32>
    %be = tensor.empty() : tensor<1024xf32>

    %A = linalg.fill ins(%c1 : f32) outs(%Ae : tensor<8x1024x1024xf32>) -> tensor<8x1024x1024xf32>
    %B = linalg.fill ins(%c1 : f32) outs(%Be : tensor<8x1024x1024xf32>) -> tensor<8x1024x1024xf32>
    %Bias = linalg.fill ins(%c1 : f32) outs(%be : tensor<1024xf32>) -> tensor<1024xf32>

    %out = func.call @bmm_bias_relu(%A, %B, %Bias)
      : (tensor<8x1024x1024xf32>, tensor<8x1024x1024xf32>, tensor<1024xf32>) -> tensor<8x1024x1024xf32>

    func.return
  }
}

