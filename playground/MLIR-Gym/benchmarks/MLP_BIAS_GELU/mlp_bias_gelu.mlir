#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  // GELU(t) ≈ 0.5 * t * [1 + tanh( sqrt(2/pi)*(t + 0.044715*t^3) )]
  func.func @mlp_bias_gelu_tanh(
      %A : tensor<2048x2048xf32>,
      %W : tensor<2048x2048xf32>,
      %B : tensor<2048xf32>) -> tensor<2048x2048xf32> {

    %c0 = arith.constant 0.0 : f32
    %half = arith.constant 5.000000e-01 : f32
    %one  = arith.constant 1.000000e+00 : f32
    %k0 = arith.constant 7.97884561e-01 : f32  // sqrt(2/pi)
    %k1 = arith.constant 4.47150000e-02 : f32  // 0.044715

    // MatMul
    %emptyM = tensor.empty() : tensor<2048x2048xf32>
    %mm_init = linalg.fill ins(%c0 : f32)
                 outs(%emptyM : tensor<2048x2048xf32>)
                 -> tensor<2048x2048xf32>
    %mm = linalg.matmul
            ins(%A, %W : tensor<2048x2048xf32>, tensor<2048x2048xf32>)
            outs(%mm_init : tensor<2048x2048xf32>)
            -> tensor<2048x2048xf32>

    // Bias + GELU(tanh) 一次完成
    %out_init = linalg.fill ins(%c0 : f32)
                 outs(%emptyM : tensor<2048x2048xf32>)
                 -> tensor<2048x2048xf32>
    %y = linalg.generic
      { indexing_maps = [
          affine_map<(m,n)->(m,n)>,
          affine_map<(m,n)->(n)>,
          affine_map<(m,n)->(m,n)>
        ],
        iterator_types = ["parallel","parallel"] }
      ins(%mm, %B : tensor<2048x2048xf32>, tensor<2048xf32>)
      outs(%out_init : tensor<2048x2048xf32>) {
      ^bb0(%mmv: f32, %bv: f32, %o: f32):
        %t  = arith.addf %mmv, %bv : f32
        %t2 = arith.mulf %t, %t : f32
        %t3 = arith.mulf %t2, %t : f32
        %a  = arith.mulf %k1, %t3 : f32
        %b  = arith.addf %t, %a : f32
        %c  = arith.mulf %k0, %b : f32
        %th = math.tanh %c : f32
        %one_plus_th = arith.addf %one, %th : f32
        %t_mul = arith.mulf %t, %one_plus_th : f32
        %gelu = arith.mulf %half, %t_mul : f32
        linalg.yield %gelu : f32
    } -> tensor<2048x2048xf32>

    return %y : tensor<2048x2048xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<2048x2048xf32>
    %1 = tensor.empty() : tensor<2048xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%1 : tensor<2048xf32>) -> tensor<2048xf32>
    %5 = call @mlp_bias_gelu_tanh(%2, %3, %4) : (tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>) -> tensor<2048x2048xf32>
    return
  }
}

