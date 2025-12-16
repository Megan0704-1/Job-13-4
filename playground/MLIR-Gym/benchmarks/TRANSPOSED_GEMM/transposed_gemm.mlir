module {
  // C[M,N] = A[M,K] * B^T[N,K]  where B_nt is (N,K)
  func.func @transposed_gemm(%A: tensor<4096x1024xf32>,
                             %B_nt: tensor<4096x1024xf32>)
      -> tensor<4096x4096xf32> {
    %zero = arith.constant 0.0 : f32
    %C0   = tensor.empty() : tensor<4096x4096xf32>
    %Cinit= linalg.fill ins(%zero : f32)
              outs(%C0 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>

    %C = linalg.generic
      { indexing_maps = [
          affine_map<(d0,d1,d2)->(d0,d2)>, // A(i,k)
          affine_map<(d0,d1,d2)->(d1,d2)>, // B_nt(j,k)
          affine_map<(d0,d1,d2)->(d0,d1)>  // C(i,j)
        ],
        iterator_types = ["parallel","parallel","reduction"] }
      ins(%A, %B_nt : tensor<4096x1024xf32>, tensor<4096x1024xf32>)
      outs(%Cinit : tensor<4096x4096xf32>) {
        ^bb0(%a: f32, %b: f32, %c: f32):
          %p = arith.mulf %a, %b : f32
          %s = arith.addf %c, %p : f32
          linalg.yield %s : f32
      } -> tensor<4096x4096xf32>
    return %C : tensor<4096x4096xf32>
  }

  func.func @main() attributes { llvm.emit_c_interface } {
    %one = arith.constant 1.0 : f32
    %A0  = tensor.empty() : tensor<4096x1024xf32>
    %B0  = tensor.empty() : tensor<4096x1024xf32>
    %A   = linalg.fill ins(%one : f32)
             outs(%A0 : tensor<4096x1024xf32>) -> tensor<4096x1024xf32>
    %B   = linalg.fill ins(%one : f32)
             outs(%B0 : tensor<4096x1024xf32>) -> tensor<4096x1024xf32>

    %C = func.call @transposed_gemm(%A, %B)
         : (tensor<4096x1024xf32>, tensor<4096x1024xf32>) -> tensor<4096x4096xf32>
    return
  }
}

