module {
    func.func @gemm(%A : tensor<2048x2048xf32>, %W : tensor<2048x2048xf32>, %B : tensor<2048xf32>) -> tensor<2048x2048xf32> {
        %c0 = arith.constant 0.0 : f32
        %empty = tensor.empty() : tensor<2048x2048xf32>
        %init0 = linalg.fill ins(%c0 : f32) outs(%empty : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>

        %mm = linalg.matmul ins(%A, %W : tensor<2048x2048xf32>, tensor<2048x2048xf32>) outs(%init0 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>

        %init1 = linalg.fill ins(%c0 : f32) outs(%empty : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
        %mm_bias = linalg.generic
        {
            indexing_maps = [
            affine_map<(m,n) -> (m,n)>,
            affine_map<(m,n) -> (n)>,
            affine_map<(m,n) -> (m,n)>
            ],
            iterator_types = ["parallel", "parallel"]
        } ins(%mm, %B : tensor<2048x2048xf32>, tensor<2048xf32>)
        outs(%init1 : tensor<2048x2048xf32>) {
            ^bb0(%a : f32, %b : f32, %o : f32):
                %0 = arith.addf %a, %b : f32
                linalg.yield %0 : f32
        } -> tensor<2048x2048xf32>

        %init2 = linalg.fill ins(%c0 : f32) outs(%empty : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
        %bias_relu = linalg.generic
        {
            indexing_maps = [
            affine_map<(m,n) -> (m,n)>,
            affine_map<(m,n) -> (m,n)>
            ],
            iterator_types = ["parallel", "parallel"]
        } ins(%mm_bias : tensor<2048x2048xf32>)
        outs(%init2 : tensor<2048x2048xf32>) {
            ^bb0(%in : f32, %out : f32):
            %0 = arith.maximumf %in, %c0 : f32
            linalg.yield %0 : f32
        } -> tensor<2048x2048xf32>

        return %bias_relu : tensor<2048x2048xf32>
    }

    func.func @main() attributes {llvm.emit_c_interface} {
        %c1 = arith.constant 1.0 : f32
        %empty = tensor.empty() : tensor<2048x2048xf32>
        %empty_bias = tensor.empty() : tensor<2048xf32>
        %A = linalg.fill ins(%c1 : f32) outs(%empty : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
        %W = linalg.fill ins(%c1 : f32) outs(%empty : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
        %B = linalg.fill ins(%c1 : f32) outs(%empty_bias : tensor<2048xf32>) -> tensor<2048xf32>

        %result = func.call @gemm(%A,%W,%B) : (tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>) -> tensor<2048x2048xf32>

        func.return
    }
}
