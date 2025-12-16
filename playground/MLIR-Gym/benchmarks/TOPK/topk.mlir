#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @row_topk_k128(%arg0: tensor<8192x8192xf32>) -> tensor<8192x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant -3.40282347E+38 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %0 = tensor.empty() : tensor<8192x8192xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8192x8192xf32>) -> tensor<8192x8192xf32>
    %2:2 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %arg0, %arg3 = %1) -> (tensor<8192x8192xf32>, tensor<8192x8192xf32>) {
      %3 = tensor.empty() : tensor<8192xf32>
      %4 = tensor.empty() : tensor<8192xindex>
      %5 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<8192xf32>) -> tensor<8192xf32>
      %6 = linalg.fill ins(%c0 : index) outs(%4 : tensor<8192xindex>) -> tensor<8192xindex>
      %7:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<8192x8192xf32>) outs(%5, %6 : tensor<8192xf32>, tensor<8192xindex>) {
      ^bb0(%in: f32, %out: f32, %out_1: index):
        %11 = linalg.index 1 : index
        %12 = arith.cmpf ogt, %in, %out : f32
        %13 = arith.select %12, %in, %out : f32
        %14 = arith.select %12, %11, %out_1 : index
        linalg.yield %13, %14 : f32, index
      } -> (tensor<8192xf32>, tensor<8192xindex>)
      %8 = tensor.empty() : tensor<8192x1xf32>
      %9 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%7#0 : tensor<8192xf32>) outs(%8 : tensor<8192x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<8192x1xf32>
      %inserted_slice = tensor.insert_slice %9 into %arg3[0, %arg1] [8192, 1] [1, 1] : tensor<8192x1xf32> into tensor<8192x8192xf32>
      %10 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%7#1 : tensor<8192xindex>) outs(%arg2 : tensor<8192x8192xf32>) {
      ^bb0(%in: index, %out: f32):
        %11 = linalg.index 1 : index
        %12 = arith.cmpi eq, %11, %in : index
        %13 = arith.select %12, %cst_0, %out : f32
        linalg.yield %13 : f32
      } -> tensor<8192x8192xf32>
      scf.yield %10, %inserted_slice : tensor<8192x8192xf32>, tensor<8192x8192xf32>
    }
    %extracted_slice = tensor.extract_slice %2#1[0, 0] [8192, 128] [1, 1] : tensor<8192x8192xf32> to tensor<8192x128xf32>
    return %extracted_slice : tensor<8192x128xf32>
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<8192x8192xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8192x8192xf32>) -> tensor<8192x8192xf32>
    %2 = call @row_topk_k128(%1) : (tensor<8192x8192xf32>) -> tensor<8192x128xf32>
    return
  }
}

