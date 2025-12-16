#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
module {
  func.func @scaled_dot_product_attention(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
    %dim_2 = tensor.dim %arg1, %c1 : tensor<?x?x?xf32>
    %dim_3 = tensor.dim %arg2, %c2 : tensor<?x?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %cst_4 = arith.constant 3.40282347E+38 : f32
    %0 = arith.negf %cst_4 : f32
    %1 = tensor.empty(%dim, %dim_0, %dim_2) : tensor<?x?x?xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%2 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %25 = arith.mulf %in, %in_6 : f32
      %26 = arith.addf %out, %25 : f32
      linalg.yield %26 : f32
    } -> tensor<?x?x?xf32>
    %4 = arith.index_cast %dim_1 : index to i64
    %5 = arith.sitofp %4 : i64 to f32
    %cst_5 = arith.constant 1.000000e+00 : f32
    %6 = math.sqrt %5 : f32
    %7 = arith.divf %cst_5, %6 : f32
    %8 = tensor.empty(%dim, %dim_0, %dim_2) : tensor<?x?x?xf32>
    %9 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3 : tensor<?x?x?xf32>) outs(%8 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %25 = arith.mulf %in, %7 : f32
      linalg.yield %25 : f32
    } -> tensor<?x?x?xf32>
    %10 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %11 = linalg.fill ins(%0 : f32) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %12 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%9 : tensor<?x?x?xf32>) outs(%11 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %25 = arith.cmpf ogt, %in, %out : f32
      %26 = arith.select %25, %in, %out : f32
      linalg.yield %26 : f32
    } -> tensor<?x?xf32>
    %13 = tensor.empty(%dim, %dim_0, %dim_2) : tensor<?x?x?xf32>
    %14 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %12 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%13 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %25 = arith.subf %in, %in_6 : f32
      linalg.yield %25 : f32
    } -> tensor<?x?x?xf32>
    %15 = tensor.empty(%dim, %dim_0, %dim_2) : tensor<?x?x?xf32>
    %16 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14 : tensor<?x?x?xf32>) outs(%15 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %25 = math.exp %in : f32
      linalg.yield %25 : f32
    } -> tensor<?x?x?xf32>
    %17 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %19 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%16 : tensor<?x?x?xf32>) outs(%18 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %25 = arith.addf %out, %in : f32
      linalg.yield %25 : f32
    } -> tensor<?x?xf32>
    %20 = tensor.empty(%dim, %dim_0, %dim_2) : tensor<?x?x?xf32>
    %21 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16, %19 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%20 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %25 = arith.divf %in, %in_6 : f32
      linalg.yield %25 : f32
    } -> tensor<?x?x?xf32>
    %22 = tensor.empty(%dim, %dim_0, %dim_3) : tensor<?x?x?xf32>
    %23 = linalg.fill ins(%cst : f32) outs(%22 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %24 = linalg.generic {indexing_maps = [#map, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%21, %arg2 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%23 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %25 = arith.mulf %in, %in_6 : f32
      %26 = arith.addf %out, %25 : f32
      linalg.yield %26 : f32
    } -> tensor<?x?x?xf32>
    return %24 : tensor<?x?x?xf32>
  }
  func.func @main() -> i32 {
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x256x512xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x256x512xf32>) -> tensor<4x256x512xf32>
    %2 = tensor.empty() : tensor<4x256x512xf32>
    %3 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<4x256x512xf32>) outs(%2 : tensor<4x256x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = linalg.index 0 : index
      %15 = linalg.index 1 : index
      %16 = linalg.index 2 : index
      %17 = arith.index_cast %14 : index to i32
      %18 = arith.index_cast %15 : index to i32
      %19 = arith.index_cast %16 : index to i32
      %c7_i32 = arith.constant 7 : i32
      %c3_i32 = arith.constant 3 : i32
      %c13_i32 = arith.constant 13 : i32
      %cst_2 = arith.constant 1.300000e+01 : f32
      %20 = arith.muli %17, %c7_i32 : i32
      %21 = arith.muli %18, %c3_i32 : i32
      %22 = arith.addi %20, %21 : i32
      %23 = arith.addi %22, %19 : i32
      %24 = arith.remsi %23, %c13_i32 : i32
      %25 = arith.sitofp %24 : i32 to f32
      %26 = arith.divf %25, %cst_2 : f32
      linalg.yield %26 : f32
    } -> tensor<4x256x512xf32>
    %4 = tensor.empty() : tensor<4x256x512xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<4x256x512xf32>) -> tensor<4x256x512xf32>
    %6 = tensor.empty() : tensor<4x256x512xf32>
    %7 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<4x256x512xf32>) outs(%6 : tensor<4x256x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = linalg.index 0 : index
      %15 = linalg.index 1 : index
      %16 = linalg.index 2 : index
      %17 = arith.index_cast %14 : index to i32
      %18 = arith.index_cast %15 : index to i32
      %19 = arith.index_cast %16 : index to i32
      %c11_i32 = arith.constant 11 : i32
      %c5_i32 = arith.constant 5 : i32
      %c17_i32 = arith.constant 17 : i32
      %cst_2 = arith.constant 1.700000e+01 : f32
      %20 = arith.muli %17, %c11_i32 : i32
      %21 = arith.muli %18, %c5_i32 : i32
      %22 = arith.addi %20, %21 : i32
      %23 = arith.addi %22, %19 : i32
      %24 = arith.remsi %23, %c17_i32 : i32
      %25 = arith.sitofp %24 : i32 to f32
      %26 = arith.divf %25, %cst_2 : f32
      linalg.yield %26 : f32
    } -> tensor<4x256x512xf32>
    %8 = tensor.empty() : tensor<4x256x512xf32>
    %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<4x256x512xf32>) -> tensor<4x256x512xf32>
    %10 = tensor.empty() : tensor<4x256x512xf32>
    %11 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<4x256x512xf32>) outs(%10 : tensor<4x256x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = linalg.index 0 : index
      %15 = linalg.index 1 : index
      %16 = linalg.index 2 : index
      %17 = arith.index_cast %14 : index to i32
      %18 = arith.index_cast %15 : index to i32
      %19 = arith.index_cast %16 : index to i32
      %c7_i32 = arith.constant 7 : i32
      %c13_i32 = arith.constant 13 : i32
      %c19_i32 = arith.constant 19 : i32
      %cst_2 = arith.constant 1.900000e+01 : f32
      %20 = arith.muli %17, %c13_i32 : i32
      %21 = arith.muli %18, %c7_i32 : i32
      %22 = arith.addi %20, %21 : i32
      %23 = arith.addi %22, %19 : i32
      %24 = arith.remsi %23, %c19_i32 : i32
      %25 = arith.sitofp %24 : i32 to f32
      %26 = arith.divf %25, %cst_2 : f32
      linalg.yield %26 : f32
    } -> tensor<4x256x512xf32>
    %cast = tensor.cast %3 : tensor<4x256x512xf32> to tensor<?x?x?xf32>
    %cast_0 = tensor.cast %7 : tensor<4x256x512xf32> to tensor<?x?x?xf32>
    %cast_1 = tensor.cast %11 : tensor<4x256x512xf32> to tensor<?x?x?xf32>
    %12 = scf.for %arg0 = %c0 to %c100 step %c1 iter_args(%arg1 = %cst) -> (f32) {
      %14 = func.call @scaled_dot_product_attention(%cast, %cast_0, %cast_1) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      %cast_2 = tensor.cast %14 : tensor<?x?x?xf32> to tensor<4x256x512xf32>
      %extracted = tensor.extract %cast_2[%c0, %c0, %c0] : tensor<4x256x512xf32>
      %15 = arith.addf %arg1, %extracted : f32
      scf.yield %15 : f32
    }
    %13 = arith.fptosi %12 : f32 to i32
    return %13 : i32
  }
}

