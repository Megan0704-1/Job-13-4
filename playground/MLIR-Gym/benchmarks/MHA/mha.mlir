#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @multi_head_attention_4d(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?x?x?xf32>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<?x?x?x?xf32>
    %dim_2 = tensor.dim %arg0, %c3 : tensor<?x?x?x?xf32>
    %dim_3 = tensor.dim %arg1, %c2 : tensor<?x?x?x?xf32>
    %dim_4 = tensor.dim %arg2, %c3 : tensor<?x?x?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant 3.40282347E+38 : f32
    %0 = arith.negf %cst_5 : f32
    %cst_6 = arith.constant 1.000000e+00 : f32
    %1 = tensor.empty(%dim, %dim_0, %dim_1, %dim_3) : tensor<?x?x?x?xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%2 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %25 = arith.mulf %in, %in_7 : f32
      %26 = arith.addf %out, %25 : f32
      linalg.yield %26 : f32
    } -> tensor<?x?x?x?xf32>
    %4 = arith.index_cast %dim_2 : index to i64
    %5 = arith.sitofp %4 : i64 to f32
    %6 = math.sqrt %5 : f32
    %7 = arith.divf %cst_6, %6 : f32
    %8 = tensor.empty(%dim, %dim_0, %dim_1, %dim_3) : tensor<?x?x?x?xf32>
    %9 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<?x?x?x?xf32>) outs(%8 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %25 = arith.mulf %in, %7 : f32
      linalg.yield %25 : f32
    } -> tensor<?x?x?x?xf32>
    %10 = tensor.empty(%dim, %dim_0, %dim_1) : tensor<?x?x?xf32>
    %11 = linalg.fill ins(%0 : f32) outs(%10 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %12 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%9 : tensor<?x?x?x?xf32>) outs(%11 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %25 = arith.cmpf ogt, %in, %out : f32
      %26 = arith.select %25, %in, %out : f32
      linalg.yield %26 : f32
    } -> tensor<?x?x?xf32>
    %13 = tensor.empty(%dim, %dim_0, %dim_1, %dim_3) : tensor<?x?x?x?xf32>
    %14 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %12 : tensor<?x?x?x?xf32>, tensor<?x?x?xf32>) outs(%13 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %25 = arith.subf %in, %in_7 : f32
      linalg.yield %25 : f32
    } -> tensor<?x?x?x?xf32>
    %15 = tensor.empty(%dim, %dim_0, %dim_1, %dim_3) : tensor<?x?x?x?xf32>
    %16 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14 : tensor<?x?x?x?xf32>) outs(%15 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %25 = math.exp %in : f32
      linalg.yield %25 : f32
    } -> tensor<?x?x?x?xf32>
    %17 = tensor.empty(%dim, %dim_0, %dim_1) : tensor<?x?x?xf32>
    %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %19 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%16 : tensor<?x?x?x?xf32>) outs(%18 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %25 = arith.addf %out, %in : f32
      linalg.yield %25 : f32
    } -> tensor<?x?x?xf32>
    %20 = tensor.empty(%dim, %dim_0, %dim_1, %dim_3) : tensor<?x?x?x?xf32>
    %21 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16, %19 : tensor<?x?x?x?xf32>, tensor<?x?x?xf32>) outs(%20 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %25 = arith.divf %in, %in_7 : f32
      linalg.yield %25 : f32
    } -> tensor<?x?x?x?xf32>
    %22 = tensor.empty(%dim, %dim_0, %dim_1, %dim_4) : tensor<?x?x?x?xf32>
    %23 = linalg.fill ins(%cst : f32) outs(%22 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    %24 = linalg.generic {indexing_maps = [#map2, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]} ins(%21, %arg2 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%23 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %25 = arith.mulf %in, %in_7 : f32
      %26 = arith.addf %out, %25 : f32
      linalg.yield %26 : f32
    } -> tensor<?x?x?x?xf32>
    return %24 : tensor<?x?x?x?xf32>
  }
  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %c100 = arith.constant 100 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x4x256x512xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4x256x512xf32>) -> tensor<2x4x256x512xf32>
    %2 = tensor.empty() : tensor<2x4x256x512xf32>
    %3 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<2x4x256x512xf32>) outs(%2 : tensor<2x4x256x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = linalg.index 0 : index
      %15 = linalg.index 1 : index
      %16 = linalg.index 2 : index
      %17 = linalg.index 3 : index
      %18 = arith.index_cast %14 : index to i32
      %19 = arith.index_cast %15 : index to i32
      %20 = arith.index_cast %16 : index to i32
      %21 = arith.index_cast %17 : index to i32
      %c7_i32 = arith.constant 7 : i32
      %c3_i32 = arith.constant 3 : i32
      %c19_i32 = arith.constant 19 : i32
      %cst_3 = arith.constant 1.900000e+01 : f32
      %22 = arith.muli %18, %c7_i32 : i32
      %23 = arith.muli %19, %c3_i32 : i32
      %24 = arith.addi %22, %23 : i32
      %25 = arith.addi %24, %20 : i32
      %26 = arith.addi %25, %21 : i32
      %27 = arith.remsi %26, %c19_i32 : i32
      %28 = arith.sitofp %27 : i32 to f32
      %29 = arith.divf %28, %cst_3 : f32
      linalg.yield %29 : f32
    } -> tensor<2x4x256x512xf32>
    %4 = tensor.empty() : tensor<2x4x256x512xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<2x4x256x512xf32>) -> tensor<2x4x256x512xf32>
    %6 = tensor.empty() : tensor<2x4x256x512xf32>
    %7 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5 : tensor<2x4x256x512xf32>) outs(%6 : tensor<2x4x256x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = linalg.index 0 : index
      %15 = linalg.index 1 : index
      %16 = linalg.index 2 : index
      %17 = linalg.index 3 : index
      %18 = arith.index_cast %14 : index to i32
      %19 = arith.index_cast %15 : index to i32
      %20 = arith.index_cast %16 : index to i32
      %21 = arith.index_cast %17 : index to i32
      %c11_i32 = arith.constant 11 : i32
      %c5_i32 = arith.constant 5 : i32
      %c17_i32 = arith.constant 17 : i32
      %cst_3 = arith.constant 1.700000e+01 : f32
      %22 = arith.muli %18, %c11_i32 : i32
      %23 = arith.muli %19, %c5_i32 : i32
      %24 = arith.addi %22, %23 : i32
      %25 = arith.addi %24, %20 : i32
      %26 = arith.addi %25, %21 : i32
      %27 = arith.remsi %26, %c17_i32 : i32
      %28 = arith.sitofp %27 : i32 to f32
      %29 = arith.divf %28, %cst_3 : f32
      linalg.yield %29 : f32
    } -> tensor<2x4x256x512xf32>
    %8 = tensor.empty() : tensor<2x4x256x512xf32>
    %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<2x4x256x512xf32>) -> tensor<2x4x256x512xf32>
    %10 = tensor.empty() : tensor<2x4x256x512xf32>
    %11 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9 : tensor<2x4x256x512xf32>) outs(%10 : tensor<2x4x256x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = linalg.index 0 : index
      %15 = linalg.index 1 : index
      %16 = linalg.index 2 : index
      %17 = linalg.index 3 : index
      %18 = arith.index_cast %14 : index to i32
      %19 = arith.index_cast %15 : index to i32
      %20 = arith.index_cast %16 : index to i32
      %21 = arith.index_cast %17 : index to i32
      %c13_i32 = arith.constant 13 : i32
      %c7_i32 = arith.constant 7 : i32
      %c19_i32 = arith.constant 19 : i32
      %cst_3 = arith.constant 1.900000e+01 : f32
      %22 = arith.muli %18, %c13_i32 : i32
      %23 = arith.muli %19, %c7_i32 : i32
      %24 = arith.addi %22, %23 : i32
      %25 = arith.addi %24, %20 : i32
      %26 = arith.addi %25, %21 : i32
      %27 = arith.remsi %26, %c19_i32 : i32
      %28 = arith.sitofp %27 : i32 to f32
      %29 = arith.divf %28, %cst_3 : f32
      linalg.yield %29 : f32
    } -> tensor<2x4x256x512xf32>
    %cast = tensor.cast %3 : tensor<2x4x256x512xf32> to tensor<?x?x?x?xf32>
    %cast_0 = tensor.cast %7 : tensor<2x4x256x512xf32> to tensor<?x?x?x?xf32>
    %cast_1 = tensor.cast %11 : tensor<2x4x256x512xf32> to tensor<?x?x?x?xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %12 = scf.for %arg0 = %c0 to %c100 step %c1 iter_args(%arg1 = %cst_2) -> (f32) {
      %14 = func.call @multi_head_attention_4d(%cast, %cast_0, %cast_1) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
      %cast_3 = tensor.cast %14 : tensor<?x?x?x?xf32> to tensor<2x4x256x512xf32>
      %extracted = tensor.extract %cast_3[%c0, %c0, %c0, %c0] : tensor<2x4x256x512xf32>
      %15 = arith.addf %arg1, %extracted : f32
      scf.yield %15 : f32
    }
    %13 = arith.fptosi %12 : f32 to i32
    return %13 : i32
  }
}

