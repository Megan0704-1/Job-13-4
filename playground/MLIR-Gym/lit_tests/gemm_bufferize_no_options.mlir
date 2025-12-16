// RUN: %mlir-opt %s \
// RUN:   --linalg-generalize-named-ops \
// RUN:   --canonicalize --cse \
// RUN:   --one-shot-bufferize \
// RUN:   --canonicalize --cse \
// RUN: | %FileCheck %s

// simple one-shot-bufferize pass inserts type conversion operators
// this makes convert-bufferization-to-memref pass failed.
// This tests demonstrate the difference. (cmp w/ bufferize-function-boundaries)

func.func @gemm(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %M: index, %N: index, %K: index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty(%M, %N) : tensor<?x?xf32>
  %C0 = linalg.fill ins(%c0 : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %C = linalg.matmul
      ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
     outs(%C0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %C : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @gemm(
// tensor type is expected is exist for no-option-bufferize (type conversion on function args)
// CHECK: tensor
// CHECK: memref

// CHECK: linalg.generic

// CHECK: bufferization.to_tensor
// CHECK-NOT: affine.for
