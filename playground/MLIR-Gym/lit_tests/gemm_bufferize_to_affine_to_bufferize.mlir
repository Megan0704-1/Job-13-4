// RUN: %mlir-opt %s \
// RUN:   --linalg-generalize-named-ops \
// RUN:   --canonicalize --cse \
// RUN:   --one-shot-bufferize=bufferize-function-boundaries \
// RUN:   --canonicalize --cse \
// RUN:   --convert-linalg-to-affine-loops \
// RUN:   --convert-bufferization-to-memref \
// RUN:   --one-shot-bufferize=bufferize-function-boundaries \
// RUN: | %FileCheck %s

// sanity check for: bufferize -> affine

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
// Expect no tensor types in the signature/result after bufferization.
// CHECK-NOT: tensor<

// CHECK-DAG:       memref.alloc
// CHECK-DAG:       affine.for
// CHECK-DAG:         affine.for
// CHECK-DAG:           affine.store

// CHECK-DAG:       memref.dim
// CHECK-DAG:       memref.dim
// CHECK-DAG:       memref.dim

// CHECK-DAG:       affine.for
// CHECK-DAG:         affine.for
// CHECK-DAG:           affine.for
// CHECK-DAG:             affine.load
// CHECK-DAG:             affine.load
// CHECK-DAG:             affine.load
// CHECK-DAG:               arith.mulf
// CHECK-DAG:               arith.addf
// CHECK-DAG:               affine.store

// CHECK:           return
