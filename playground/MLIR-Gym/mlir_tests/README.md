# MLIR benchmark tests

to add test, there are some things we need to do
1. Add FileCheck commands, for example
Say this is the original pass pipeline
```
// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(convert-scf-to-cf,convert-arith-to-llvm),finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)" \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN: | FileCheck %s
```

we change the mlir-cpu-runner to translate, and logics to produce executables
```
// RUN: %mlir-opt %s -pass-pipeline="builtin.module(func.func(convert-scf-to-cf,convert-arith-to-llvm),finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)" |\
// RUN: %mlir-translate --mlir-to-llvmir \
// RUN: | %clang -x ir -opaque-pointers - -o %t.original %link-dir && %t.original > $t.original.out
// RUN: %FileCheck %s --input-file=$t.original.out
```

and add custom pipeline variables to match output
```
// RUN: %mlir-opt %s -pass-pipeline="%{custom-pipeline}" |\
// RUN: %mlir-translate --mlir-to-llvmir \
// RUN: | %clang -x ir -opaque-pointers - -o %t.custom %link-dir && %t.custom > $t.custom.out
// RUN: %FileCheck %s --input-file=$t.custom.out
// REQUIRES: request_custom_pipeline
```
in lit.cfg.py, we set a flag for `request_custom_pipeline` to dynamically invoke the last filecheck only if custom-pipeline variables is provided as lit params
