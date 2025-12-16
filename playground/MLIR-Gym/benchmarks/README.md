## Sample command for getting baseline:
```bash 
# cd to project root
./benchmarks/benchmark.sh \
--mlir benchmarks/GEMM/gemm.mlir \
--build ../../llvm-project/build/ \
--entry main \
--runs 10 \
--pipeline "--pass-pipeline=$(< benchmarks/baseline/gemm_pipeline)" \
--backend aot \
--cpu 10
```

AOT baseline
| L0 | B0 | C0 |
|--------|--------|--------|
| 34530.00 ms | 1.92 kB | 10.00 ms |

JIT baseline
| L0 | B0 | C0 |
|--------|--------|--------|
| 36515.00 ms | 2.15 kB | 10.00 ms |

The benchmark script summarizes compile time (C0), object file byte size (B0) and execution latency (L0)

execution policy is set for 1 cpu core
median statistic is used for compile time and latency metrics

There will be 2 pipelines in the baseline folder
xxx_pipeline is the minimum passes to convert, lower and run
xxx_target_pipeline is a better combination of passes that give smaller code size or lower latency or both

C0, B0, and L0 are served as context for the feature-state
