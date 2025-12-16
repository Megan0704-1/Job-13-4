##Analysis

### 1. LoopAnalysis
Purpose: Turn raw MLIR into a fixed-shape, invariant per-loop feature block that’s stable under benign IR rewrites (renaming, spacing, many interchanges) and safe to batch.
What it parses: Scans the IR text (no mlir-opt needed) for scf.for, scf.parallel, and affine.for inside each func.func body. It is brace-aware (tracks {/} depth) and identifies which function a loop belongs to.

#### Per-loop features (6 channels):
- log_tripcount — log1p of the estimated iteration count (from literal lb/ub/step; unknown → 0 and flagged as unknown).
- step_is_pow2 — 1 if step is a power of two (for scf.parallel: all dims pow2).
- is_unit_stride_minor — placeholder (filled later by memory analysis).
- loop_carries_reuse_any — placeholder (filled later by memory analysis).
- depth_norm — loop depth normalized within its function to [0,1].
- is_parallel_hint — 1 for scf.parallel, else 0.

#### Masks:
- loops_mask [N] — which rows in the fixed [N,6] tensor are real loops (1) vs padding (0).
- known_mask [N,6] — per-feature validity; e.g., tripcount known vs symbolic, memory-derived channels not yet known, therefore is marked as False in LoopAnalysis.

#### Depth handling:
Records brace depth at each header, then re-bases per function so outermost loop has depth 1. depth_norm uses each function’s max depth, keeping values comparable across functions.

#### Output:
- loop_feats [N,6] (float32),
- loops_mask [N] (bool/int8),
- known_mask [N,6] (bool/int8),
- records (raw metadata: loop_id, kind, function, source span) for downstream joins (e.g., memory analysis).

### 2. MemoryAnalysis
- Fills in missing parts of feature after LoopAnalysis. (is_unit_stride_minor, loop_carries_reuse_any)
- Introduce per array features.

#### For LoopAnalysis features:
- is_unit_stride_minor: this feature tells, if this is the loop updating (mutating/accessing) the last dimension of the "target" array.
- loop_carries_reuse_any: this feature tells, if this loop does not change elements of any arrays (broadcast/copy)
    - note that if it accesses a dimension, value is set, otherwise 0.

Let's see examples

#### Example 1 : GEMM C[i,j] += A[i,k] * B[k,j], loops (i, j, k)
Signals
# row-major C
- is_unit_stride_minor[i] = 0
- is_unit_stride_minor[j] = 1 (j is iterating the minor dim).
- is_unit_stride_minor[k] = 0

- loop_carries_reuse_any[i] = 1
- loop_carries_reuse_any[j] = 1
- loop_carries_reuse_any[k] = 1

#### Example 2 — 1D strided copy: dst[i] = src[2*i]
Signals

- is_unit_stride_minor[i] = 0 (effective stride 2),
- stride_hist[i] mass in {>W} or 2..W depending on W.

The agent should avoid naive vectorize (gathers) and consider interchange (if there’s an inner loop) or wider tiling with scalarization.
The feature prevents it from repeatedly selecting a doomed vectorize pass.

Example 3 — Broadcast add: Y[i,j] = X[i,j] + b[j]
Signals

- loop_carries_reuse_any[i] = 1 for b[j] (independent of i).
- is_unit_stride_minor[j] = 1 (row-major).

Tile on i (outer) to keep b[j] hot; vectorize on j.
Hope the agent learns the pattern (outer reuse + inner unit-stride) rather than memorizing specific indices.
