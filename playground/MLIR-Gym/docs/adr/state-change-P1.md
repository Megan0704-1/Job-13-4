## ADR: State - Feature engineering (Phase 1)

- Status: Approved
- Date: 2025-09-25
- Issue: [#32](https://github.com/MPSLab-ASU/MLIR-Gym/issues/32)
- Scope: MLIRGym State

## Context
Current observations:
I didn't really think about engineering on the state and observation. Past designs such as OP_CNT, uses a dictionary using intrinsics as keys and their counts in current IR as values.
The design doesn't capture the computation graph in anyways, and difficult to maintain as more dialects and operations are produced throughout the lowering pipeline.
Another design IR_STR uses transformers to encode entire IR_STR, from the past experience, using some encoder that I am not familiar with does not help when things went wrong.
Hence I decide to take a while and implement this, extracting features and recording statistics from the IR, and this requires careful state design.
The state must align with legality/termination while remaining reproducible.

**Note.**
MLIRGym now has **Action masking**, **STOP/EVAL**, and **reward redesign** in flight.

### Loop Analysis (what / goal / output / observation)

#### What
Walk the IR, detect loop bands, and summarize each loop with iteration-space invariants that don’t change under benign rewrites (renaming, canonicalize, legal interchange).

#### Goal
Give the agent just enough loop shape to decide tiling/interchange/vectorization, in a fixed shape and Markov way (no SSA/name dependence).

#### Output (per loop, up to L)
- log_tripcount — log1p(tripcount) if lb/ub/step are constant; else 0.
- step_is_pow2 — 1 if step is a power of two; else 0.
- depth_norm — loop depth normalized within its band [0,1] (outer→inner).
- is_parallel_hint — 1 for scf.parallel/affine.parallel; else 0.
Also emit masks:
- loops_mask[L] — which rows are real loops (padding otherwise).
- known_mask[L,4] — channelwise validity for the four features above.

Observation representation
- obs["loop_feats"] ∈ ℝ^{L×4} in the order above.
- obs["loop_mask"] ∈ {0,1}^{L}.
- obs["known_mask"] ∈ {0,1}^{L×4}.

### Memory Analysis (what / goal / output / observation)

#### What
Analyze loads/stores by combining array layout/strides with affine/linalg indexing maps, then project results per loop.

#### Goal
Tell the agent, for each loop, (a) whether it gives unit-stride (vectorization target) and (b) whether it carries reuse across iterations (tiling/interchange target). Provide small, layout-aware signals that are stable across legal rewrites.

#### Core output (per loop j, j=1..L)
- unit_stride[j] ∈ {1, 0, −1}
= 1 if stepping loop j with step=1 changes at least one accessed array’s linear element address by exactly +1 (true minor-dim walk);
= 0 if candidates exist but none give +1;
= −1 if unknown (insufficient static info).

- reuse_any[j] ∈ {0,1}
= 1 if for any accessed array, its address is independent of loop j (all coefficients wrt j are zero) → cross-iteration reuse.

Optional extras (config-gated, also per loop)
- contiguity_score[j] ∈ [0,1] = 1/(1+|Δ(j)−1|) best across arrays.
- stride_hist[j,•] with 4 buckets: {=1, 2..W, >W, unknown} (W = HW vector width).
- reuse_count[j] = number of arrays with reuse on j.

Observation representation
Packed as a separate “memory” channel:

- obs["mem_feats"] ∈ ℝ^{C×L}, where number of rows are the enabled memory channels in a fixed order (e.g., [unit_stride, reuse_any, contiguity_score, stride_hist(4), reuse_count]).
Note. unit_stride uses −1 for unknown.
- obs["mem_mask"] ∈ {0,1}^{C} indicates which memory channels are active.

Computation notes (brief)
unit_stride uses element-stride × index coefficients to compute per-loop linear Δ (in elements), and checks Δ==1 with loop step=1.
reuse_any is structural (address doesn’t depend on the loop), not “reduction-only”; generalizes to many kernels (e.g., GEMM: A[i,k] independent of j ⇒ reuse on j).
Unknown/static-insufficient cases are masked/−1, never guessed.

## Scope
- **FeatureExtractor v2:** loop and accesses invariants, normalization tripcount, accesses usage.

## Out of Scope (future PRs)
- Learned encoders / GNNs.
- PBRS/value shaping (will consume v2 state).
- GPU-specific details beyond the HW embedding.
