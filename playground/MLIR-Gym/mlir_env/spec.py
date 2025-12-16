from enum import Enum


class Observations(Enum):
    OP_CNT = "OpCountState"
    IR_STR = "IRStringState"


class Rewards(Enum):
    BYTE_SIZE = "ByteSizeReward"
    IR_LEN = "IRLenReward"
    LLVM_INST_CNT = "LLVMInstCountReward"


class Actions(Enum):
    # RemoveDeadValues = "--remove-dead-values"
    # SCCP = "--sccp"
    # SnapshotOpLocations = "--snapshot-op-locations"
    # SROA = "--sroa"
    # StripDebugInfo = "--strip-debuginfo"
    # TopologicalSort = "--topological-sort"
    # ExpandStridedMetadata = "--expand-strided-metadata"

    # Affine passes
    # AffineDataCopyGenerate = "--affine-data-copy-generate"
    # AffineExpandIndexOps = "--affine-expand-index-ops"
    # AffineLoopCoalescing = "--affine-loop-coalescing"
    # AffineLoopFusion = "--affine-loop-fusion"
    # AffineLoopInvariantCodeMotion = "--affine-loop-invariant-code-motion"
    # AffineLoopNormalize = "--affine-loop-normalize"
    # AffineLoopTile = "--affine-loop-tile"
    # AffineLoopUnroll = "--affine-loop-unroll"
    # AffineLoopUnrollJam = "--affine-loop-unroll-jam"
    # AffineParallelize = "--affine-parallelize"
    # AffinePipelineDataTransfer = "--affine-pipeline-data-transfer"
    # AffineScalRep = "--affine-scalrep"
    # AffineSimplifyStructures = "--affine-simplify-structures"
    # AffineSuperVectorize = "--affine-super-vectorize"

    # Arith passes
    # ArithEmulateUnsupportedFloats = "--arith-emulate-unsupported-floats"
    # ArithEmulateWideInt = "--arith-emulate-wide-int"
    # ArithExpand = "--arith-expand"
    # ArithIntNarrowing = "--arith-int-range-narrowing"
    # ArithUnsignedWhenEquivalent = "--arith-unsigned-when-equivalent"

    # Async passes
    # AsyncFuncToAsyncRuntime = "--async-func-to-async-runtime"
    # AsyncParallelFor = "--async-parallel-for"
    # AsyncRuntimePolicyBasedRefCounting = "--async-runtime-policy-based-ref-counting"
    # AsyncRuntimeRefCounting = "--async-runtime-ref-counting"
    # AsyncRuntimeRefCountingOpt = "--async-runtime-ref-counting-opt"
    # AsyncToAsyncRuntime = "--async-to-async-runtime"

    # Bufferization passes
    # BufferDeallocation = "--buffer-deallocation"
    # BufferDeallocationSimplification = "--buffer-deallocation-simplification"
    # BufferHoisting = "--buffer-hoisting"
    # BufferLoopHoisting = "--buffer-loop-hoisting"
    # BufferResultsToOutParams = "--buffer-results-to-out-params"
    # BufferizationLowerDeallocations = "--bufferization-lower-deallocations"

    # Canonicalization and Simplification
    # Canonicalize = "--canonicalize"
    # CSE = "--cse"
    # DuplicateFunctionElimination = "--duplicate-function-elimination"
    # EliminateEmptyTensors = "--eliminate-empty-tensors"
    # EnsureDebugInfoScopeOnLLVMMFunc = "--ensure-debug-info-scope-on-llvm-func"
    # ExpandRealloc = "--expand-realloc"
    # FoldMemrefAliasOps = "--fold-memref-alias-ops"
    # FoldTensorSubsetOps = "--fold-tensor-subset-ops"

    # Conversion passes
    # ConvertAffineForToGPU = "--convert-affine-for-to-gpu"
    # ConvertAMDGPUToROCDL = "--convert-amdgpu-to-rocdl"
    # ConvertArithToLLVM = "--convert-arith-to-llvm"
    # ConvertArithToSPIRV = "--convert-arith-to-spirv"
    # ConvertAsyncToLLVM = "--convert-async-to-llvm"
    # ConvertBufferizationToMemRef = "--convert-bufferization-to-memref"
    # ConvertCFToLLVM = "--convert-cf-to-llvm"
    # ConvertCFToSPIRV = "--convert-cf-to-spirv"
    # ConvertComplexToLibM = "--convert-complex-to-libm"
    # ConvertComplexToLLVM = "--convert-complex-to-llvm"
    # ConvertComplexToSPIRV = "--convert-complex-to-spirv"
    # ConvertComplexToStandard = "--convert-complex-to-standard"
    # ConvertFuncToLLVM = "--convert-func-to-llvm"
    # ConvertFuncToSPIRV = "--convert-func-to-spirv"
    # ConvertGPUToNVVM = "--convert-gpu-to-nvvm"
    # ConvertGPUToROCDL = "--convert-gpu-to-rocdl"
    # ConvertGPUToSPIRV = "--convert-gpu-to-spirv"
    # ConvertIndexToLLVM = "--convert-index-to-llvm"
    # ConvertLinalgToAffineLoops = "--convert-linalg-to-affine-loops"
    # ConvertLinalgToLoops = "--convert-linalg-to-loops"
    # ConvertLinalgToParallelLoops = "--convert-linalg-to-parallel-loops"
    # ConvertMathToLibM = "--convert-math-to-libm"
    # ConvertMathToLLVM = "--convert-math-to-llvm"
    # ConvertMemRefToSPIRV = "--convert-memref-to-spirv"
    # FinalizeMemRefToLLVM = "--finalize-memref-to-llvm"
    # ConvertNVVMToLLVM = "--convert-nvvm-to-llvm"
    # ConvertSCFToCF = "--convert-scf-to-cf"
    # ConvertSCFToOpenMP = "--convert-scf-to-openmp"
    # ConvertSCFToSPIRV = "--convert-scf-to-spirv"
    # ConvertSPIRVToLLVM = "--convert-spirv-to-llvm"
    # ConvertTensorToLinalg = "--convert-tensor-to-linalg"
    # ConvertTensorToSPIRV = "--convert-tensor-to-spirv"
    # ConvertToLLVM = "--convert-to-llvm"
    # ConvertUBToLLVM = "--convert-ub-to-llvm"
    # ConvertUBToSPIRV = "--convert-ub-to-spirv"
    # ConvertVectorToARMNeon = "--convert-vector-to-arm-neon"
    # ConvertVectorToLLVM = "--convert-vector-to-llvm"
    # ConvertVectorToSPIRV = "--convert-vector-to-spirv"

    # Transform passes
    # TransformInterpreter = "--transform-interpreter"
    # TestTransformDialectEraseSchedule = "--test-transform-dialect-erase-schedule"

    # Control Flow passes
    # ControlFlowSink = "--control-flow-sink"
    # LiftCFToSCF = "--lift-cf-to-scf"

    # GPU Passes
    # GPUAsyncRegion = "--gpu-async-region"
    # GPUDecomposeMemRefs = "--gpu-decompose-memrefs"
    # GPUKernelOutlining = "--gpu-kernel-outlining"
    # GPUMapParallelLoops = "--gpu-map-parallel-loops"
    # GPUModuleToBinary = "--gpu-module-to-binary"
    # GPUToLLVM = "--gpu-to-llvm"

    # Linalg passes
    # LinalgDetensorize = "--linalg-detensorize"
    # LinalgFuseElementwiseOps = "--linalg-fuse-elementwise-ops"
    # LinalgGeneralizeNamedOps = "--linalg-generalize-named-ops"
    # LinalgInlineScalarOperands = "--linalg-inline-scalar-operands"
    # LinalgNamedOpConversion = "--linalg-named-op-conversion"

    # Loop optimization passes
    # LoopInvariantCodeMotion = "--loop-invariant-code-motion"
    # SCFForLoopCanonicalization = "--scf-for-loop-canonicalization"
    # SCFForLoopPeeling = "--scf-for-loop-peeling"
    # SCFForLoopSpecialization = "--scf-for-loop-specialization"
    # SCFParallelLoopFusion = "--scf-parallel-loop-fusion"
    # SCFParallelLoopSpecialization = "--scf-parallel-loop-specialization"
    # SCFParallelLoopTiling = "--scf-parallel-loop-tiling"

    # Memory optimization passes
    # Mem2Reg = "--mem2reg"
    # NormalizeMemRefs = "--normalize-memrefs"
    # OneShotBufferize = "--one-shot-bufferize"
    # OneShotBufferize_FunctionBoundaries = "--one-shot-bufferize='bufferize-function-boundaries'"
    # PromoteBuffersToStack = "--promote-buffers-to-stack"

    # Sparse tensor passes
    # SparseBufferRewrite = "--sparse-buffer-rewrite"
    # SparseStorageSpecifierToLLVM = "--sparse-storage-specifier-to-llvm"
    # SparseTensorCodegen = "--sparse-tensor-codegen"
    # SparseTensorConversion = "--sparse-tensor-conversion"
    # SparseVectorization = "--sparse-vectorization"

    # Vector passes
    # VectorToLLVM = "--convert-vector-to-llvm"
    # VectorToSPIRV = "--convert-vector-to-spirv"

    # Miscellaneous
    # Inline = "--inline"
    # SymbolDCE = "--symbol-dce"
    # SymbolPrivatize = "--symbol-privatize"
    # ReconcileUnrealizedCasts = "--reconcile-unrealized-casts"

    # Pipelines
    # BufferDeallocationPipeline = "--buffer-deallocation-pipeline"
    # SparseAssembler = "--sparse-assembler"

    # Lowerings
    # LowerAffine = "--lower-affine"
    # LowerHostToLLVM = "--lower-host-to-llvm"

    # Termination - mlirGym specific pass
    TERMINATION = ""
    SUCCESS1 = '-transform-interpreter -test-transform-dialect-erase-schedule -one-shot-bufferize="bufferize-function-boundaries" -convert-linalg-to-loops -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-scf-to-cf --finalize-memref-to-llvm -convert-func-to-llvm -convert-cf-to-llvm -reconcile-unrealized-casts'
    FAIL1 = "--convert-vector-to-spirv"
    FAIL2 = "--sparse-tensor-codege"


class ActionDescriptions(Enum):
    pass


class Penalty(Enum):
    DependencyNotFound = -18

    Parsing = -15
    Translation = -13.5
    FileIO = -2.9
    CompileTime = -15.4
    BinaryType = -1.8
    ExecutionTime = -5.4
    Accuracy = -2.3


class Success(Enum):
    Parsing = 20
    Translation = 6.6
    FileIO = 8.25
    CompileTime = 27
    BinaryType = 12.5
    ExecutionTime = 50
    Accuracy = 150
