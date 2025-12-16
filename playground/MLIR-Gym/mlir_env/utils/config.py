# mlir_env/config.py
import os
import json
import mlir_env
from pathlib import Path


def get_project_root():
    return os.path.dirname(mlir_env.__path__[0])


def get_env_file(env):
    file_path = json.loads(env.spec.to_json())["kwargs"]["mlir_file"]
    file_name_ext = file_path.split("/")[-1]
    file_name = file_name_ext.split(".")[0]
    return file_name


def get_common_config():
    """
    return shared configuration settings.
    """

    home = str(Path.home())
    project_root = os.environ.get("MLIR_GYM_ROOT")
    project_encoder_path = project_root + "/projects/encoder"

    llvm_build_dir = os.environ.get("LLVM_BUILD_DIR")
    compile_install_dir = llvm_build_dir + "/bin"

    configs = {
        "home": home,
        "compile_bin": compile_install_dir,
        "mlir_opt": compile_install_dir + "/mlir-opt",
        "mlir_runner": compile_install_dir + "/mlir-runner",
        "llvm_lit": llvm_build_dir + "/llvm-lit",
        "mlir_file": project_root + "/mlir_tests/tensor-matmul/test-tensor-matmul.mlir",
        "mlir_testfiles": project_root + "/mlir_tests",
        "checkpoint_path": project_root + "/checkpoints",
        "log_path": project_root + "/logs",
        "snapshots": project_root + "/snapshots",
        "llvmir_outdir": project_root + "/scripts/llvm_ir",
        "encoder_path": project_encoder_path,
    }

    to_create = ["checkpoint_path", "llvmir_outdir", "log_path", "snapshots"]

    for name, path in configs.items():
        if not os.path.exists(path) and name in to_create:
            os.makedirs(path)
            print(f"Can't find {name}\nCreating dir at {path}\n========")

    return configs


def get_lowering_dialect(*args):
    valid_entries = {
        "acc": ("acc/acc_ops.txt", "ACC"),
        "affine": ("affine/affine_ops.txt", "AFFINE"),
        "amdgpu": ("amdgpu/amdgpu_ops.txt", "AMDGPU"),
        "amx": ("amx/amx_ops.txt", "AMX"),
        "ArmSME": ("ArmSME/ArmSME_ops.txt", "ARMSME"),
        "arm_neon": ("arm_neon/arm_neon_ops.txt", "ARM_NEON"),
        "arm_sve": ("arm_sve/arm_sve_ops.txt", "ARM_SVE"),
        "arith": ("arith/arith_ops.txt", "ARITH"),
        "async": ("async/async_ops.txt", "ASYNC"),
        "builtin": ("builtin/builtin_ops.txt", "BUILTIN"),
        "bufferization": ("bufferization/bufferization_ops.txt", "BUFFERIZATION"),
        "cf": ("cf/cf_ops.txt", "CF"),
        "complex": ("complex/complex_ops.txt", "COMPLEX"),
        "emitc": ("emitc/emitc_ops.txt", "EMITC"),
        "func": ("func/func_ops.txt", "FUNC"),
        "gpu": ("gpu/gpu_ops.txt", "GPU"),
        "index": ("index/index_ops.txt", "INDEX"),
        "linalg": ("linalg/linalg_ops.txt", "LINALG"),
        "llvm": ("llvm/llvm_ops.txt", "LLVM"),
        "math": ("math/math_ops.txt", "MATH"),
        "memref": ("memref/memref_ops.txt", "MEMREF"),
        "mesh": ("mesh/mesh_ops.txt", "MESH"),
        "ml_program": ("ml_program/ml_program_ops.txt", "ML_PROGRAM"),
        "mpi": ("mpi/mpi_ops.txt", "MPI"),
        "nvgpu": ("nvgpu/nvgpu_ops.txt", "NVGPU"),
        "nvvm": ("nvvm/nvvm_ops.txt", "NVVM"),
        "pdl": ("pdl/pdl_ops.txt", "PDL"),
        "pdl_interp": ("pdl_interp/pdl_interp_ops.txt", "PDL_INTERP"),
        "polynomial": ("polynomial/polynomial_ops.txt", "POLYNOMIAL"),
        "quant": ("quant/quant_ops.txt", "QUANT"),
        "rocdl": ("rocdl/rocdl_ops.txt", "ROCDL"),
        "scf": ("scf/scf_ops.txt", "SCF"),
        "shape": ("shape/shape_ops.txt", "SHAPE"),
        "sparse_tensor": ("sparse_tensor/sparse_tensor_ops.txt", "SPARSE_TENSOR"),
        "spirv": ("spirv/spirv_ops.txt", "SPIRV"),
        "tensor": ("tensor/tensor_ops.txt", "TENSOR"),
        "tosa": ("tosa/tosa_ops.txt", "TOSA"),
        "transform": ("transform/transform_ops.txt", "TRANSFORM"),
        "ub": ("ub/ub_ops.txt", "UB"),
        "vector": ("vector/vector_ops.txt", "VECTOR"),
        "x86vector": ("x86vector/x86vector_ops.txt", "X86VECTOR"),
        "xegpu": ("xegpu/xegpu_ops.txt", "XEGPU"),
    }

    return {
        entry[0]: entry[1]
        for dialect_name, entry in valid_entries.items()
        if dialect_name in args
    }
