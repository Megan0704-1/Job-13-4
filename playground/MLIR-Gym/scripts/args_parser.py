import os
import argparse


def get_all_mlir_tests(tests_path):
    mlir_testfiles = []
    for root, _, files in os.walk(tests_path):
        for filename in files:
            if filename.endswith(".mlir"):
                mlir_testfiles.append(os.path.join(root, filename))

    return mlir_testfiles


def parse_args():
    parser = argparse.ArgumentParser("parser for mlir-v0")
    parser.add_argument(
        "-f",
        "--mlir-file",
        type=str,
        default="/root/compiler_gym/mlir_playground/tests/loop_fusion.mlir",
        help="The mlir file to be optimized",
    )
    parser.add_argument(
        "-s",
        "--max-steps",
        type=int,
        default=100,
        help="The maximum number of steps (passes) to run the environment",
    )
    parser.add_argument(
        "--compiler",
        type=str,
        default="/root/llvm-project/build/bin/mlir-opt",
        help="The mlir-opt compiler path",
    )
    parser.add_argument(
        "-e",
        "--episode",
        type=int,
        default=int(1e3),
        help="The number of episode running.",
    )
    parser.add_argument(
        "-oM",
        "--observation_mode",
        type=str,
        default="IR_STR",
        help="The observation mode for mlir gym. Options:[OP_CNT, IR_STR(default)]",
    )
    parser.add_argument(
        "-rM",
        "--reward_mode",
        type=str,
        default="LLVM_INST_CNT",
        help="The observation mode for mlir gym. Options:[BYTE_SIZE, IR_LEN, LLVM_INST_CNT(default)]",
    )
    return parser.parse_args()


def parse_dialects():
    dialect_name = [
        "acc",
        "affine",
        "amdgpu",
        "amx",
        "ArmSME",
        "arm_neon",
        "arm_sve",
        "arith",
        "async",
        "builtin",
        "bufferization",
        "cf",
        "complex",
        "emitc",
        "func",
        "gpu",
        "index",
        "linalg",
        "llvm",
        "math",
        "memref",
        "mesh",
        "ml_program",
        "mpi",
        "nvgpu",
        "nvvm",
        "pdl",
        "pdl_interp",
        "polynomial",
        "quant",
        "rocdl",
        "scf",
        "shape",
        "sparse_tensor",
        "spirv",
        "tensor",
        "tosa",
        "transform",
        "ub",
        "vector",
        "x86vector",
        "xegpu",
    ]
    return dialect_name
