# mlir_env/actions/space.py
from mlir_env.core.immutable import Phase, Requirement
from mlir_env.actions.action import Action


def get_action_space_default():
    return [
        # 0
        Action("canonicalize", "func", Requirement()),
        # 1
        Action("cse", "func", Requirement()),
        # 2
        Action(
            "linalg-generalize-named-ops,linalg-fuse-elementwise-ops",
            "func",
            Requirement(all_of=Phase.TENSOR | Phase.LINALG),
        ),
        # 3
        Action(
            "linalg-generalize-named-ops,linalg-fold-into-elementwise",
            "func",
            Requirement(all_of=Phase.TENSOR | Phase.LINALG),
        ),
        # 4
        Action(
            "one-shot-bufferize{bufferize-function-boundaries}",
            "module",
            Requirement(all_of=Phase.TENSOR, none_of=Phase.MEMREF),
        ),
        # 5
        Action(
            "convert-linalg-to-loops",
            "func",
            Requirement(all_of=Phase.LINALG | Phase.MEMREF),
        ),
        # 6
        Action(
            "convert-linalg-to-affine-loops",
            "func",
            Requirement(all_of=Phase.LINALG | Phase.MEMREF),
        ),
        # 7
        Action(
            "convert-vector-to-scf",
            "func",
            Requirement(all_of=Phase.VECTOR, none_of=Phase.LLVM),
        ),
        # 8
        Action(
            "convert-scf-to-cf",
            "func",
            Requirement(all_of=Phase.SCF, none_of=Phase.LLVM),
        ),
        # 9
        Action(
            "lower-affine",
            "module",
            Requirement(all_of=Phase.AFFINE, none_of=Phase.LLVM),
        ),
        # 10
        Action(
            "affine-loop-tile{tile-size=2}",
            "func",
            Requirement(all_of=Phase.AFFINE | Phase.MEMREF),
        ),
        # 11
        Action(
            "convert-math-to-llvm",
            "module",
            Requirement(all_of=Phase.MEMREF, any_of=Phase.MATH, none_of=Phase.TENSOR),
        ),
        # 12
        Action(
            "linalg-generalize-named-ops,convert-arith-to-llvm",
            "module",
            Requirement(all_of=Phase.MEMREF, any_of=Phase.ARITH, none_of=Phase.TENSOR),
        ),
        # 13
        Action(
            "convert-vector-to-llvm",
            "module",
            Requirement(all_of=Phase.VECTOR, none_of=Phase.TENSOR),
        ),
        # 14
        Action(
            "convert-func-to-llvm",
            "module",
            Requirement(all_of=Phase.FUNC, none_of=Phase.TENSOR),
        ),
        # 15
        Action(
            "convert-cf-to-llvm",
            "module",
            Requirement(all_of=Phase.MEMREF, any_of=Phase.CF, none_of=Phase.TENSOR),
        ),
        # 16
        Action(
            "finalize-memref-to-llvm",
            "module",
            Requirement(all_of=Phase.MEMREF | Phase.LLVM, none_of=Phase.TENSOR),
        ),
        # 17
        Action("reconcile-unrealized-casts", "module", Requirement(any_of=Phase.LLVM)),
        # 18
        Action("EVAL", "env", Requirement()),
    ]
