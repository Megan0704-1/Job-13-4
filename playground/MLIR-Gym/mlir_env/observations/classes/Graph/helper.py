# mlir_env/observations/classes/Graph/helper.py
# This file has all helpers for extractor and engine

import mlir.ir as mlir_ir
from typing import Tuple

from ..utils.general_utils import (
    is_loop, parent_op,
)

def in_loop(op) -> int:
    p = parent_op(op)
    while p is not None:
        if is_loop(p):
            return 1
        p = parent_op(p)
    return 0

def get_type_info(ty: mlir_ir.Type) -> Tuple[int, int]:
    try:
        if hasattr(ty, "shape") and getattr(ty, "shape", None) is not None:
            rank = len(ty.shape)
            bits = 0
            try:
                ety = ty.element_type
                if hasattr(ety, "width"):
                    bits = int(ety.width)
            except Exception:
                pass
            return rank, bits
    except Exception:
        pass
    return 0, 0

def get_linalg_generic_info(op) -> Tuple[int, int]:
    if op.operation.name == "linalg.generic":
        it = op.operation.attributes["iterator_types"]
        s = str(it)
        return s.count("parallel"), s.count("reduction")

    return 0, 0

def has_linalg_generic_parent(op) -> bool:
    p = parent_op(op)
    return (p is not None) and (getattr(p.operation, "name", "") == "linalg.generic")

