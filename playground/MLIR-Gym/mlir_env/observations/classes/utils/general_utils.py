# utils/general_utils.py
# This class is important for parsing and preserving loop shape information
# As IR gets lowered, mlir introduces SSA values to replace constant lb, ub, strides

from __future__ import annotations
from typing import Optional, Tuple, List, Union
import math
from mlir import ir as mlir_ir
import hashlib

# --------- global variables ----------
POW2 = lambda x: (x is not None) and x > 0 and (x & (x - 1)) == 0
LOOP_OP_NAMES = ("affine.for", "scf.for", "scf.parallel", "affine.parallel")
SUPPORTED_LOADS  = {"affine.load", "memref.load"}
SUPPORTED_STORES = {"affine.store", "memref.store"}
SUPPORTED_MEMOPS = SUPPORTED_LOADS | SUPPORTED_STORES

DYNAMIC_VAL = mlir_ir.ShapedType.get_dynamic_size()
UNKNOWN = -1

def get_id(scopeName: str, op: mlir_ir.Operation, mode: str = "normal") -> Union[int, str]:
    '''
    return hash of scopeName + op, hash result type depends on mode.
    '''
    opName = str(op.name)
    opLoc = str(op.location)
    raw = scopeName + opLoc + opName
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    hi = hashlib.blake2b(raw.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(hi, byteorder="little") if mode == "int" else h


def is_loop(op: mlir_ir.Operation) -> bool:
    return op.operation.name in LOOP_OP_NAMES


def parent_op(op: mlir_ir.Operation) -> mlir_ir.Operation | None:
    parent = op.operation.parent
    return parent

# visit every op in the given region
def walk_ops_recursive(root: mlir_ir.Operation) -> Iterator[mlir_ir.Operation]:
    """
    Pre-order walk of an MLIR operation tree.
    Return each operation, recursively visits all nested regions/blocks/ops.
    Yields:
      mlir_ir.Operation instances in pre-order.
    """
    def _go(op: mlir_ir.Operation) -> Iterator[mlir_ir.Operation]:
        # 1) visit self
        yield op

        # 2) explicit regions -> blocks -> operations traversal
        for region in op.operation.regions:
            # region may be empty
            for block in region.blocks:
                for child in list(block.operations):
                    yield from _go(child)

    yield from _go(root)

def get_loop_iv(op: mlir_ir.Operation) -> Tuple[mlir_ir.BlockArgument,...]:
    '''Return induction variables of a loop op'''
    if not is_loop(op): return None

    name = op.name
    if "parallel" in name:
        return list(op.regions[0].blocks[0].arguments)
    return [op.induction_variable]

def const_index_like(val: mlir_ir.Value) -> int | None:
    """Resolve an index-like integer from small def-use chains."""
    assert isinstance(val, mlir_ir.Value), f"const_index_like expects input to has type: mlir.ir.Value, got {type(val)} instead."

    def _go(v: mlir_ir.Value) -> int | None:
        op = v.owner
        while not hasattr(op, "operation"):
            op = op.owner

        if op is None:
            return None

        name = op.operation.name

        # memref.dim would not be a constant bound
        if name == "memref.dim": return None

        # arith.constant
        if name == "arith.constant" and "value" in op.attributes:
            attr = op.attributes["value"]

            if isinstance(attr, mlir_ir.IntegerAttr) or isinstance(
                attr, mlir_ir.IndexAttr
            ):
                return int(attr.value)
            elif isinstance(attr, mlir_ir.FloatAttr):
                return float(attr.value)

        # simple casts/extensions that keep the scalar value
        if name in ("arith.index_cast", "arith.extsi", "arith.extui", "arith.trunci"):
            return _go(op.operands[0])

        # affine.apply of a constant AffineMap
        if name == "affine.apply" and "map" in op.attributes:
            amap = op.attributes["map"]
            if isinstance(amap, mlir_ir.AffineMapAttr):
                m = amap.value
                if len(m.results) == 1 and m.n_dims == 0 and m.n_symbols == 0:
                    expr = m.results[0]
                    if isinstance(expr, mlir_ir.AffineConstantExpr):
                        return int(expr.value)

        return None

    return _go(val)


# ---- helpers ----

def _get_const_from_affine_map(amap: mlir_ir.AffineMap, op: mlir_ir.Operation | None, key: str | None, dim: int = 0):
    assert isinstance(amap, mlir_ir.AffineMap), f"_get_const_from_affine_map expects input to have type mlir.ir.AffineMap, get {type(amap)} instead."

    # const bounds
    if len(amap.results) == 1 and amap.n_dims == 0 and amap.n_symbols == 0:
        expr = amap.results[0]
        if mlir_ir.AffineConstantExpr.isinstance(expr):
            return int(mlir_ir.AffineConstantExpr(expr).value)

    # const symbol bounds
    if key and op:
        if len(amap.results) == 1 and amap.n_dims == 0 and amap.n_symbols >=1:
            boundOps = getattr(op, key)
            return sum([const_index_like(operand) for operand in boundOps if const_index_like(operand) is not None])

    return None

def _affine_for_const_bound(op: mlir_ir.Operation, which: str) -> int | None:
    """
    affine.for has attributes lower_bound / upper_bound as AffineMapAttr.
    Returns int only when it's a constant map: (d0)[s0] -> (cst).
    """
    assert op.operation.name == "affine.for"
    keyMap = {"lb": "lowerBoundMap", "ub": "upperBoundMap"}[which]
    keyOp = {"lb": "lowerBoundOperands", "ub": "upperBoundOperands"}[which]

    amap = getattr(op, keyMap)
    if amap is None or not isinstance(amap, mlir_ir.AffineMapAttr):
        return None

    m = amap.value
    return [_get_const_from_affine_map(m, op, keyOp, 0)]


def _scf_for_const_bound(op: mlir_ir.Operation, which: str) -> int | None:
    """
    scf.for operands: [lb, ub, step].
    default returns lower constant bounds.
    """
    assert op.operation.name == "scf.for"

    match which:
        case "lb":
            idx = 0
        case "ub":
            idx = 1
        case _:
            idx = 0

    if len(op.operands) <= idx:
        return None
    return [const_index_like(op.operands[idx])]


def _affine_for_const_step(op: mlir_ir.Operation) -> int | None:
    """affine.for step is an IntegerAttr or a SSA value"""
    assert op.operation.name == "affine.for"

    step = getattr(op, "step")

    if hasattr(step, "value"):
        return [getattr(step, "value")] if step is not None else None
    return [const_index_like(step)]

def _scf_for_const_step(op: mlir_ir.Operation) -> int | None:
    """scf.for step is operand #2."""
    assert op.operation.name == "scf.for"

    if len(op.operands) < 3:
        return None
    return [const_index_like(op.operands[2])]

def _affine_parallel_const_step(op: mlir_ir.Operation) -> int | None:
    '''Affine steps are constant by default'''
    assert op.operation.name == "affine.parallel"

    steps = getattr(op, "steps")
    const_steps = []
    for s in steps:
        if isinstance(s, mlir_ir.IntegerAttr) or isinstance(s, mlir_ir.IndexAttr):
            const_steps.append(int(s.value))
    return const_steps if const_steps != [] else None


def _scf_parallel_const_step(op: mlir_ir.Operation) -> int | None:
    assert op.operation.name == "scf.parallel"

    steps = getattr(op, "step")
    const_steps = [const_index_like(s) for s in steps]
    return const_steps if const_steps != [] else None

def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b

def _affine_parallel_bounds(op: mlir_ir.Operation) -> Tuple[List[int], List[int]]:
    assert op.operation.name == "affine.parallel"

    map_operands = op.mapOperands
    lb_map = op.lowerBoundsMap.value
    ub_map = op.upperBoundsMap.value

    assert len(lb_map.results) == len(ub_map.results), "unaligned parallel affine op."

    result_dims = len(lb_map.results)
    lb_dims = lb_map.n_dims
    ub_dims = ub_map.n_dims

    lb_operands = list(map_operands[:lb_dims])
    ub_operands = list(map_operands[lb_dims:])

    # TODO(megan.kuo) simply copy broadcast,
    # need to loop through affine dim expr to actually map from input to results
    # note. affine map apply is not currently available in python bindings.
    if lb_dims < result_dims:
        left = result_dims - lb_dims
        for i in range(left):
            lb_operands.append(lb_operands[i%lb_dims])
    if ub_dims < result_dims:
        left = result_dims - ub_dims
        for i in range(left):
            ub_operands.append(ub_operands[i%ub_dims])

    lbs, ubs = [], []
    for lb_val, ub_val in zip(lb_operands, ub_operands):
        lbs.append(const_index_like(lb_val))
        ubs.append(const_index_like(ub_val))

    return lbs, ubs

def _tripcount(
    lb: int | None, ub: int | None, step: int | None
) -> int | None:
    if lb is None or ub is None or step is None or step <= 0:
        return None
    span = ub - lb
    if span <= 0:
        return 0
    return _ceildiv(span, step)


def _get_parallel_segments(op: mlir_ir.Operation) -> tuple[list[mlir_ir.Value], list[mlir_ir.Value], list[mlir_ir.Value]]:
    """
    affine.parallel: lowerBoundMap, upperBoundMap, steps
    scf.parallel: lowerBound, upperBound step
    Return (lbs, ubs, steps).
    Note. lbs[i], ubs[i] and steps[i] compose the i loop.
    """
    name = op.name

    lbs, ubs, steps = [], [], []

    # 1) get a list of ir values for bounds
    match name:
        case "affine.parallel":
            lbs, ubs = _affine_parallel_bounds(op)
            steps = _affine_parallel_const_step(op)
        case "scf.parallel":
            lb_vals = list(op.lowerBound)
            ub_vals = list(op.upperBound)
            step_vals = list(op.step)
            #2) apply const_index_like trace each value
            for lv,uv,sv in zip(lb_vals, ub_vals, step_vals):
                lbs.append(const_index_like(lv))
                ubs.append(const_index_like(uv))
                steps.append(const_index_like(sv))

    return lbs, ubs, steps
