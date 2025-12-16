# anatysis/memory_api.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Any
import numpy as np

from .loop_utils import LoopInfo
from ....utils.general_utils import DYNAMIC_VAL

from mlir import ir as mlir_ir

# ------------- dataclasses ---------------

@dataclass
class ArrayLayout:
    '''data class for an array'''
    rank: int
    strides_elems: List[Optional[int]] # unit: elements
    unknown: bool = False

@dataclass
class AccessUse:
    '''Answers how is an array being accessed by a loop construct'''
    kind: str # 'load' or 'store'
    A: np.ndarray # (rank, N) dims × loops(band)
    has_unknown: bool
    memref_val: Any

@dataclass
class MemoryFeatures:
    is_unit_stride_minor: np.ndarray        # [L] in {1,0,-1}
    loop_carries_reuse_any: np.ndarray      # [L] in {1,0}
    contiguity_score: Optional[np.ndarray] = None   # [L] in [0,1]
    reuse_count: Optional[np.ndarray] = None        # [L]
    strided_hist: Optional[np.ndarray] = None        # [L,4] (=1, 2..W, >W, unknown)

@dataclass
class FeatureMask:
    unit_known: np.ndarray
    contig_known: np.ndarray
    stride_known: np.ndarray


@dataclass
class MemInfoOut:
    feats: MemoryFeatures
    masks: FeatureMask

# ------------- helpers ---------------

def _layout_from_memref_type(ty: mlir_ir.MemRefType) -> ArrayLayout:
    """
    Prefer true StridedLayoutAttr when present (element strides),
    otherwise fall back to row-major deduction from static shape.
    Dynamic dims produce unknown strides (None) above that dimension.
    """
    # Try true strided layout first
    lay = ty.layout
    rank = ty.rank
    shape = list(ty.shape)

    StridedLayoutAttr = getattr(mlir_ir, "StridedLayoutAttr", None)
    if StridedLayoutAttr.isinstance(lay):
        stride_lay = StridedLayoutAttr(lay)
        # In Python bindings, lay.strides is typically an ArrayAttr of IntegerAttr
        strides_raw: List[int] = [int(s) for s in stride_lay.strides]

        # make sure every stride is a static value
        if all([s != DYNAMIC_VAL for s in strides_raw]):
            strides: List[Optional[int]] = [None if s < 0 else int(s) for s in strides_raw]
            return ArrayLayout(rank=rank, strides_elems=strides, unknown=any(s is None for s in strides))

    # if not, fallback: deduce row-major strides from static shape
    strides: List[Optional[int]] = [None] * rank
    prod: Optional[int] = 1
    for d in range(rank - 1, -1, -1):
        if prod is not None:
            strides[d] = prod
        sz = shape[d]
        if sz == mlir_ir.ShapedType.get_dynamic_size():
            prod = None
        else:
            prod = None if prod is None else prod * int(sz)
    # Unknown if any but the minor stride is None
    return ArrayLayout(rank=rank, strides_elems=strides, unknown=any(s is None for s in strides[:-1]))


def _coeffs_from_result_expr(amap: mlir_ir.AffineMap, dim: int, n_dims: int) -> Tuple[List[int], bool]:
    """
    Extract per-dim integer coefficients for the 'dim'-th result of an AffineMap.
    e.g.,
    AffineMap((d0, d1) -> (d0*3, d1-2)), this function returns [3, 1]
    """
    # Import Affine*Expr classes locally to avoid failing on old bindings at import time
    from mlir.ir import AffineAddExpr, AffineMulExpr, \
                        AffineDimExpr, AffineSymbolExpr, AffineConstantExpr

    expr = amap.results[dim]
    coeffs = [0] * n_dims
    unknown = False

    def visit(e, scale=1):
        nonlocal unknown
        if AffineDimExpr.isinstance(e):
            dimE = AffineDimExpr(e)
            coeffs[int(dimE.position)] += int(scale)
        # don't care, constant dim does not contribute to scale strides.
        elif AffineSymbolExpr.isinstance(e) or AffineConstantExpr.isinstance(e):
            return
        elif AffineAddExpr.isinstance(e):
            addE = AffineAddExpr(e)
            visit(addE.lhs, scale)
            visit(addE.rhs, scale)
        elif isinstance(e, AffineMulExpr):
            mulE = AffineMulExpr(e)
            # const * linear
            if isinstance(mulE.lhs, AffineConstantExpr):
                visit(mulE.rhs, scale * int(mulE.lhs.value))
            elif isinstance(mulE.rhs, AffineConstantExpr):
                visit(mulE.lhs, scale * int(mulE.rhs.value))
            else:
                unknown = True
        else:
            # min/max/mod/div/ceildiv/floordiv and others → unknown
            unknown = True

    visit(expr, 1)
    return coeffs, unknown


# TODO(megan.kuo) add selection mechanism to filter memory ops
def _select_candidates(uses: List[AccessUse], policy: str) -> List[AccessUse]:
    '''not sure about this, maybe we should set a ARRAY limit? it is not fact that every loop has manipulation of an array.'''
    if policy == "store_first":
        stores = [u for u in uses if u.kind == "store"]
        return stores if stores else uses
    elif policy == "all":
        return uses
    else:  # "any"
        return uses


def _delta_elements_for_loop(access: AccessUse, layout: ArrayLayout, loop_idx: int, loop_step: Optional[int]) -> Optional[int]:
    '''compute the linear offset of every iteration of loop `loop_index`'''
    if loop_step is None or loop_step < 0:
        return None
    if layout.unknown:
        return None

    # 1) get stride from layout
    S = layout.strides_elems
    acc = 0
    # 2) iterate through memref type output dimensions
    for d in range(access.A.shape[0]):
        # 3) get updating scale
        coeff = int(access.A[d, loop_idx])
        if coeff == 0:
            continue
        # 4) get the dimension stride
        assert S[d] != 0, "Bug: stride should never be 0."
        stride = S[d]
        if stride is None:
            return None
        # 5) accumulate
        acc += stride * coeff
    # 6) finally multiply with step (offset for the next iteration)
    return abs(acc) * int(loop_step)
