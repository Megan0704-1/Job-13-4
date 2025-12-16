# observations/classes/SV/analysis/utils/memory_api.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from mlir import ir as mlir_ir

from ....utils.general_utils import SUPPORTED_MEMOPS, SUPPORTED_LOADS, SUPPORTED_STORES, UNKNOWN, walk_ops_recursive
from .memory_utils import ArrayLayout, AccessUse, _layout_from_memref_type, _coeffs_from_result_expr, _delta_elements_for_loop

# -----------------------------------------------------------------------------
# helper apis for memory analysis
# -----------------------------------------------------------------------------

def collect(L: int, moduleView: "ModuleView", iv2idx: Dict[mlir.ir.Value, int] = {}):
    '''Scan moduleView, collect access information to accesses, layout information to layouts'''
    # A has a context of: which loop (depth/index) is responsible for updating this array dimension, value records the update scale.
    layouts: Dict[Any, ArrayLayout] = {}
    accesses: Dict[Any, List[AccessUse]] = {}

    for _, funcOp in moduleView.functions():
        for op in walk_ops_recursive(funcOp):
            name = op.operation.name
            if name not in SUPPORTED_MEMOPS:
                continue

            if name in SUPPORTED_LOADS:
                # %0 = affine.load %arg0[%arg3, %arg5] : memref<2048x2048xf32>
                # %0 = memref.load %arg0[%arg3, %arg5] : memref<2048x2048xf32, strided<[?, ?], offset: ?>>
                start_idx = 0
                kind = "load"
            elif name in SUPPORTED_STORES:
                # affine.store %cst, %alloc[%arg3, %arg4] : memref<2048x2048xf32>
                # memref.store %cst, %alloc[%arg3, %arg4] : memref<2048x2048xf32>
                start_idx = 1
                kind = "store"
            else:
                assert "MemoryAccessAnalyzer::_collect, find op that passed SUPPORTED_MEMOPS but not captured by memory_utils.py"

            memref_val = op.operands[start_idx] # mlir array object
            indexing_dims = op.operands[start_idx+1:]
            memref_ty = mlir_ir.MemRefType(memref_val.type)
            rank = memref_ty.rank

            # read the real StrideLayoutAttr first
            # otherwise, fall back to row-major stride calculation
            # layouts {key: MemRefType obj, value: ArrayLayout}
            if memref_val not in layouts:
                layouts[memref_val] = _layout_from_memref_type(memref_ty)

            # dim->loop mapping
            # 1) iv2idx is __post_init__, a map from iv to loop index
            # 2) iterate the dimension of array, get the iv that is indexing the dimension
            #    query iv2idx to get loop index
            # 3) get a array dimension to loop index map
            dim2idx: Dict[int, int] = {}
            for dim, v in enumerate(indexing_dims):
                if not isinstance(v, mlir_ir.BlockArgument): continue
                for iv in v.owner.arguments:
                    loop_idx = iv2idx.get(iv, None)
                    if loop_idx is not None:
                        dim2idx[dim] = loop_idx

            amap_attr = getattr(op, "map", None)
            if isinstance(amap_attr, mlir_ir.AffineMapAttr):
                amap = amap_attr.value
                n_dims = int(amap.n_dims)
                if len(indexing_dims) != n_dims or len(amap.results) != rank:
                    A = np.zeros((rank, L), dtype=np.int64)
                    accesses.setdefault(memref_val, []).append(
                        AccessUse(kind=kind, A=A, has_unknown=True, memref_val=memref_val)
                    )
                    continue

                A = np.zeros((rank, L), dtype=np.int64)
                has_unknown = False

                # loop i; loop j; loop k
                # access pattern: array[k * 2, j, k]
                # affine map: (d0, d1, d2) -> (d2 * 2, d1, d2)
                # dim2idx = (d2, d1, d2)
                # coeffs = [0, 0, 2], [0,1,0], [0,0,1] -> read as, first dim is accessed by d2 with coeff equals 2
                # resulting A matrix:
                #   A[0,d0]=0, A[1,d0]=0, A[2,d0]=0
                #   A[0,d1]=0, A[1,d1]=1, A[2,d1]=0
                #   A[0,d2]=2, A[1,d2]=0, A[2,d2]=1
                for res_dim in range(rank):
                    coeffs, unk = _coeffs_from_result_expr(amap, res_dim, n_dims)
                    has_unknown = has_unknown or unk
                    for dim, loop_idx in dim2idx.items():
                        A[res_dim, loop_idx] += int(coeffs[dim])

                accesses.setdefault(memref_val, []).append(
                    AccessUse(kind=kind, A=A, has_unknown=has_unknown, memref_val=memref_val)
                )
            else:
                # identity map
                if len(indexing_dims) != rank:
                    A = np.zeros((rank, L), dtype=np.int64)
                    accesses.setdefault(memref_val, []).append(
                        AccessUse(kind=kind, A=A, has_unknown=True, memref_val=memref_val)
                    )
                    continue

                A = np.zeros((rank, L), dtype=np.int64)
                has_unknown = False
                for res_dim in range(rank): # result dim == dim idx
                    if res_dim in dim2idx: # gives one to the indexing iv
                        loop_idx = dim2idx[res_dim]
                        A[res_dim, loop_idx] += 1

                accesses.setdefault(memref_val, []).append(
                    AccessUse(kind=kind, A=A, has_unknown=has_unknown, memref_val=memref_val)
                )

    return accesses, layouts

# answers if a loop gives unit stride to an array
def compute_is_unit_stride_minor(L:int, accesses: Dict[Any, List[AccessUse]], layouts: Dict[Any, ArrayLayout], ivs: List[Any], iv2idx: Dict[Any, int], iv2steps: Dict[Any, int]) -> np.ndarray:
    '''
    Returns:
    - U: np.ndarray with shape (N, L). N is the number of memref operations, L is the number of loop accesses.
    Rule:
    - If exist any candidate (loop access memref) that has delta = 1, val = 1
    - If exist some candidates but none of them have delta = 1, val = 0
    - If no candidate or all of them are unknown, val = -1
    '''
    U = np.full(L , 0, dtype=np.int8)
    cand = np.zeros(L, dtype=np.int8)
    unknown = np.zeros(L, dtype=np.int8)

    # 1) for each access
    for mem in accesses.keys():
        # 2) get layout from memref
        layout = layouts.get(mem) # ArrayLayout
        arrays = accesses.get(mem)

        # 3) for each loop iv
        for loopId, iv in enumerate(ivs):
            if loopId >= L: break

            # 4) get step and loop index
            step = iv2steps.get(iv, None)
            loop_idx = iv2idx.get(iv, None)
            if step is None or loop_idx is None or layout is None or layout is None:
                unknown[loopId] = 1
                continue

            # 5) for each access
            for access in arrays:
                # if access has unknown values
                if access.has_unknown:
                    unknown[loopId] = 1
                    continue

                # 6) calculate delta of element for this loop access
                delta = _delta_elements_for_loop(access, layout, loop_idx, step)

                if delta is None:
                    unknown[loopId] = 1
                    continue

                cand[loopId] = 1

                if delta == 1:
                    U[loopId] = 1
                    break

    # unknown is marked with -1: no cand / all unknown
    U = np.where((U == 0) & ((cand == 0) | (unknown == 1)), UNKNOWN, U)
    return U

# compute if loop carries over values across iterations
def compute_loop_carries_reuse_any(L: int, accesses: Dict[Any, List[AccessUse]]) -> np.ndarray:
    reuse = np.zeros(L, dtype=np.int8)
    for _, arrays in accesses.items():
        for access in arrays:
            # np.all(access.A == 0, axis=0) -> L
            reuse = np.maximum(reuse, np.all(access.A == 0, axis=0).astype(np.int8))
    return reuse

# bonus analysis
def compute_contiguity_score(L: int, accesses: Dict[Any, List[AccessUse]],
                             layouts: Dict[Any, ArrayLayout], ivs: List[Any], iv2idx: Dict[Any, int], iv2steps: Dict[Any, int],
                             is_unit: np.ndarray) -> np.ndarray:
    '''
    Contiguity score is defined as 1/(1 + |delta - 1|)
    Get the best score of a loop (best vectorization score)
    '''
    best = np.zeros(L, dtype=np.float32)

    for mem in accesses.keys():
        layout = layouts.get(mem)
        arrays = accesses.get(mem)

        for loopId, iv in enumerate(ivs):
            if loopId >= L: break

            step = iv2steps.get(iv, None)
            loop_idx = iv2idx.get(iv, None)

            for access in arrays:
                delta = _delta_elements_for_loop(access, layout, loop_idx, step)

                if delta is None or access.has_unknown or delta == 0:
                    score = 0.0
                else:
                    if delta == 1:
                        score = 1.0
                    else:
                        score = 1.0 / (1.0 + abs(delta - 1.0))

                best[loopId] = max(best[loopId], score)
    best = np.where(is_unit == 1, 1.0, best)
    return best

# bonus analysis
def compute_reuse_count(L: int, accesses: Dict[Any, List[AccessUse]]) -> np.ndarray:
    '''
    Reuse counter for # of memrefs being reused by a loop.
    Returns (L, )
    '''
    cnt = np.zeros(L, dtype=np.int8)
    for _, arrays in accesses.items():
        col_zero_any = np.zeros(L, dtype=np.int8)
        for access in arrays:
            col_zero_any = np.maximum(col_zero_any, np.all(access.A == 0, axis=0).astype(np.int8))
        cnt += col_zero_any
    return cnt

# bonus analysis
def compute_strided_hist(L: int, accesses: Dict[Any, List[AccessUse]],
                        layouts: Dict[Any, ArrayLayout], ivs: List[Any], iv2idx: Dict[Any, int], iv2steps: Dict[Any, int], vector_W: int) -> np.ndarray:
    """4 bucket: =1、2..W、>W、unknown。"""
    hist = np.zeros((L, 4), dtype=np.int8)
    for mem in accesses.keys():
        layout = layouts.get(mem)
        arrays = accesses.get(mem)
        for access in arrays:
            for loopId, iv in enumerate(ivs):
                if loopId >= L: break

                step = iv2steps.get(iv, None)
                loop_index = iv2idx.get(iv, None)
                delta = _delta_elements_for_loop(access, layout, loop_index, step)
                if delta is None or delta==0: continue

                if access.has_unknown:
                    hist[loopId, 3] += 1
                else:
                    if delta == 1:
                        hist[loopId, 0] += 1
                    elif 2 <= delta <= vector_W:
                        hist[loopId, 1] += 1
                    elif delta > vector_W:
                        hist[loopId, 2] += 1
                    else:
                        hist[loopId, 3] += 1
    return hist
