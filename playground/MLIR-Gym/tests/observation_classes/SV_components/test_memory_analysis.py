# tests/test_memory_delta_and_masks.py
import numpy as np
import pytest

from mlir_env.observations.classes.SV.analysis.utils.memory_api import (
    compute_is_unit_stride_minor,
    compute_loop_carries_reuse_any,
    compute_contiguity_score,
    compute_reuse_count,
    compute_strided_hist,
)
from mlir_env.observations.classes.SV.analysis.utils.memory_utils import (
    ArrayLayout, AccessUse,
)

def _mk_layout(strides):
    return ArrayLayout(rank=len(strides), strides_elems=strides, unknown=any(s is None for s in strides[:-1]))

def _mk_access(A):
    # A shape: (rank, L)
    return AccessUse(kind="load", A=np.asarray(A, dtype=np.int64), has_unknown=False, memref_val=object())

def test_unit_stride_must_multiply_step():
    # rank=1, L=1; coeffs=1 -> non unit if step = 2 or delta != 1
    L = 1
    layout = _mk_layout([1])  # stride=1
    access = _mk_access([[1]])  # A[0,0]=1
    accesses = { "arr": [access] }
    layouts = { "arr": layout }
    steps = np.array([2], dtype=np.int64)  # step=2
    ivs = [0]
    iv2idx = {0:0}
    iv2steps = {0:2}

    out = compute_is_unit_stride_minor(L, accesses, layouts, ivs, iv2idx, iv2steps)
    assert out.shape == (L,)
    assert out[0] == 0,  "delta is 2, if we got 1 here means we forgot to multiply steps (2)"

def test_unknown_propagates_and_masks_contiguity():
    '''
    for i:
      for j:
        access1[i]
        access2[j]
    '''
    L = 2
    layout1 = _mk_layout([None])  # unknown
    layout2 = _mk_layout([1])     # known, stride = 1

    access1 = _mk_access([[1, 0]])   # loop0
    access2 = _mk_access([[0, 1]])   # loop1

    accesses = { "a": [access1], "b": [access2] }
    layouts  = { "a": layout1,   "b": layout2  }

    ivs = [0, 1]
    iv2idx = {0:0, 1:1}
    iv2steps = {0:1, 1:1}

    unit = compute_is_unit_stride_minor(L, accesses, layouts, ivs, iv2idx, iv2steps)
    contig = compute_contiguity_score(L, accesses, layouts, ivs, iv2idx, iv2steps, unit)

    assert unit[0] == -1, "unknown stride should set loop0 to unknown (-1)"
    assert contig[1] == 1.0, "loop1 is expects to have perfect contiguity (delta=1)"

def test_strided_hist_buckets_and_reuse_count():
    # 3 buckets (coeffs=[1,3,9]) -> should fall in bucket 0, 1, 2 (3 is unknown bucket)
    L = 3
    W = 8
    # stride = 1, delta = step * coeff
    layout = _mk_layout([1])

    A = [[1, 3, 9]]  # coeffs
    access = _mk_access(A)
    accesses = {"arr":[access]}
    layouts  = {"arr":layout}

    ivs = [0, 1, 2]
    iv2idx = {0:0, 1:1, 2:2}
    iv2steps = {0:1, 1:1, 2:1}

    hist = compute_strided_hist(L, accesses, layouts, ivs, iv2idx, iv2steps, W)
    assert hist[0,0] > 0, "loop d0 (delta = 1) should fall in the 0th bucket"
    assert hist[1,1] > 0, "loop d1 (delta = 3) should fall in the 1st bucket"
    assert hist[2,2] > 0, "loop d2 (delta = 8) should fall in the 2nd bucket"

    reuse = compute_reuse_count(L, accesses)
    assert reuse.shape == (L,)
    assert reuse.sum() == 0, "A[:,j]==0, expects no reuse here"

