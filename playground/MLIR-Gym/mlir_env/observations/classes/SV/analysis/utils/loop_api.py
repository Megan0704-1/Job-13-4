# analysis/loops_api.py
from __future__ import annotations
from typing import List, Optional, Tuple
import math
from collections import deque
import numpy as np
from mlir import ir as mlir_ir

from .loop_utils import ScalarDimResolver, LoopInfo

from ....utils.general_utils import POW2, const_index_like, is_loop, parent_op, walk_ops_recursive, get_loop_iv

def band_from_innermost(loop_op: mlir_ir.Operation) -> list[LoopInfo]:
    """Get a band of loops, expecting one outermost loop Op as input"""
    # climb to the outermost consecutive loop
    top = loop_op
    while True:
        p = parent_op(top)
        if p is None or not is_loop(p):
            break
        top = p

    band = []
    cur = top
    resolver = ScalarDimResolver()

    # collect outer to inner loops
    q = deque()
    q.append(cur)
    visited = set()

    while len(q):
        cur = q.popleft()
        if not is_loop(cur):
            continue
        if cur in visited:
            continue
        band.append(cur)
        visited.add(cur)

        children = list(walk_ops_recursive(cur))
        if any([is_loop(child) for child in children]):
            for child in children:
                if is_loop(child):
                    q.append(child)
        else:
            break

    return [
        LoopInfo(
            op=o,
            iv=get_loop_iv(o),
            step=resolver.step_of(o),
            properties=resolver.try_tripcount(o),
            kind=o.operation.name,
            index=i,
        )
        for i, o in enumerate(band)
    ]


def extract_loop_features(infos: list[LoopInfo]) -> np.ndarray:
    """Construct loop feature by input infos, expecting 1 loop construct as input"""
    N = len(infos)
    feats = np.zeros((N, 4), dtype=np.float32)

    for j, L in enumerate(infos):
        tc, succ, pow2 = L.properties
        feats[j, 0] = 0.0 if not succ else math.log1p(float(tc))
        feats[j, 1] = 1.0 if pow2 else 0.0
        feats[j, 2] = (
            0.0 if N <= 1 else float(L.index) / float(max(1, N - 1))
        )  # depth_norm (outer→inner)
        feats[j, 3] = 1.0 if L.kind in ("scf.parallel", "affine.parallel") else 0.0
    return feats
