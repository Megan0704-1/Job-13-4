# observations/classes/SV/analysis/loop_utils.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Iterable, Any, Tuple
import numpy as np
import re, hashlib, math
from mlir import ir as mlir_ir
import mlir.dialects as md

# for private helper handlers
from ....utils import general_utils as helper

# for public helpers
from ....utils.general_utils import POW2, const_index_like

# ---------- dataclasses ----------


@dataclass
class LoopInfo:
    """per loop info"""

    op: mlir_ir.Operation
    iv: tuple[
        mlir_ir.BlockArgument, ...
    ]  # BlockArgument (induction variable) or tuple for parallel
    step: Tuple[Optional[int],...]
    properties: tuple[
        int | None, bool | None, bool | None
    ]  # tripcount, success, is_step_pow2
    index: int  # band-local index [0..N-1]
    kind: str  # 'affine.for' | 'scf.for' | 'scf.parallel' | 'affine.parallel'


@dataclass
class LoopInfoOut:
    """All loops info in a module"""

    loops_mask: np.ndarray
    loop_feats: np.ndarray
    known_mask: np.ndarray
    records: list[LoopInfo]


# -------- scalar / constant dim resolve for LoopInfoExtractor ----


class ScalarDimResolver:
    """
    scalar/dimension resolver.
    """

    # (affine.for / scf.for)
    def step_of(self, op: mlir_ir.Operation) -> int | None:
        name = op.operation.name
        if name == "affine.for":
            return helper._affine_for_const_step(op)
        if name == "scf.for":
            return helper._scf_for_const_step(op)
        if name == "affine.parallel":
            return helper._affine_parallel_const_step(op)
        if name == "scf.parallel":
            return helper._scf_parallel_const_step(op)
        return None

    def bounds_of(self, op: mlir_ir.Operation) -> tuple[int | None, int | None]:
        name = op.operation.name
        lb, ub = None, None
        if name == "affine.for":
            lb = helper._affine_for_const_bound(op, "lb")
            ub = helper._affine_for_const_bound(op, "ub")
        if name == "scf.for":
            lb = helper._scf_for_const_bound(op, "lb")
            ub = helper._scf_for_const_bound(op, "ub")
        return (lb, ub)

    def try_tripcount_single(
        self, op: mlir_ir.Operation
    ) -> tuple[int | None, bool, bool]:
        """
        Returns (tripcount, success, is_pow2_step) for affine.for / scf.for.
        """
        step = self.step_of(op)
        lb, ub = self.bounds_of(op)
        tc = helper._tripcount(lb[0], ub[0], step[0])
        return tc, (tc is not None), POW2(step[0])

    def try_tripcount_parallel(
        self, op: mlir_ir.Operation
    ) -> tuple[int | None, bool, bool]:
        """
        scf.parallel: tripcount is the product over dimensions if all are constant.
        Returns (product_tripcount, success, all_steps_pow2).
        """
        assert "parallel" in op.name, "try_tripcount_parallel expects to work on parllel op."
        lbs, ubs, steps = helper._get_parallel_segments(op)
        if not lbs or not ubs or not steps or not (len(lbs) == len(ubs) == len(steps)):
            return None, False, False

        total = 1
        all_pow2 = True
        for lb, ub, st in zip(lbs, ubs, steps):
            tc = helper._tripcount(lb, ub, st)
            if tc is None:
                return None, False, False
            total *= max(1, tc)
            all_pow2 = all_pow2 and POW2(st)
        return total, True, all_pow2

    def try_tripcount(self, op: mlir_ir.Operation) -> tuple[int | None, bool, bool]:
        name = op.operation.name
        if name in ("affine.for", "scf.for"):
            return self.try_tripcount_single(op)
        if "parallel" in name:
            return self.try_tripcount_parallel(op)
        # other loop types can be added here
        return None, False, False

    # Expose value resolver if other passes need it
    def resolve_value_int(self, v: mlir_ir.Value) -> int | None:
        return const_index_like(v)
