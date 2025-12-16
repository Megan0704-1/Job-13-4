# observations/classes/SV/analysis/memory_analysis.py
from __future__ import annotations
from typing import Dict, List, Any
import numpy as np
from mlir import ir as mlir_ir

from mlir_env.observations.classes.SV.immutable import MemParams
from mlir_env.observations.classes.SV.analysis.utils.loop_utils import ScalarDimResolver, LoopInfo
from mlir_env.observations.classes.SV.analysis.utils.memory_api import (
    collect,
    compute_is_unit_stride_minor,
    compute_loop_carries_reuse_any,
    compute_contiguity_score,
    compute_reuse_count,
    compute_strided_hist,
)
from mlir_env.observations.classes.SV.analysis.utils.memory_utils import MemoryFeatures, FeatureMask, MemInfoOut

# -----------------------------------------------------------------------------
# Analyzer
# -----------------------------------------------------------------------------
class MemoryInformationExtractor:
    """
    Analyze affine.load/store inside a region given a loop band description.

    Params (MemParams):
      - vector_width_elements: int, 用於 stride 直方圖的分桶
      - unit_stride_policy: 'store_first' | 'any' | 'all'
      - compute_bonus: 是否額外計算 {contiguity_score, reuse_count, stride_hist}

    run() inputs:
      - moduleView: ModuleView
      - records: List[LoopInfo]  (外→內), 每個 LoopInfo 需能取得 op

    Output:
      - MemoryFeatures（per-loop 的 is_unit_stride_minor / reuse_any 等）
    """
    def __init__(self, params: MemParams):
        self.vector_W = max(2, int(params.vector_width_elements))
        self.policy = params.unit_stride_policy
        self.compute_bonus = bool(params.compute_bonus)

        self.L = 0
        self.ivs: List[mlir_ir.BlockArgument] = []
        self.iv2idx: Dict[mlir_ir.BlockArgument, int] = {}
        self.iv2steps: Dict[mlir_ir.BlockArgument, int] = {}
        self._resolver = ScalarDimResolver()

    def _init_from_records(self, records: List[LoopInfo], max_loops: int):
        '''records: from LoopInfoOut, each of the LoopInfo records one property of a loop; max_loops, how many loops we are looking at.'''
        self.L = max_loops

        # reset
        self.reset()

        for pos, record in enumerate(records):
            op = record.op
            ivs = record.iv
            # get ivs
            # parallel ops will have several iv maps to same index (index: loop depth in a band)
            if len(ivs) > 0:
                for d, v in enumerate(ivs):
                    self.ivs.append(v)
                    self.iv2idx[v] = int(pos)
                    self.iv2steps[v] = record.step[d]

    def run(self, moduleView: "ModuleView", records: List[LoopInfo], max_loops: int) -> MemoryFeatures:
        self._init_from_records(records, max_loops)

        # collect accesses and layouts -> accesses: (N, list(...)), layouts: (N,)
        accesses, layouts = collect(self.L, moduleView, self.iv2idx)

        # collect unit stride -> U: (L, )
        is_unit = compute_is_unit_stride_minor(self.L, accesses, layouts, self.ivs, self.iv2idx, self.iv2steps)
        # collect reuse -> (L, )
        reuse_any = compute_loop_carries_reuse_any(self.L, accesses)

        contig = reuse_cnt = shist = None
        if self.compute_bonus:
            # collect per loop contiguity score -> (L, )
            contig = compute_contiguity_score(self.L, accesses, layouts, self.ivs, self.iv2idx, self.iv2steps, is_unit)
            # collect per loop reuse counts-> (L, )
            reuse_cnt = compute_reuse_count(self.L, accesses)
            # collect strides stored in bucket (4) -> (L, 4)
            shist = compute_strided_hist(self.L, accesses, layouts, self.ivs, self.iv2idx, self.iv2steps, self.vector_W)

        feats = MemoryFeatures(is_unit, reuse_any, contig, reuse_cnt, shist)

        # derived masks
        mem_unit_known = (is_unit != -1)
        mem_contig_known = None
        mem_stride_known = None
        if contig is not None:
            mem_contig_known = (contig > 0.0) | mem_unit_known
        if shist is not None:
            non_unk = (shist[:,0] + shist[:,1] + shist[:,2]) > 0.0
            mem_stride_known = non_unk

        masks = FeatureMask(mem_unit_known, mem_contig_known, mem_stride_known)

        return MemInfoOut(feats, masks)

    def reset(self):
        self.ivs: List[mlir_ir.BlockArgument] = []
        self.iv2idx: Dict[mlir_ir.BlockArgument, int] = {}
        self.iv2steps: Dict[mlir_ir.BlockArgument, int] = {}
        self._resolver = ScalarDimResolver()

