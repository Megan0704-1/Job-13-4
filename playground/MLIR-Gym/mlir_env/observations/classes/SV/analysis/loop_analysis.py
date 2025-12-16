# observations/classes/SV/analysis/loop_analysis.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, List
import numpy as np
import math

from mlir_env.observations.classes.SV.immutable import LoopParams
from mlir_env.observations.classes.SV.analysis.utils.loop_utils import LoopInfoOut
from mlir_env.observations.classes.SV.analysis.utils.loop_api import (
    band_from_innermost,
    extract_loop_features,
)
from mlir_env.observations.classes.utils.general_utils import (
    is_loop,
    walk_ops_recursive,
)

import mlir.ir as mlir_ir


class LoopInformationExtractor:
    def __init__(self, params: LoopParams):
        self.N = int(params.max_loops)

    def run(self, moduleView: ModuleView) -> LoopInfoOut:
        # Prepare loop analysis output:
        moduleOp = moduleView.module
        # 1) declare fields with shape: first N
        N = self.N
        records = []

        # 2) walk every function
        for funcName, funcOp in moduleView.functions():
            visited = set()
            for op in walk_ops_recursive(funcOp):
                # 3) skip if not a loop op
                if not is_loop(op):
                    continue
                # 4) skip if visited
                if op in visited:
                    continue
                visited.add(op)

                # 5) get band from loop op (outer to inner)
                bandinfo = band_from_innermost(op)
                if not bandinfo:
                    continue

                # 6) get feature from loop construct
                feats = extract_loop_features(bandinfo)

                # 7) records loop info
                for idx, loopInfo in enumerate(bandinfo):
                    visited.add(loopInfo.op)
                    # [debug] loopId = get_id(funcName, loopInfo.op)
                    records.append((loopInfo, feats[idx]))

        # 8) caps
        if len(records) > self.N:
            records = records[: self.N]

        # 9) construct loop info output
        loop_feats = np.zeros((N, 4), dtype=np.float32)
        known_mask = np.zeros((N, 4), dtype=bool)
        loops_mask = np.zeros((N,), dtype=bool)

        for i, (loopInfo, feat) in enumerate(records):
            loops_mask[i] = True
            loop_feats[i] = feat
            _, succ, _ = loopInfo.properties
            known_mask[i, 0] = succ  # is tripcount known?
            known_mask[i, 1] = succ  # is step_pow2 known?
            known_mask[i, 2] = True  # normalize loop depth known?
            known_mask[i, 3] = True  # parallel hint?

        # records back as your dataclass if you need them for join
        return LoopInfoOut(
            loops_mask=loops_mask,
            loop_feats=loop_feats,
            known_mask=known_mask,
            records=[li for li, _ in records],
        )
