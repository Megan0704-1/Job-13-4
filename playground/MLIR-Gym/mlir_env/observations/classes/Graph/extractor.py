# mlir_env/observations/classes/Graph/extractor.py
# This file implement how feature is extracted from mlir nodes

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
import mlir.ir as mlir_ir

from mlir_env.views.viewer import ModuleView

from ..utils.general_utils import walk_ops_recursive, get_id
from .immutable import GraphParams, Features as PARAMS
from .helper import in_loop, get_type_info, get_linalg_generic_info, has_linalg_generic_parent

import pdb

class GraphExtractor:
    '''
    Takes an MLIR returns a dataflow graph with PARAMS.feature_dim features
    Definitions
    - node: mlir.Operations
    - edge: def-use (OpResult.owner -> use op)
    '''

    def __init__(self, params: GraphParams):
        self.p = params

    def extract(self, ir_text: str):
        """
        Return:
          node_feats: List[List[float]] # continuous feature (N,4)
          edges: List[Tuple[int,int]]
          op_type_ids: List[int] # (N,)
          labels: Optional[List[str]]
        """
        mv = ModuleView(ir_text)
        node_feats: List[List[float]] = []
        node_map: Dict[Any, int] = {}
        edges: List[Tuple[int, int]] = []
        op_type_ids: List[int] = []
        labels: List[str] = []

        def push_node(op: Any, feats: List[float]):
            op_name = op.operation.name
            idx = len(node_feats)
            if idx >= self.p.max_nodes:
                # exceed max node configuration
                return -1

            assert len(feats) == self.p.feature_dim, f"feature dimension mismatch, input feature length: {len(feats)} v.s. expect feature length {self.p.feature_dim}"

            node_map[op] = idx
            node_feats.append(feats)

            # id type: hash (we treat BlockArgument as block_arg)
            if self.p.emit_ids:
                name = op_name if op_name is not None else "block_arg"
                dialect = name.split(".")[0] if "." in name else name
                op_type_ids.append(get_id(dialect+name, op, mode="int"))

            return idx

        def build_node(op) -> int:
            if op in node_map:
                return node_map[op]
            rank, bits = (0, 0)
            if len(op.results) > 0:
                rank, bits = get_type_info(op.results[0].type)
            feats = [
                float(rank),
                float(in_loop(op)),
                float(len(op.operands)),
                float(len(op.results)),
                float(bits),
                float(get_linalg_generic_info(op)[0]),
                float(get_linalg_generic_info(op)[1]),
            ]
            return push_node(op, feats)

        with mv.ctx:
            for _, funcOp in mv.functions():
                # Operations + edges
                for op in walk_ops_recursive(funcOp):
                    name = op.operation.name
                    if name in ("builtin.module", "func.func"):
                        continue
                    if has_linalg_generic_parent(op):
                        continue

                    this_idx = build_node(op)
                    if this_idx < 0:
                        # reach max node limit, do not push any nodes in graph.
                        break

                    for operand in op.operands:
                        if hasattr(operand, "owner") and isinstance(operand.owner, mlir_ir.Operation):
                            src_op = operand.owner
                            src_idx = node_map.get(src_op)
                            if src_idx is None:
                                src_idx = build_node(src_op)
                                if src_idx < 0:
                                    break
                            edges.append((src_idx, this_idx))

        assert node_feats != [] and edges != [], "Should never return empty observation"
        return node_feats, edges, op_type_ids

