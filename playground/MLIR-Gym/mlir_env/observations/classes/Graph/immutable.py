# mlir_env/observations/classes/Graph/immutable.py
# this file declares the parameters used for constructing data flow graph

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum

@dataclass
class GraphParams:
    '''
    - max_nodes: maximum number of nodes to observe
    - feature dim
    - emit ids: Make sure op->int mapping does not drift
    '''
    max_nodes: int = 256
    feature_dim: int = 7
    emit_ids: bool = True

@dataclass
class Features(IntEnum):
    '''
    The 7 dimensions of feature dim
    '''
    RANK = 0
    IN_LOOP = 1
    N_OPERANDS = 2
    N_RESULTS = 3
    TYPE_BITS = 4
    N_PARALLEL = 5
    N_REDUCTION = 6

@dataclass
class GraphOutParams:
    node_features: np.ndarray
    adj: np.ndaarray
    node_mask: np.ndarray

    op_type_ids: np.ndarray = None
