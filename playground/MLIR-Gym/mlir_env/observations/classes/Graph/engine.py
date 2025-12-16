# observations/classes/Graph/engine.py

from __future__ import annotations
import os
import numpy as np
from typing import Any, Dict, Optional
from dataclasses import dataclass
import yaml
from gymnasium import spaces

from mlir_env.core.config import Config
from mlir_env.core.context import EpisodeContext

from mlir_env.views.spec import ViewKind
from mlir_env.views.viewer import ModuleView
from mlir_env.views.view_runner import ViewRunner

from mlir_env.observations.interface import State, register_state

from .immutable import GraphParams, GraphOutParams
from .extractor import GraphExtractor

@register_state("gnn_dataflow")
class GraphEngine(State):
    '''
    Transform an MLIR to a fixed-size observation
    - node_features: (N, F)
    - adj: (N, N) {0, 1} // adjacency matrix
    - node_mask: (N,) {0, 1}
    '''

    def __init__(self, cfg: Config, ctx: EpisodeContext, params: GraphParams):
        self.cfg = cfg
        self.ctx = ctx
        self.params = params
        self.N = int(params.max_nodes)
        self.F = int(params.feature_dim)

        self.extractor = GraphExtractor(params)

        self._space = spaces.Dict({
            "node_features": spaces.Box(-np.inf, np.inf, shape=(self.N, self.F), dtype=np.float32),
            "adj": spaces.MultiBinary((self.N, self.N)),
            "node_mask": spaces.MultiBinary(self.N),
            "op_type_ids": spaces.Box(0, np.inf, shape=(self.N, ), dtype=np.int64)
        })

    @classmethod
    def from_yaml(cls, cfg: Config, ctx: EpisodeContext, path:str) -> GraphEngine:
        assert os.path.exists(path), f"{path} does not exist, assert at GraphEngine"

        graph_param = {}
        if path:
            with open(path) as fh:
                data = yaml.safe_load(fh) or {}
            graph_params = data.get("graphParams", {})
        params = GraphParams(**graph_params)
        return cls(cfg=cfg, ctx=ctx, params=params)

    def build_space(self) -> spaces.Space:
        return self._space

    def observe(self, ir_text: str):
        viewer = ViewRunner(self.cfg)
        view_ir = viewer.view(ir_text, ViewKind.GRAPH)

        feats, edges, type_ids = self.extractor.extract(ir_text)

        node_feats = np.zeros((self.N, self.F), dtype=np.float32)
        mask = np.zeros((self.N,), dtype=bool)
        graph_edges = np.zeros((self.N, self.N), dtype=np.int8)
        for idx, feat in enumerate(feats[:self.N]):
            node_feats[idx, :] = np.asarray(feat, dtype=np.float32)
            mask[idx] = True

        for (u, v) in edges:
            if u < self.N and v < self.N:
                graph_edges[u, v] = 1

        out = GraphOutParams(
            node_features = node_feats,
            adj = graph_edges,
            node_mask = mask
        )

        if self.params.emit_ids:
            op_type_ids = np.zeros((self.N,), dtype=np.int64)

            for idx, type_id in enumerate(type_ids[:self.N]):
                op_type_ids[idx] = int(type_id)

            out.op_type_ids = op_type_ids

        return out.__dict__

    def schema_id(self) -> str:
        return f"graph:dataflow@N={self.N},F={self.F}"

    def schema_meta(self) -> dict[str, Any]:
        return {
            "max_nodes": self.N,
            "feature_dim": self.F,
        }

    def reset(self) -> None:
        pass
