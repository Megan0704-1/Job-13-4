# observations/classes/SV/packer.py

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
from gymnasium import spaces

from mlir_env.observations.classes.SV.immutable import LoopParams, MemParams, PackerParams
from mlir_env.observations.classes.SV.analysis.utils.loop_utils import LoopInfoOut
from mlir_env.observations.classes.SV.analysis.utils.memory_utils import MemInfoOut

class Packer:
    def __init__(
        self,
        loop_params: LoopParams,
        mem_params: MemParams,
        probes_params: ProbesParams | None,
        history_params: HistoryParams | None,
        hw_params: HardwareParams | None,
        packer_params: PackerParams,
    ):
        self.lp = loop_params
        self.mp = mem_params
        self.pp = probes_params
        self.hp = history_params
        self.hw = hw_params
        self.pk = packer_params

    def build_space(self) -> spaces.Space:
        dict_space: dict[str, spaces.Space] = {}

        if self.can_extract_feature(self.lp):
            L = self.lp.max_loops
            dict_space["loop_feats"] = spaces.Box(-np.inf, np.inf, shape=(L,4), dtype=np.float32)
            dict_space["loop_mask"] = spaces.MultiBinary(L) # per loop
            dict_space["known_mask"] = spaces.MultiBinary((L, 4)) # per loop

        if self.can_extract_feature(self.mp):
            L = self.lp.max_loops

            # feats
            dict_space["mem_unit_stride"] = spaces.Box(-1, 1, shape=(L,), dtype=np.int8)
            dict_space["mem_reuse_any"] = spaces.Box(0, 1, shape=(L,), dtype=np.int8)
            # masks
            dict_space["mem_unit_known"] = spaces.MultiBinary(L)

            if self.mp.compute_bonus:
                # feats
                dict_space["mem_contig_score"] = spaces.Box(0.0, 1.0, shape=(L,), dtype=np.float32)
                dict_space["mem_reuse_cnt"]    = spaces.Box(0.0, np.inf, shape=(L,), dtype=np.int8)
                dict_space["mem_strided_hist"]  = spaces.Box(0.0, np.inf, shape=(L, 4), dtype=np.int8)
                # masks
                dict_space["mem_contig_known"] = spaces.MultiBinary(L)
                dict_space["mem_stride_known"] = spaces.MultiBinary(L)

        # TODO(megan.kuo) placeholders
        if "probes" in self.pk.include_channels and self.pp and self.pp.enabled:
            N, P, C = self.lp.max_loops, self.pp.n_probe, self.pp.probe_channels
            dict_space["probes"] = spaces.Box(0, 1, shape=(N, P, C), dtype=np.float32)
            dict_space["probes_mask"] = spaces.MultiBinary((N, P))

        if "history" in self.pk.include_channels and self.hp and self.hp.enabled:
            T, D, F = self.hp.history_len, self.hp.param_dim, self.hp.irreversible_flags
            dict_space["history_ids"] = spaces.Box(
                low=-1, high=9999, shape=(T,), dtype=np.int64
            )
            dict_space["history_params"] = spaces.Box(
                0, 1, shape=(T, D), dtype=np.float32
            )
            dict_space["irreversible"] = spaces.MultiBinary(F)

        if "hw" in self.pk.include_channels and self.hw:
            H = len(self.hw.fields)
            dict_space["hw"] = spaces.Box(0, 1, shape=(H,), dtype=np.float32)

        # Always okay to add action-related tensors later if desired
        return spaces.Dict(dict_space)

    def pack(
        self, *, loops: LoopInfoOut, mems: MemInfoOut = None, probes=None, history=None, hw=None
    ) -> dict[str, Any]:
        '''Pack all analysis InfoOut struct'''
        obs: dict[str, Any] = {}

        if self.can_extract_feature(loops):
            obs["loop_feats"] = loops.loop_feats.astype(np.float32)
            obs["loop_mask"] = loops.loops_mask.astype(np.int8)
            obs["known_mask"] = loops.known_mask.astype(np.int8)

        if self.can_extract_feature(mems):
            # feats
            obs["mem_unit_stride"] = mems.feats.is_unit_stride_minor.astype(np.int8)
            obs["mem_reuse_any"] = mems.feats.loop_carries_reuse_any.astype(np.int8)

            # masks
            obs["mem_unit_known"] = mems.masks.unit_known.astype(bool)
            if self.mp.compute_bonus:
                # feats
                obs["mem_contig_score"] = mems.feats.contiguity_score.astype(np.float32)
                obs["mem_reuse_cnt"] = mems.feats.reuse_count.astype(np.int8)
                obs["mem_strided_hist"] = mems.feats.strided_hist.astype(np.int8)
                # masks
                obs["mem_contig_known"] = mems.masks.contig_known.astype(bool)
                obs["mem_stride_known"] = mems.masks.stride_known.astype(bool)

        if "probes" in self.pk.include_channels and probes is not None:
            obs["probes"] = probes.probes.astype(np.float32)
            obs["probes_mask"] = probes.probes_mask.astype(np.int8)

        if "history" in self.pk.include_channels and history is not None:
            obs["history_ids"] = history.last_k_ids.astype(np.int64)
            obs["history_params"] = history.last_k_params.astype(np.float32)
            obs["irreversible"] = history.irreversible_flags.astype(np.int8)

        if "hw" in self.pk.include_channels and hw is not None:
            obs["hw"] = hw.hw_embed.astype(np.float32)

        return obs

    def can_extract_feature(self, feats: Any) -> bool:
        feat_exist = feats is not None
        if isinstance(feats, LoopInfoOut) or isinstance(feats, LoopParams):
            return "loop" in self.pk.include_channels and feat_exist
        if isinstance(feats, MemInfoOut) or isinstance(feats, MemParams):
            return "mems" in self.pk.include_channels and feat_exist
        return False
