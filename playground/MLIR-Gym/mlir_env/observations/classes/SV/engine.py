# observations/classes/SV/engine.py

from __future__ import annotations
import os
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
from mlir_env.observations.classes.SV.immutable import LoopParams, MemParams, PackerParams
from mlir_env.observations.classes.SV.analysis.loop_analysis import (
    LoopInformationExtractor,
)
from mlir_env.observations.classes.SV.analysis.memory_analysis import (
    MemoryInformationExtractor,
)
from mlir_env.observations.classes.SV.packer import Packer


@register_state("structured_vector")
class SVEngine(State):
    def __init__(
        self,
        cfg: Config,
        ctx: EpisodeContext,
        loop_params: LoopParams,
        mem_params: MemParams,
        probes_params: ProbesParams | None = None,
        history_params: HistoryParams | None = None,
        hw_params: HardwareParams | None = None,
        packer_params: PackerParams | None = None,
    ):

        self.cfg = cfg
        self.ctx = ctx
        self.L = 0

        # keeps reference of parameters for extractors and packers
        self.loop_params = loop_params
        self.mem_params = mem_params
        self.probes_params = probes_params
        self.history_params = history_params
        self.hw_params = hw_params
        self.packer_params = packer_params or PackerParams()

        # instantiate extractors
        self.loop_extractor = LoopInformationExtractor(loop_params)
        self.mem_extractor = MemoryInformationExtractor(mem_params)

        self.probes_extractor = None
        if probes_params and probes_params.enabled:
            # self.probes_extractor = ProbesInformationExtractor(probes_params)
            pass

        self.history_extractor = None
        if history_params and history_params.enabled:
            # self.history_extractor = HistoryInformationExtractor(history_params, registry=...)
            pass

        # TODO(megan.kuo)
        self.hw_provider = HardwareProvider(hw_params) if hw_params else None

        # state-wise loop dimension
        self.L = loop_params.max_loops

        # pack
        self.packer = Packer(
            loop_params=loop_params,
            mem_params=mem_params,
            probes_params=probes_params,
            history_params=history_params,
            hw_params=hw_params,
            packer_params=self.packer_params,
        )

        self._space = self.packer.build_space()

    @classmethod
    def from_yaml(cls, cfg: Config, ctx: EpisodeContext, path: str) -> SVEngine:
        assert os.path.exists(path), f"{path} does not exist, assert at SVEngine"
        data: dict[str, Any] = {}
        if path:
            with open(path) as fh:
                data = yaml.safe_load(fh) or {}

        def load(section: str, cls_):
            paramCfg = data.get(section)
            return cls_(**paramCfg) if isinstance(paramCfg, dict) else None

        # TODO(megan.kuo) placeholders
        loop_params = load("loopParams", LoopParams) or LoopParams()
        mem_params = load("memoryParams", MemParams) or MemParams()
        probes_params = None  # load("probeParams", ProbesParams)
        history_params = None  # load("historyParams", HistoryParams)
        hw_params = None  # load("hardwareInfo", HardwareParams)
        packer_params = load("packerParams", PackerParams) or PackerParams()

        return cls(
            cfg,
            ctx,
            loop_params,
            mem_params,
            probes_params,
            history_params,
            hw_params,
            packer_params,
        )

    def build_space(self) -> spaces.Space:
        """Returns the cache space established by self.packer"""
        return self._space

    def observe(self, ir_text: str) -> Any:
        viewer = ViewRunner(self.cfg)

        # 1) get affine view for extracting loop info
        if self.ctx.view_option == "affine":
            view_ir = viewer.view(ir_text, ViewKind.AFFINE)
        elif self.ctx.view_option == "scf":
            view_ir = viewer.view(ir_text, ViewKind.SCF)

        view_module = ModuleView(view_ir)
        loops_info = self.loop_extractor.run(view_module)

        # 2)  get memref view for extracting memory info
        # memref_view_ir = viewer.view(affine_view_ir, ViewKind.MEMREF)
        # memref_module = ModuleView(memref_view_ir)
        mem_info = (
            self.mem_extractor.run(view_module, loops_info.records, self.L) if self.mem_extractor else None
        )

        # TODO(megan.kuo)
        probes_info = (
            self.probes_extractor.run(loops_info, mem_info)
            if self.probes_extractor
            else None
        )
        history_info = self.history_extractor.run() if self.history_extractor else None
        hw_info = self.hw_provider.read() if self.hw_provider else None
        obs = self.packer.pack(
            loops=loops_info,
            mems=mem_info,
            probes=probes_info,
            history=history_info,
            hw=hw_info,
        )
        return obs

    def schema_id(self) -> str:
        # include which channels are on + caps for traceability
        chans = ",".join(sorted(self.packer_params.include_channels))
        return f"sv:structured_vector@[{chans}]"

    def schema_meta(self) -> dict[str, Any]:
        return {
            "loop": self.loop_params.__dict__,
            "mem": self.mem_params.__dict__ if self.mem_params else None,
            "probes": self.probes_params.__dict__ if self.probes_params else None,
            "history": self.history_params.__dict__ if self.history_params else None,
            "hardware": self.hw_params.__dict__ if self.hw_params else None,
            "packer": self.packer_params.__dict__,
        }

    def reset(self) -> None:
        pass
