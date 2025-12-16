import os
import subprocess
import json
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from mlir_env.core.immutable import StepReturn
from mlir_env.core.config import Config
from mlir_env.core.context import EpisodeContext
from mlir_env.actions.layer import ActionLayer

from mlir_env.rewards.config import RewardConfig
from mlir_env.rewards.rewarder import Rewarder

from mlir_env.observations.config import StateConfig
from mlir_env.observations.builder import StateBuilder

# utils
from mlir_env.utils.common_utils import *
from mlir_env.utils.ir_utils import LRUCache

seed = 42


class MLIRWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "ren": 4}

    def __init__(
        self,
        cfg_path: str = None,
        ctx_path: str = None,
        state_cfg_path: str = None,
        render_mode=None,
    ):
        super().__init__()

        # 1) config
        _cfg_path = (
            (os.environ.get("MLIR_GYM_ROOT") + "/benchmarks/settings/gemm_config.yaml")
            if cfg_path is None
            else os.environ.get("MLIR_GYM_ROOT") + "/" + cfg_path
        )
        self.cfg = Config.from_yaml(_cfg_path)

        # 2) context (per episode) -> post init at reset stage
        _ctx_path = (
            (
                os.environ.get("MLIR_GYM_ROOT")
                + f"/benchmarks/settings/gemm_{self.cfg.compile_option}_context.yaml"
            )
            if ctx_path is None
            else os.environ.get("MLIR_GYM_ROOT") + "/" + ctx_path
        )
        self.ctx = EpisodeContext.from_yaml(self.cfg, _ctx_path)
        self.action_layer = ActionLayer(cfg=self.cfg, ctx=self.ctx)
        self.action_space = self.action_layer.space

        ## rewards
        reward_cfg_path = (
            os.environ.get("MLIR_GYM_ROOT") + "/settings/rewards/log_speedup.yaml"
        )
        self.rewarder = Rewarder(cfg=RewardConfig.from_yaml(reward_cfg_path))

        ## state
        _state_cfg_path = (
            (
                os.environ.get("MLIR_GYM_ROOT")
                + "/settings/observations/structured_vector_config.yml"
            )
            if state_cfg_path is None
            else os.environ.get("MLIR_GYM_ROOT") + "/" + state_cfg_path
        )
        self.state_cfg = StateConfig.from_yaml(_state_cfg_path)
        self.state_builder = StateBuilder.from_config(
            self.cfg, self.ctx, self.state_cfg
        )
        self.observation_space = self.state_builder.get_space()

        # environment maintained variables
        self.ir = None
        self.total_run = None
        self.cnt = 0
        self.episode_id = 0

        # utils
        self.cache = LRUCache(capacity=2048, namespace=self.cfg.hash())

    def get_info(self):
        return {
            "episode_id": self.episode_id,
            "round": self.cnt,
            "ir": self.ir,
            "mask": self.action_layer.mask(self.ir),
        }

    def get_observation(self, ir_text: str = None):
        if ir_text is None:
            return self.state_builder.get_observation(self.ir)
        return self.state_builder.get_observation(ir_text)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        """
        we should always reset state before getting observations or info.
        """

        ### --- Issue: #30: action-change --- ###
        self.ctx = self.ctx.reset_for()
        self.action_layer = ActionLayer(cfg=self.cfg, ctx=self.ctx)

        # environment maintained variables
        self.ir = self.ctx.mlir_content
        self.total_run = self.ctx.max_steps
        self.cnt = 0
        self.episode_id += 1

        ### --- Issue: #32: state-change (Phase 1) --- ###
        return self.get_observation(), self.get_info()

    def step(self, action_idx):
        is_eval = self.action_layer.idx_to_action(action_idx).is_eval()
        # 0) look up cache if action is EVAL
        if is_eval and self.cache.lookup(self.ir):
            payload = self.cache.get(self.ir)
            obs = self.get_observation(self.ir)
            return obs, *list(payload.values())

        # 1) apply an action, get a StepReturn
        sr = self.action_layer.apply_action(self.ir, action_idx)
        # 2) From the StepReturn statistics, get a reward
        r = self.rewarder.get_reward(self.ctx, sr)
        # 3) Get new observation
        new_obs = self.get_observation(sr.result.new_ir)
        # 4) Update environment maintained vars
        self.ir = sr.result.new_ir
        self.cnt += 1

        # get new mask
        new_mask = self.action_layer.mask(self.ir)

        # update info
        info = self.get_info()
        info.update(
            {
                "ir": sr.result.new_ir,
                "index": sr.info.index,
                "mask": new_mask,
                "status": sr.result.status,
                "changed": ir_has_changed.check(sr.result.e_code),
            }
        )
        info["round"] += 1
        payload = {
            "reward": float(r),
            "trun": bool(sr.trun),
            "term": bool(sr.term),
            "info": info,
        }

        # 5) cache eval result
        if is_eval:
            # eval does not modify ir
            self.cache.add(self.ir, payload)
            return new_obs, *list(payload.values())
        elif self.cnt == self.total_run:
            payload["trun"] = True
            return new_obs, *list(payload.values())
        else:
            payload["trun"] = False
            return new_obs, *list(payload.values())

    def render(self):
        print("TODO")

    def close(self):
        print(f"total pass run: {self.total_run - self.cnt}")
        print(f"end ir: {self.ir}")
        print("exit")
