# mlir_env/utils/adaptors.py

import numpy as np
from typing import Dict, List
import gymnasium as gym
from gymnasium import spaces
from scripts.utils.general_utils import bin012

class ForwardAdaptor(gym.Wrapper):
    """
    Forwards self defined fields and methods out from gym default wrappers.
    e.g., action_layer is a field defined in MLIRWorldEnv,
    by wrapping this adaptor around the env, we are now able to access
    action_layer field by calling `env.action_layer`
    """

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(self.env.unwrapped, name)


class ActionMaskWrapper(gym.Wrapper):
    """
    Pack
    1) action mask [0, 1, 1, 0, ...]
    2) legal action buckets (n_tile, n_buff, ...) [binarize]
    3) prev_cat (FUSE, BUFFERIZE, TILE, ...)
    into observation.
    note. the bucket rule is the same as q_smoke
    """

    CATEGORIES = ("FUSE","BUFFERIZE","TILE","CONVERT","EVAL","OTHER")
    CAT_MAP = {cat : idx for idx, cat in enumerate(CATEGORIES)}

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.prev_cat = "OTHER"
        self.n_actions = int(self.action_space.n)
        self.action_groups: Dict[str, List[int]] = self._build_action_groups()
        self.action_mask: np.ndarray = None
        n_cat = len(self.CATEGORIES)
        augment = {
            "action_mask": spaces.MultiBinary(self.n_actions),
            "legal_bins": spaces.MultiDiscrete([3,3,3]),
            "prev_cat_idx": spaces.Discrete(n_cat),
        }

        assert isinstance(self.observation_space, spaces.Dict), f"GraphEngine is expected to ouptut a Dict-typed observation, got {type(self.observation_space)}"
        self.observation_space = spaces.Dict({
            **self.observation_space, **augment
        })

    def _build_action_groups(self) -> Dict[str, List[int]]:
        groups = {k: [] for k in self.CATEGORIES}
        for i in range(self.n_actions):
            groups[self._classify_action(i)].append(i)
        return groups

    def _classify_action(self, idx: int) -> str:
        act = self.env.action_layer.idx_to_action(idx)
        name = (getattr(act, "name", None) or str(act)).lower()

        if "eval" in name: return "EVAL"
        if "tile" in name: return "TILE"
        if "fuse" in name or "fold" in name: return "FUSE"
        if "buffer" in name: return "BUFFERIZE"
        if "convert" in name or "lower" in name: return "CONVERT"
        return "OTHER"

    def _augment(self, obs: dict, info: dict) -> dict:
        # 1) build legal action groups (bucket)
        mask = np.asarray(info.get("mask", np.ones(self.action_space.n, dtype=np.int8)), dtype=bool)
        self.action_mask = mask

        n_tile = int(mask[self.action_groups["TILE"]].sum()) if self.action_groups["TILE"] else 0
        n_fuse = int(mask[self.action_groups["FUSE"]].sum()) if self.action_groups["FUSE"] else 0
        n_convert = int(mask[self.action_groups["CONVERT"]].sum()) if self.action_groups["CONVERT"] else 0
        legal_bins = np.asarray([bin012(n_tile), bin012(n_fuse), bin012(n_convert)], dtype=np.int64)

        # 2) last action index
        prev_cat_idx = self.CAT_MAP[self.prev_cat]

        # 3) pack back to original obs
        out = dict(obs)
        out["action_mask"] = mask.astype(np.int8)
        out["legal_bins"] = legal_bins
        out["prev_cat_idx"] = np.int64(prev_cat_idx)

        return out

    def action_masks(self) -> np.ndarray:
        '''
        A method for sb3 action wrapper to call
        Returns: [0, 1, 1, ..., n_actions]
        '''
        if self.action_mask is None:
            return np.ones(self.action_space.n, dtype=bool)
        return self.action_mask.astype(bool)

    def reset(self, **kwargs):
        self.prev_cat = "OTHER"
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs, info), info

    def step(self, action: int):
        # record the prev act before stepping over the action
        self.prev_cat =  self._classify_action(int(action))
        # take original step
        obs, r, term, trun, info = self.env.step(action)
        # return augmented obs with action infos packed
        return self._augment(obs, info), r, term, trun, info
