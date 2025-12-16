import os
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from mlir_env.utils.adaptors import ActionMaskWrapper
from scripts.utils.general_utils import load_train_config
from .feature_extractor import GraphFeaturesExtractor

def make_env(env_base):
    def mask_fn(env):
        return env.action_masks()

    def _thunk():
        env = env_base
        env = ActionMaskWrapper(env)
        env = ActionMasker(env, mask_fn)
        return env
    return _thunk


def train(env, ppo_cfg):
    cfg = load_train_config(ppo_cfg)

    # pack envs, expects a callable factory
    env = DummyVecEnv([make_env(env)])

    policy_kwargs = dict(
        features_extractor_class=GraphFeaturesExtractor,
        features_extractor_kwargs=dict(
            gcn_hidden=cfg.ppo.gcn_hidden,
            gcn_out=cfg.ppo.gcn_out,
        ),
    )

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        n_steps=cfg.ppo.n_steps,
        batch_size=cfg.ppo.batch_size,
        learning_rate=cfg.ppo.lr,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        ent_coef=cfg.ppo.ent_coef,
        clip_range=cfg.ppo.clip_range,
        verbose=1,
    )

    model.learn(total_timesteps=cfg.ppo.timesteps)
    outp = Path(cfg.ppo.save_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    model.save(cfg.ppo.save_path)
    print(f"[OK] PPO-GNN saved to: {cfg.ppo.save_path}")
