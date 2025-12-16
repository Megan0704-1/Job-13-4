# scripts/gnn/eval.py
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from mlir_env.utils.adaptors import ActionMaskWrapper
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

def evaluate(env, model_path, n_eval_episodes=20, deterministic=True):
    eval_env = DummyVecEnv([make_env(env)])
    model = MaskablePPO.load(model_path, env=eval_env)

    # use_masking=False for comparison
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        warn=False,
        use_masking=False,
    )
    print(f"[EVAL] episodes={n_eval_episodes} deterministic={deterministic} "
          f"mean_reward={mean_reward:.3f} ± {std_reward:.3f}")
    return mean_reward, std_reward

