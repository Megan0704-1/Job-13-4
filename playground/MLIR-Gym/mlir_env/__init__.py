# mlir_env/core/__init__.py
from gymnasium.envs.registration import register

register(
    id="mlir_env/mlir-v5",
    entry_point="mlir_env.core:MLIRWorldEnv",
)
