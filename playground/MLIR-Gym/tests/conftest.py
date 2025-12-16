# tests/conftest.py

import pytest
from mlir_env.core.config import Config
from mlir_env.core.context import EpisodeContext


# a reusable fixture for all tests
@pytest.fixture
def globalVar():
    return {
        "config.yaml": "./benchmarks/settings/gemm_config.yaml",
        "context.yaml": "./benchmarks/settings/gemm_aot_context.yaml",
        "rewards.yaml": "./settings/rewards/log_speedup.yaml",
        "state_sv_config.yaml": "./settings/observations/structured_vector_config.yml",
        "graph_config.yaml": "./settings/observations/graph_config.yml",
    }


@pytest.fixture
def cfg_from_yaml(globalVar):
    cfg = Config.from_yaml(globalVar["config.yaml"])
    return cfg


@pytest.fixture
def ctx_from_yaml(cfg_from_yaml, globalVar):
    ctx = EpisodeContext.from_yaml(cfg_from_yaml, globalVar["context.yaml"])
    return ctx
