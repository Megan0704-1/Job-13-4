# tests/test_reward_config.py

import io
import textwrap
from mlir_env.rewards.config import RewardConfig


def test_reward_config_defaults():
    cfg = RewardConfig()
    assert cfg.final == "log_speedup"
    assert cfg.potential == "none"
    assert cfg.gamma == 0.99
    assert isinstance(cfg.final_params, dict)
    assert isinstance(cfg.potential_params, dict)
    assert isinstance(cfg.step_params, dict)


def test_reward_config_from_yaml(tmp_path):
    yaml_text = textwrap.dedent(
        """
    final: log_speedup
    potential: none
    gamma: 0.97
    final_params:
      c_penalty: 7.5
      lambda_ct: 1.0
    potential_params:
      foo: bar
    step_params:
      eval_on_truncate_penalty: 0.3
    """
    )
    p = tmp_path / "reward.yaml"
    p.write_text(yaml_text)
    cfg = RewardConfig.from_yaml(str(p))
    assert cfg.final == "log_speedup"
    assert cfg.potential == "none"
    assert cfg.gamma == 0.97
    assert cfg.final_params["c_penalty"] == 7.5
    assert cfg.final_params["lambda_ct"] == 1.0
    assert cfg.potential_params["foo"] == "bar"
    assert cfg.step_params.get("eval_on_truncate_penalty") == 0.3


def test_reward_config_from_path(globalVar):
    cfg = RewardConfig.from_yaml(globalVar.get("rewards.yaml"))
    assert cfg.final == "log_speedup"
    assert cfg.potential == "none"
    assert cfg.gamma == 0.99
    assert cfg.final_params["c_penalty"] == 10.0
    assert cfg.final_params["lambda_ct"] == 0.0
    assert float(cfg.final_params["corrections"]) == 1e-6
    assert cfg.step_params.get("eval_on_truncate_penalty") == 0.05
