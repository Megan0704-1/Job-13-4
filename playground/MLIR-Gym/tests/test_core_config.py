# tests/test_config.py

from mlir_env.core.config import Config


def test_config_from_yaml(globalVar):
    cfg = Config.from_yaml(globalVar.get("config.yaml"))
    assert len(cfg.backends) != 0


def test_config_hash(globalVar):
    cfg = Config.from_yaml(globalVar.get("config.yaml"))
    assert cfg.hash() == cfg.hash()
    assert cfg._hash_id == cfg.hash()
