# tests/test_reward_interface.py

import types
from mlir_env.rewards.interface import (
    FinalReward,
    Potential,
    register_final,
    register_potential,
    FINAL_REWARD_REGISTRY,
    POTENTIAL_REGISTRY,
)


def test_registries_exist():
    assert isinstance(FINAL_REWARD_REGISTRY, dict)
    assert isinstance(POTENTIAL_REGISTRY, dict)


def test_register_final_decorator_registers_class():
    @register_final("__dummy_final__")
    class _DummyFinal(FinalReward):
        def compute(self, ctx, metrics) -> float:
            return 42.0

    assert "__dummy_final__" in FINAL_REWARD_REGISTRY
    cls = FINAL_REWARD_REGISTRY["__dummy_final__"]
    inst = cls()
    assert isinstance(inst, FinalReward)
    # smoke compute
    assert inst.compute(ctx=None, metrics=None) == 42.0


def test_register_potential_decorator_registers_class():
    @register_potential("__dummy_potential__")
    class _DummyPotential(Potential):
        def value(self, obs: dict, feats=None) -> float:
            return 3.14

        def on_terminal(self, obs: dict, metrics, ctx) -> None:
            pass

    assert "__dummy_potential__" in POTENTIAL_REGISTRY
    cls = POTENTIAL_REGISTRY["__dummy_potential__"]
    inst = cls()
    assert isinstance(inst, Potential)
    assert inst.value(obs={}) == 3.14
    # on_terminal should not raise
    inst.on_terminal(obs={}, metrics=None, ctx=None)
