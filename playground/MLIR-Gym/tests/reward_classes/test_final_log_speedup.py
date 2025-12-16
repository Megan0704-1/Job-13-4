# tests/reward_classes/test_final_log_speedup.py

import math
import numpy as np
import pytest
from mlir_env.rewards.config import RewardConfig
from mlir_env.rewards.finals.log_speedup import LogSpeedup


class DummyCtx:
    def __init__(self, L0=100.0, C0=50.0, B0=200.0):
        self.L0, self.C0, self.B0 = L0, C0, B0

    def get_baseline(self):
        return (self.L0, self.C0, self.B0)


class DummyMetrics:
    def __init__(self, latency_ms=None, compile_ms=None, size_kb=None):
        self.latency_ms = latency_ms
        self.compile_ms = compile_ms
        self.size_kb = size_kb


# reward class type check
def test_param_type_casting(globalVar):
    from mlir_env.rewards.finals.log_speedup import LogSpeedup

    rcfg = RewardConfig.from_yaml(globalVar.get("rewards.yaml"))
    reward_cls = LogSpeedup(**rcfg.final_params)

    params = [
        "c_penalty",
        "lambda_ct",
        "alpha_ct",
        "beta_kb",
        "gamma_kb",
        "corrections",
    ]
    for p in params:
        assert isinstance(reward_cls.__dict__.get(p), float)


def test_success_basic_signs():
    ctx = DummyCtx(L0=100.0)
    fr = LogSpeedup(c_penalty=10.0, lambda_ct=0.0, beta_kb=0.0)

    # faster → positive
    r_fast = fr.compute(ctx, DummyMetrics(latency_ms=80.0))
    assert r_fast > 0.0

    # equal → ~0
    r_same = fr.compute(ctx, DummyMetrics(latency_ms=100.0))
    assert np.allclose(abs(r_same), 0.0)

    # slower → negative
    r_slow = fr.compute(ctx, DummyMetrics(latency_ms=120.0))
    assert r_slow < 0.0


def test_compile_time_penalty_only_when_over_budget():
    ctx = DummyCtx(L0=100.0, C0=50.0)
    # budget = alpha_ct * C0 = 1.2 * 50 = 60
    fr = LogSpeedup(c_penalty=10.0, lambda_ct=1.0, alpha_ct=1.2)

    r_no = fr.compute(ctx, DummyMetrics(latency_ms=100.0, compile_ms=60.0))
    assert np.allclose(abs(r_no), 0.0)

    r_over = fr.compute(ctx, DummyMetrics(latency_ms=100.0, compile_ms=72.0))
    assert pytest.approx(r_over, rel=1e-6) == -0.2


def test_size_penalty_only_when_over_budget():
    ctx = DummyCtx(B0=200.0)
    fr = LogSpeedup(c_penalty=10.0, beta_kb=0.5, gamma_kb=1.0)

    r_no = fr.compute(ctx, DummyMetrics(latency_ms=100.0, size_kb=200.0))
    assert np.allclose(abs(r_no), 0.0)

    r_over = fr.compute(ctx, DummyMetrics(latency_ms=100.0, size_kb=250.0))
    # 25% over with beta=0.5 => -0.125
    assert np.allclose(r_over, -0.125)


def test_failure_or_missing_latency_returns_c_penalty():
    ctx = DummyCtx()
    fr = LogSpeedup(c_penalty=7.5)
    assert fr.compute(ctx, None) == -7.5
    assert fr.compute(ctx, DummyMetrics(latency_ms=None)) == -7.5


def test_epsilon_guard_prevents_log_blowup():
    ctx = DummyCtx(L0=100.0)
    fr = LogSpeedup(corrections=1e-6)
    r = fr.compute(ctx, DummyMetrics(latency_ms=1e-12))
    assert math.isfinite(r) and r > 0.0
