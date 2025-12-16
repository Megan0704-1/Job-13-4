# rewards/finals/log_speedup.py
# This file implements final rewards formula specified in docs/adr/reward-change.md

import math
from dataclasses import dataclass
from mlir_env.rewards.interface import FinalReward, register_final


@dataclass
@register_final("log_speedup")
class LogSpeedup(FinalReward):
    c_penalty: float = 10.0
    lambda_ct: float = 0.0  # constant scale for compile time
    alpha_ct: float = 1.2  # constant budget for compile time
    beta_kb: float = 0.0  # constant scale for byte size
    gamma_kb: float = 1.0  # constant budget for byte size
    corrections: float = 1e-06  # epsilon

    def __post_init__(self):
        try:
            self.c_penalty = float(self.c_penalty)
            self.lambda_ct = float(self.lambda_ct)
            self.alpha_ct = float(self.alpha_ct)
            self.beta_kb = float(self.beta_kb)
            self.gamma_kb = float(self.gamma_kb)
            self.corrections = float(self.corrections)
        except ValueError as e:
            print(f"Failed to implicitly cast LogSpeedUp reward class parameters. Exit")
            exit(1)

    def compute(self, ctx, metrics) -> float:
        """
        If metrics is None -> treat as failure.
        R_final = -log(lat / L0)
                  - lambda_ct * max(0, ct / (alpha_ct * C0) - 1)
                  - beta_kb   * max(0, kb / (gamma_kb * B0) - 1)
        """
        # Failure or missing latency -> big negative
        if (metrics is None) or (metrics.latency_ms is None):
            return -self.c_penalty

        L0, C0, B0 = ctx.get_baseline()
        eps = self.corrections

        latency = max((metrics.latency_ms / max(L0, eps)), eps)
        r = -math.log(latency)

        # compile time soft constraints
        if self.lambda_ct and (metrics.compile_ms is not None) and (C0 is not None):
            Cb = self.alpha_ct * max(C0, eps)
            over = (metrics.compile_ms / Cb) - 1.0
            if over > 0.0:
                r -= self.lambda_ct * over

        # size kb soft constraints
        if self.beta_kb and (metrics.size_kb is not None) and (B0 is not None):
            Bb = self.gamma_kb * max(B0, eps)
            over = (metrics.size_kb / Bb) - 1.0
            if over > 0.0:
                r -= self.beta_kb * over

        return r
