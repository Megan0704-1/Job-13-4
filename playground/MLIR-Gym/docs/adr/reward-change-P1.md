## ADR: Final-Only Reward (Phase 1)

- Status: Accepted
- Date: 2025-09-20
- Issue: [#31](https://github.com/MPSLab-ASU/MLIR-Gym/issues/31)
- Scope: MLIRGym Reward

### Context:
Need a stable reward aligned with my primary goal -> reducing latency before adding shaping / PBRS.
Current design is un-debuggable and un-testable (becuase of mixed signals)

### Decision:
Adopt a final-reward, latency-centric reward.
Score only at evaluation runner and truncation.
Failures receive a large negative constant.
**Note.** Reward shaping, PBRS are deferred to Phase 2; Observation-change is deferred to another ADR. And Phase 2 change is expected to be tightly coupled with observation-change.

### Final Reward Proposed Design:
$$
R_{\text{final}} = -C
$$

$$
\begin{aligned}
R_{\text{final}}
&= -\log\\left(\frac{\mathrm{lat}}{\mathrm{L0}}\right) \\
&\quad - \lambda \max\\left(0, \frac{\mathrm{ct}}{\mathrm{ct}_{\mathrm{budget}}} - 1\right) \\
&\quad - \beta \max\\left(0, \frac{\mathrm{bytes}}{\mathrm{bytes}_{\mathrm{budget}}} - 1\right).
\end{aligned}
$$

- Definitions
    - lat: measured latency (ms) of the final IR.
    - L0: application latency baseline.
    - ct: measured compile time (ms).
    - $ct_{budget}$ = $\alpha * C0$, where C0 is the application compile-time baseline, default $\alpha = 1.2$
    - $bytes_{budget}$ = $\gamma * B0$, where B0 is the application object file byte-size baseline, default $\gamma = 1.0$
    - $\lambda,\ \beta$ default to 0.
    - $C$ is a large negative constant for failure cost.

### Plan of Change:
- Add an interface for the user to configure reward settings.
- Reward settings are planned to be loaded from yaml in Config.
- Add a rewarder class hook into mlir_world.py. The job includes calculating the final rewards.
- Add ABCs for FinalReward and Potential calculators, and use a global list to store the registries.

### Design:
1. Get RewardConfig from settings.
2. Instantiate a rewarder with RewardConfig in mlir_world.py.
3. The rewarder class will instantiate different classes (FinalReward / Potential := None).
4. Call compute from the instance, which will return different values based on different scenarios.

| Scenarios/factors | ok | term | trun | error | final_reward | potential |
|:-----------------|:---------:|:-----------:|:-----------:|:-------------:|:------------------:|:------------------:|
| Valid runs | True | False | False | NA | d_reward | NA |
| Masked runs | False | False | False | E:MASKED | TBD | NA |
| Run timeout | False | True | False | E:TIMEOUT | c_penalty | NA |
| Run failed | False | True | False | E:FAILED | c_penalty | NA |
| Reach max step | True/False | True | True | NA/E:FAILED | compute(...)+non_step_cost/c_penalty | NA |
| Valid eval | True | True | False | NA | compute(...) | NA |
| Eval Failed | False | True | False | E:FAILED | c_penalty | NA |

### Testing:
- Unit: Test latency ratio signs, budget edges, failure paths, and NaN/<=0 handling.
- Stub e2e: Ensure STOP and truncation both score once, consistent with the formula.
- Real e2e: Add a new test_reward_final_only.py to validate environment reward vs. the formula (with tolerance).
