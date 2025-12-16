## Environment setup
### LLVM + MLIR
Build LLVM with PythonBindings, put the build into separate directory. (e.g., /home/llvm-project)
*note.* llvm build cmake variables can refer to llvm-build CI.

ref: https://mlir.llvm.org/docs/Bindings/Python/
```bash
python -m pip install -r mlir/python/requirements.txt
# Run mlir tests. For example, to run python bindings tests only using ninja:
ninja check-mlir-python
```
```bash
# export python path to your system
export PYTHONPATH=$(cd build && pwd)/tools/mlir/python_packages/mlir_core
```

### MLIR-Gym
Go to envs/install.sh and update the paths
```bash
vim envs/install.sh
# change the path to your custom settings
source envs/install.sh
```
Autotune with specific CPU core
```bash
CPU_CORE=31 python run.py
```

## Feature testing
```bash
python -m pytest tests/
```

## Lit testing
```bash
llvm-lit lit_tests/
```

## Formatting
```bash
pip install pre-commit
pre-commit install
```

## Example run
```bash
# [] means optional
# <> required-configurables

# 1. e2e sanity run
CPU_CORE=11 python run.py \
    e2e \
    --cfg-path <benchmarks/settings/gemm_config.yaml> \
    --ctx-path <benchmarks/settings/mlp_bias_relu_aot_context.yaml> \
    --state-cfg-path <settings/observations/graph_config.yml>

# 2. q-learning run
CPU_CORE=11 python run.py \
    q-learn \
    --cfg-path <benchmarks/settings/gemm_config.yaml> \
    --ctx-path <benchmarks/settings/mlp_bias_relu_aot_context.yaml> \
    --state-cfg-path <settings/observations/graph_config.yml>
    --out-path <trial_11.result> 

# 3. gnn training run
CPU_CORE=11 python run.py \
    train \
    --cfg-path <benchmarks/settings/gemm_config.yaml> \
    --ctx-path <benchmarks/settings/mlp_bias_relu_aot_context.yaml> \
    --state-cfg-path <settings/observations/graph_config.yml>
    --ppo-cfg <settings/algo/ppo_gnn.yaml> 

# 4. gnn eval run
CPU_CORE=11 python run.py \
    eval \
    --cfg-path <benchmarks/settings/gemm_config.yaml> \
    --ctx-path <benchmarks/settings/mlp_bias_relu_aot_context.yaml> \
    --state-cfg-path <settings/observations/graph_config.yml>
    --model-path <out/ppo_gnn_model.zip>
```

TODO(Issue #30)
1. [General] Gather config, context and application paths and combine to one setting files
2. [Action] TBD noop_window, past_actions
3. [Action] Action masking before sampling
4. [Reward] TBD PBRS, reward shaping
5. [Reward] Autotune the step params (sample declared in settings)
6. [Views] Add vector views
7. [State] Add hardware parameters and probes analysis
