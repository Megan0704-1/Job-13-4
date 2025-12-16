#! /bin/bash
export PYTHONPATH=$HOME/workspace/install/lib/python:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/workspace/install/lib:${LD_LIBRARY_PATH}
export LLVM_BUILD_DIR=$HOME/workspace/llvm-project/build
export MLIR_GYM_ROOT=$HOME/workspace/branches/change-3
export LLVM_TOOL_DIR=$HOME/workspace/llvm-project/build/bin

conda activate mlir-gym
