# The MLIRGym customize MLIR pass
1. termination_pass

- Unfortunately, there is currently only one pass available
- To build it run the following in the terminal
```bash
cd /root/compiler_gym/mlir_playground/backend/passes && mkdir -p build
cd build && rm -rf * && cmake -G Ninja .. -DMLIR_DIR=/root/llvm-project/build/lib/cmake/mlir/ -DCMAKE_INSTALL_PREFIX=/root/compiler_gym/mlir_playground/install/ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ninja && ninja install
```

#TODO automate the shared libarory and set PYTHONPATH process
the shared library of mlir_gym_compiler.so is now installed at your specified cmake install prefix
update env python variable to search for the library
```bash
export PYTHONPATH=path/to/install/lib/python:$PYTHONPATH
export LD_LIBRARY_PATH=path/to/install/lib:${LD_LIBRARY_PATH}
```

# Debug
```bash
ldd path/to/install/lib/libMLIRGymCompiler.so
```
