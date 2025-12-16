# The utils folder contains utility functions as script listed below

- get_ops.py: This python file generates distince ops used in mlir/test/Dialect/xxx folder
```bash
python get_ops.py -td ~/Polygeist/llvm-project/mlir/include/mlir/Dialect/Affine/IR/AffineOps.td  -n affine
```

After parsing the ops in dialect, there are 2 places to modify
1. main test.py file in project root (if you want to lower to the dialect)
2. mlir_env/utils/config.py: get_lowering_dialect for it to be visiable to the program
