import lit.util
import lit.formats

config.name = "MLIR Lit Test"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]
config.substitutions = [
    ("%mlir-opt", "/root/llvm-project/build/bin/mlir-opt"),
    ("%mlir-translate", "/root/llvm-project/build/bin/mlir-translate"),
    ("%mlir-cpu-runner", "/root/llvm-project/build/bin/mlir-cpu-runner"),
    ("%mlir_runner_utils", "/root/llvm-project/build/lib/libmlir_runner_utils.so"),
    ("%mlir_c_runner_utils", "/root/llvm-project/build/lib/libmlir_c_runner_utils.so"),
    ("%FileCheck", "/root/llvm-project/build/bin/FileCheck"),
    ("%clang", "/root/llvm-project/build/bin/clang"),
    (
        "%link-dir",
        "-lm -lmlir_runner_utils -lmlir_c_runner_utils -L/root/llvm-project/build/lib",
    ),
]

# conditional invoke for llvm lit
custom_pipeline = lit_config.params.get("custom-pipeline", "")
if custom_pipeline:
    config.available_features.add("request_custom_pipeline")
    config.substitutions.extend(
        [("%{custom-pipeline}", lit_config.params.get("custom-pipeline", ""))]
    )
