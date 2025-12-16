# -*- Python -*-
import os
import lit.formats

# Basic config
config.name = "MLIRGym"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".mlir", ".ll", ".td", ".txt"]
config.excludes = ["Inputs", "CMakeLists.txt", "lit.site.cfg.py", "lit.cfg.py"]

# Roots
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root  # run in-place

# Env: allow tests to find shared libs if needed
ld_path_var = "LD_LIBRARY_PATH" if os.name != "nt" else "PATH"
for var in [ld_path_var, "PATH"]:
    config.environment[var] = os.environ.get(var, "")

# Tool substitutions (prefer PATH discovery)
tools = ["mlir-opt", "mlir-translate", "FileCheck", "not", "count"]

tool_dirs = []
for env_var in ["LLVM_TOOL_DIR"]:
    d = os.environ.get(env_var)
    if d:
        tool_dirs.append(d)


# Small helper: search for each tool in tool_dirs then PATH
def which(tool, extra_dirs):
    for d in extra_dirs:
        p = os.path.join(d, tool)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    # Fallback to PATH
    for p in os.environ.get("PATH", "").split(os.pathsep):
        cand = os.path.join(p, tool)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
        if os.path.isfile(cand + ".exe"):
            return cand + ".exe"
    return tool  # let lit fail with a nice message


# Register tool substitutions like %mlir_opt, %FileCheck, etc.
for t in tools:
    path = which(t, tool_dirs)
    config.substitutions.append((f'%{t.replace("-", "_")}', path))

# Convenience aliases matching LLVM style
config.substitutions += [
    ("%mlir-opt", which("mlir-opt", tool_dirs)),
    ("%filecheck", which("FileCheck", tool_dirs)),
]

# Optional features
if which("mlir-opt", tool_dirs) == "mlir-opt":
    pass  # will fail at runtime if missing
