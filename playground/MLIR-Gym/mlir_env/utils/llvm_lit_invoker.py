import subprocess
from mlir_env.utils.config import get_common_config


def run_lit(path, flags):
    llvm_lit_bin = get_common_config()["llvm_lit"]
    try:
        cmd = [llvm_lit_bin, "-v", path, "--param", f"custom-pipeline={flags}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return "Passed" in result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error in llvm lit: {e}")
        return False
