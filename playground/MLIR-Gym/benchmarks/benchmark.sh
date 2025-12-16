#!/usr/bin/env bash
set -euo pipefail

# ---- usage ----
# benchmarks/benchmark.sh \
#   --mlir path/to/program.mlir \
#   --build $LLVM_BUILD_DIR \
#   --entry main \
#   --runs 9 \
#   --pipeline "--pass-pipeline=builtin.module(func.func(canonicalize,cse),...)" \
#   --json      # optional: emit JSON only

MLIR_FILE=""
BUILD_DIR=""
ENTRY="main"
RUNS=9
PIPELINE=""
SHARED_LIBS="libmlir_runner_utils.so,libmlir_c_runner_utils.so"
JSON=0
WARMUP=1
LLVM_O=3
BACKEND="jit"
CPU_CORE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mlir)      MLIR_FILE="$2"; shift 2;;
    --build)     BUILD_DIR="$2"; shift 2;;
    --entry)     ENTRY="$2"; shift 2;;
    --runs)      RUNS="$2"; shift 2;;
    --pipeline)  PIPELINE="$2"; shift 2;;
    --shared-libs) SHARED_LIBS="$2"; shift 2;;
    --json)      JSON=1; shift 1;;
    --warmup)    WARMUP="$2"; shift 2;;
    --llvm-O)    LLVM_O="$2"; shift 2;;
    --backend)   BACKEND="$2"; shift 2;;
    --cpu)       CPU_CORE="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "${MLIR_FILE}" || -z "${BUILD_DIR}" ]]; then
  echo "ERROR: --mlir and --build are required" >&2
  exit 1
fi

BIN="${BUILD_DIR}/bin"
LIB="${BUILD_DIR}/lib"

mlir_opt="${BIN}/mlir-opt"
mlir_translate="${BIN}/mlir-translate"
mlir_runner="${BIN}/mlir-runner"
llc="${BIN}/llc"
opt="${BIN}/opt"
clang="${BIN}/clang"

for tool in "$mlir_opt" "$mlir_translate" "$mlir_runner" "$llc" "$opt" "$clang"; do
  [[ -x "$tool" ]] || { echo "missing executable: $tool" >&2; exit 1; }
done

# file size (portable)
if stat --version >/dev/null 2>&1; then
  STAT_CMD='stat -c %s'
elif stat -f %H . >/dev/null 2>&1; then
  STAT_CMD='stat -f %z'
else
  STAT_CMD='wc -c <'
fi

# Python helpers
median_py() { python3 - "$@" <<'PY'
import sys, statistics
xs = [float(x) for x in sys.argv[1:]]
print(f"{statistics.median(xs):.6f}")
PY
}
stdev_py() { python3 - "$@" <<'PY'
import sys, statistics
xs = [float(x) for x in sys.argv[1:]]
print(f"{(statistics.pstdev(xs) if xs else 0.0):.6f}")
PY
}
ms_py() { python3 - "$@" <<'PY'
import sys
print(f"{float(sys.argv[1])*1000.0:.2f}")
PY
}

# pin CPU (best-effort)
export OMP_NUM_THREADS=1
PIN=""
if command -v taskset &>/dev/null; then PIN="taskset -c ${CPU_CORE}"; fi
if command -v numactl &>/dev/null; then PIN="$PIN numactl --physcpubind=${CPU_CORE} --localalloc"; fi

# parse pipeline into argv array
OPT_ARGS=()
if [[ -n "$PIPELINE" ]]; then
  if [[ "$PIPELINE" == --pass-pipeline=* ]]; then
    OPT_ARGS=( "$PIPELINE" )
  else
    # split by space into multiple single-pass flags
    read -r -a OPT_ARGS <<< "$PIPELINE"
  fi
fi


# temp files & cleanup
LOWERED_MLIR="$(mktemp --suffix=.mlir)"
LL_FILE="$(mktemp --suffix=.ll)"
LL_OPT_FILE="$(mktemp --suffix=.opt.ll)"
OBJ_FILE="$(mktemp --suffix=.o)"
EXEC_FILE="$(mktemp --suffix=.exe)"
cleanup() { rm -f "$LOWERED_MLIR" "$LL_FILE" "$OBJ_FILE" "$LL_OPT_FILE" "$EXEC_FILE"; }
trap cleanup EXIT

# compile wall-time (mlir-opt passes only)
# warmup
echo "[warm up] compiling..."
for _ in $(seq 1 "$WARMUP"); do
  $PIN "$mlir_opt" "${OPT_ARGS[@]}" "$MLIR_FILE" -o /dev/null 1>/dev/null || true
done

echo "[measurement] mlir-opt..."
C_TIMES=()
for _ in $(seq 1 "$RUNS"); do
  T=$({ /usr/bin/time -f "%e" $PIN \
    "$mlir_opt" "${OPT_ARGS[@]}" "$MLIR_FILE" -o /dev/null 1>/dev/null ; } 2>&1)
  C_TIMES+=("$T")
done
C0=$(median_py "${C_TIMES[@]}")
CSD=$(stdev_py "${C_TIMES[@]}")

# lower to llvm.mlir & compile to object
echo "[preparing] lowering..."
"$mlir_opt" "${OPT_ARGS[@]}" "$MLIR_FILE" -o "$LOWERED_MLIR"
"$mlir_translate" --mlir-to-llvmir "$LOWERED_MLIR" -o "$LL_FILE"

# latency measurements (JIT / AOT)
# 1) linked shared libs
if [[ "$BACKEND" == "jit" ]]; then
    echo "[JIT preparing] linking..."
    "$llc" -filetype=obj "$LL_FILE" -o "$OBJ_FILE"
else
    echo "[AOT preparing] linking..."
    # run standard llvm ir optimization pipeline at -O<level>
    "$opt" "-O${LLVM_O}" "$LL_FILE" -o "$LL_OPT_FILE"
    # codegen
    "$llc" "-O=${LLVM_O}" -filetype=obj "$LL_OPT_FILE" -o "$OBJ_FILE"

    # shared libs
    LIB_LINKS=()
    IFS=, read -r -a _libs <<< "$SHARED_LIBS"
    for lib in "${_libs[@]}"; do
        if [[ -f "${LIB}/${lib}" ]]; then
            LIB_LINKS+=( "-L" "$LIB" "-Wl,-rpath,$LIB" "-l:${lib}" "-lm" )
        else
            LIB_LINKS+=( "-l:${lib}" )
        fi
    done
    "$clang" "-O${LLVM_O}" "$OBJ_FILE" -o "$EXEC_FILE" "${LIB_LINKS[@]}"
fi

# 2) measure execution time
# latency via mlir-runner (JIT), warmup & median
L_TIMES=()
if [[ "$BACKEND" == "jit" ]]; then
    echo "[JIT measurement] executing..."
    for _ in $(seq 1 "$RUNS"); do
      T=$({ /usr/bin/time -f "%e" $PIN \
          "$mlir_runner" -O3 -e "$ENTRY" -entry-point-result=void \
          $(IFS=,; for lib in $SHARED_LIBS; do [[ -f "${LIB}/${lib}" ]] && printf ' -shared-libs=%s' "${LIB}/${lib}" || printf ' -shared-libs=%s' "$lib"; done) \
          < "$LOWERED_MLIR" 1>/dev/null ; } 2>&1)
      L_TIMES+=("$T")
    done
else
    echo "[AOT warm up measurement] executing..."
    for _ in $(seq 1 "$WARMUP"); do
        $PIN "$EXEC_FILE" 1>/dev/null || true
    done
    echo "[AOT real measurement] executing..."
    for _ in $(seq 1 "$RUNS"); do
        # gemm.mlir returns a tensor of values, leads to non 0 return code
        T=$({ /usr/bin/time -f "%e" $PIN "$EXEC_FILE" 1>/dev/null; true; } 2>&1 | tail -n1)
        L_TIMES+=("$T")
    done
fi
L0=$(median_py "${L_TIMES[@]}")
LSD=$(stdev_py "${L_TIMES[@]}")

# byte measurement
echo "[size measurement] evaluating..."
if [[ "$STAT_CMD" == "wc -c <" ]]; then
  B0_BYTES=$(bash -c "$STAT_CMD \"$OBJ_FILE\"")
else
  B0_BYTES=$($STAT_CMD "$OBJ_FILE")
fi
B0_KB=$(python3 - <<PY
print(f"{float('$B0_BYTES')/1024.0:.2f}")
PY
)


# emit results
if [[ "$JSON" -eq 1 ]]; then
  # JSON only (stdout)
  python3 - <<PY
import json
print(json.dumps({
  "compile_ms": float("${C0}")*1000.0,
  "latency_ms": float("${L0}")*1000.0,
  "size_kb": float("${B0_KB}"),
  "runs": int("${RUNS}"),
  "warmup": int("${WARMUP}"),
  "compile_stdev_over_s": float("${CSD}"),
  "latency_stdev_over_s": float("${LSD}"),
  "backend": "${BACKEND}"
}))
PY
else
  echo "================== Baseline Metrics - (${BACKEND}) =================="
  echo "C0 (compile_time_ms, pass-only): $(ms_py "$C0") ms"
  echo "B0 (code_bytes, object size):    ${B0_KB} KB"
  echo "L0 (latency_ms, runner median):  $(ms_py "$L0") ms"
  echo "======================================================"
  echo ""
  echo "Note."
  echo "- C0 measures mlir-opt passes only (no codegen/JIT)"
  echo "- L0 uses mlir-runner median to reduce JIT variability"
  echo "- B0 is object file size in KB"
fi
