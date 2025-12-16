# mlir_env/utils/benchmark_utils.py
import re
from typing import Dict, Union, Optional
from mlir_env.core.immutable import Metrics
from contextlib import contextmanager
import os, tempfile
from pathlib import Path
import json

_UNIT_TO_KB = {"B": 1 / 1024, "KB": 1, "MB": 1024, "GB": 1024 * 1024}


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse_baseline_metrics(stdout: Union[bytes, str]) -> Metrics:
    """Parse the 'Baseline Metrics' block and return Metrics(lat_ms, c_ms, kb)."""
    text = (
        stdout.decode("utf-8", errors="ignore")
        if isinstance(stdout, (bytes, bytearray))
        else stdout
    )
    text.strip()

    try:
        obj = json.loads(text)
        return Metrics(
            latency_ms=float(obj["latency_ms"]),
            compile_ms=float(obj["compile_ms"]),
            size_kb=float(obj["size_kb"]),
        )
    except:
        pass
    mjson = re.search(
        r'\{[^{}]*"compile_ms"[^{}]*"latency_ms"[^{}]*"size_kb"[^{}]*\}',
        text,
        flags=re.DOTALL,
    )
    if mjson:
        obj = json.loads(mjson.group(0))
        return Metrics(
            latency_ms=float(obj["latency_ms"]),
            compile_ms=float(obj["compile_ms"]),
            size_kb=float(obj["size_kb"]),
        )

    # Fallback: parse human-readable metrics anywhere in the text
    c0m = re.search(r"C0.*?:\s*([\d,]+(?:\.\d+)?)\s*ms", text, flags=re.IGNORECASE)
    l0m = re.search(r"L0.*?:\s*([\d,]+(?:\.\d+)?)\s*ms", text, flags=re.IGNORECASE)
    b0m = re.search(
        r"B0.*?:\s*([\d,]+(?:\.\d+)?)\s*([KMG]?B)", text, flags=re.IGNORECASE
    )
    if not (c0m and l0m and b0m):
        snippet = text[:400].replace("\n", "\\n")
        raise ValueError(f"Failed to parse metrics. Snippet: {snippet}")
    c0_ms = _num(c0m.group(1))
    l0_ms = _num(l0m.group(1))
    b0_val = _num(b0m.group(1))
    unit = b0m.group(2).upper()
    b0_kb = b0_val * _UNIT_TO_KB.get(unit, 1)
    return Metrics(latency_ms=l0_ms, compile_ms=c0_ms, size_kb=b0_kb)


@contextmanager
def mktemp(ir_text: str, filename: Optional[str] = "run.mlir"):
    """
    TBD(megan.kuo) not yet decided to write to a file or not.
    As a temporary solution, a temp file which will later get cleanup is applied.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(ir_text)
        yield path
