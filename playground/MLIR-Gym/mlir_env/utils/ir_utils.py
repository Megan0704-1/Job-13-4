# utils/ir_utils.py
# This file defines the utils for manipulating IR
# including:
# - Phase immutable type
# - get IR hash

import re
import hashlib
import os, json, tempfile, threading
from pathlib import Path
from collections import OrderedDict, abc
from typing import Any, Optional, Callable
import numpy as np

from mlir_env.core.immutable import Phase, ErrorBits, Requirement


def detect_phase_from_ir(ir_text: str) -> Phase:
    p = Phase(0)
    if re.search(r"\blinalg\.", ir_text):
        p |= Phase.LINALG
    if re.search(r"\btensor\.", ir_text):
        p |= Phase.TENSOR
    if re.search(r"\bmemref\.", ir_text):
        p |= Phase.MEMREF
    if re.search(r"\bvector\.", ir_text):
        p |= Phase.VECTOR
    if re.search(r"\baffine\.", ir_text):
        p |= Phase.AFFINE
    if re.search(r"\bmath\.", ir_text):
        p |= Phase.MATH
    if re.search(r"\barith\.", ir_text):
        p |= Phase.ARITH
    if re.search(r"\bscf\.", ir_text):
        p |= Phase.SCF
    if re.search(r"\bcf\.", ir_text):
        p |= Phase.CF
    if re.search(r"\bfunc\.", ir_text):
        p |= Phase.FUNC
    if re.search(r"\bllvm\.", ir_text):
        p |= Phase.LLVM
    return p


def to_jsonable(x):
    # Fast path: already JSON-serializable scalars
    if x is None or isinstance(x, (bool, int, float, str)):
        return x

    # numpy
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):  # e.g., np.int64, np.float32, np.bool_
        return x.item()

    # path-like
    if isinstance(x, Path):
        return str(x)

    # bytes-like
    if isinstance(x, (bytes, bytearray, memoryview)):
        try:
            return bytes(x).decode("utf-8")
        except Exception:
            return repr(x)

    # mappings (dict-like): ensure string keys, recurse on values
    if isinstance(x, abc.Mapping):
        return {str(k): to_jsonable(v) for k, v in x.items()}

    # sequences / sets: convert to list and recurse
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]

    # As a last resort: if json can handle it, keep; else stringify
    try:
        json.dumps(x)
        return x
    except TypeError:
        return str(x)


def get_ir_hash(ir_text: str) -> str:
    return hashlib.sha256(ir_text.encode("utf-8")).hexdigest()[:16]


class LRUCache:
    def __init__(
        self, capacity: int = 1024, cache_dir: str = None, namespace: str = None
    ):
        self.capacity = capacity
        self._lock = threading.RLock()
        self.lru_cache: "OrderedDict[str, Any]" = OrderedDict()

        # default cache loc: /home/<user>/gym_cache
        base = Path(cache_dir or os.path.expanduser("~/gym_cache"))
        self.dir = base / namespace
        self.dir.mkdir(parents=True, exist_ok=True)
        self.suffix = ".json"

    def get_path(self, handle: str) -> Path:
        return self.dir / f"{handle}{self.suffix}"

    def lookup(self, ir: str) -> bool:
        handle = get_ir_hash(ir)
        with self._lock:
            if handle in self.lru_cache:
                return True
            return self.get_path(handle).exists()

    def get(self, ir: str) -> Any:
        handle = get_ir_hash(ir)
        with self._lock:
            if handle in self.lru_cache:
                data = self.lru_cache.pop(handle)
                # move to end (MRU)
                self.lru_cache[handle] = data
                return data

            # not in cache
            path = self.get_path(handle)
            if path.exists():
                data = json.loads(path.read_text())
                self.insert_mru(ir, data)
                return data

            assert False, "Should not reach here"

    def insert_mru(self, ir: str, payload: Any) -> None:
        handle = get_ir_hash(ir)
        payload = to_jsonable(payload)

        if handle in self.lru_cache:
            self.lru_cache.pop(handle)

        self.lru_cache[handle] = payload

        while len(self.lru_cache) > self.capacity:
            self.lru_cache.popitem(last=False)

    def add(self, ir: str, payload: Any) -> None:
        handle = get_ir_hash(ir)
        path = self.get_path(handle)
        data = json.dumps(to_jsonable(payload))
        with self._lock:
            self.insert_mru(ir, payload)
            self.atomic_write(path, data)

    @staticmethod
    def atomic_write(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(path.parent), prefix=".tmp_", suffix=path.suffix
        )

        try:
            with os.fdopen(fd, "w") as f:
                f.write(data)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
