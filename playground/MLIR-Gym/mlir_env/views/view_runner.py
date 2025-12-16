# views/view_runner.py

from __future__ import annotations
import hashlib, subprocess, os, shlex, threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

from mlir_env.views.spec import ViewKind, VIEW_SPECS
from mlir_env.views.utils import ToolCaps
from mlir_env.actions.executor import SubprocessExecutor, CommandSpec


class ViewRunner:
    def __init__(self, cfg: "Config", executor: SubprocessExecutor | None = None):
        self.mlir_opt = cfg.mlir_opt_bin
        self.timeout_s = cfg.apply_timeout_s

        self.cache_enabled = cfg.viewer_cache_enabled
        self._cache_cap = cfg.viewer_cache_cap
        self._cache: OrderedDict[str,str] = OrderedDict()
        self._lock = threading.Lock()

        self.exec = executor or SubprocessExecutor()
        self._caps: ToolCaps | None = None

    def _probe(self) -> ToolCaps:
        if self._caps:
            return self._caps
        ver = subprocess.run(
            [self.mlir_opt, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        def ok(pass_flag: str) -> bool:
            try:
                self.exec.run(
                    CommandSpec(argv=[self.mlir_opt, pass_flag], timeout_s=5),
                    b"module {}",
                )
                return True
            except Exception:
                return False

        caps = ToolCaps(
            version=ver.stdout.decode(errors="ignore").strip(),
            oneshot=ok("--one-shot-bufferize=bufferize-function-boundaries"),
            affine=ok("--convert-linalg-to-affine-loops"),
        )
        self._caps = caps
        return caps

    def _resolve_spec(self, kind: ViewKind) -> list[str]:
        caps = self._probe()
        seen = set()
        while True:
            spec = VIEW_SPECS[kind]
            req = spec.requires
            if (
                (not req.get("oneshot", False) or caps.oneshot)
                and (not req.get("affine", False) or caps.affine)
                and (not req.get("vectorize", False) or caps.vectorize)
            ):
                return spec.passes
            if spec.fallback is None or spec.fallback in seen:
                raise RuntimeError(f"View {kind} not supported by tool caps {caps}")
            seen.add(kind)
            kind = spec.fallback

    def _cache_key(self, ir: str, view: ViewKind, pipeline: list[str]) -> str:
        h = hashlib.sha256()
        h.update(self._probe().version.encode())
        h.update(view.value.encode())
        for p in pipeline:
            h.update(p.encode())
        h.update(ir.encode())
        return h.hexdigest()

    def view(self, ir_text: str, view: ViewKind) -> str:
        pipeline = self._resolve_spec(view)
        key = self._cache_key(ir_text, view, pipeline)

        if self.cache_enabled:
            with self._lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key]

        res = self.exec.run(
            CommandSpec(argv=[self.mlir_opt, *pipeline], timeout_s=self.timeout_s),
            ir_text.encode(),
        )

        if res.timed_out:
            raise TimeoutError(
                f"mlir-opt timed out for view={view} after {self.timeout_s}s"
            )
        if res.returncode != 0:
            stderr = res.stderr.decode("utf-8", errors="ignore").splitlines()
            raise RuntimeError(
                f"[{view}] pipeline failed\nargv={[self.mlir_opt,*pipeline]}\nstderr:\n"
                + "\n".join(stderr[:30])
            )

        out = res.stdout.decode("utf-8", errors="ignore")

        # LRU
        if self.cache_enabled:
            with self._lock:
                self._cache[key] = out
                self._cache.move_to_end(key)
                if len(self._cache) > self._cache_cap:
                    self._cache.popitem(last=False)
        return out
