# views/spec.py

from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum


@dataclass(frozen=True)
class ViewSpec:
    name: str
    passes: List[str]
    requires: Dict[str, bool]  # requirements
    fallback: Optional[str] = None  # alternative viewkind


class ViewKind(str, Enum):
    GRAPH = "graph"
    AFFINE = "affine"
    SCF = "scf"
    MEMREF = "memref"
    VECTOR = "vector"


VIEW_SPECS: Dict[ViewKind, ViewSpec] = {
    ViewKind.GRAPH: ViewSpec(
        name="graph",
        passes=[
            "--linalg-generalize-named-ops",
            "--canonicalize",
            "--cse",
            "--affine-simplify-structures",
            "--sccp",
            "--canonicalize",
            "--cse",
        ],
        requires={"oneshot": False, "affine": False, "vectorize": False},
        fallback=None,
    ),
    ViewKind.AFFINE: ViewSpec(
        name="affine",
        passes=[
            "--linalg-generalize-named-ops",
            "--convert-bufferization-to-memref",
            "--one-shot-bufferize=bufferize-function-boundaries",
            "--canonicalize",
            "--cse",
            "--convert-linalg-to-affine-loops",
            "--affine-simplify-structures",
            "--sccp",
            "--canonicalize",
            "--cse",
        ],
        requires={"oneshot": True, "affine": True, "vectorize": False},
        fallback=ViewKind.SCF,
    ),
    ViewKind.SCF: ViewSpec(
        name="scf",
        passes=[
            "--linalg-generalize-named-ops",
            "--convert-bufferization-to-memref",
            "--one-shot-bufferize=bufferize-function-boundaries",
            "--expand-strided-metadata",
            "--convert-linalg-to-loops",
            "--canonicalize",
            "--cse",
        ],
        requires={"oneshot": True, "affine": False, "vectorize": False},
        fallback=ViewKind.MEMREF,
    ),
    # try the best to simplify array constructs for memory analysis
    ViewKind.MEMREF: ViewSpec(
        name="memref",
        passes=[
            "--canonicalize",
            "--cse",
            "--linalg-generalize-named-ops",
            "--convert-bufferization-to-memref",
            "--one-shot-bufferize=bufferize-function-boundaries",
            "--normalize-memrefs",
            "--affine-simplify-structures",
            "--expand-strided-metadata",
            "--sccp",
            "--canonicalize",
            "--cse",
        ],
        requires={"oneshot": True, "affine": False, "vectorize": False},
        fallback=ViewKind.GRAPH,
    ),
    # TODO(megan.kuo) placeholders
    ViewKind.VECTOR: ViewSpec(
        name="vector",
        passes=[
            "--linalg-generalize-named-ops",
            "--one-shot-bufferize=bufferize-function-boundaries",
            "--linalg-vectorize",
            "--canonicalize",
            "--cse",
        ],
        requires={"oneshot": True, "vectorize": True, "affine": False},
        fallback=ViewKind.SCF,
    ),
}
