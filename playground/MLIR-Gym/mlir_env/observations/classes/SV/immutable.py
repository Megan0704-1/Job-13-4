# mlir_env/observations/classes/SV/immutable.py
# this file declares the parameters used for different analyses

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any


@dataclass
class LoopParams:
    max_loops: int = (
        4  # maximum number of loops to parse, nested loops are counted separately.
    )
    normalize_depth: bool = True


@dataclass
class MemParams:
    vector_width_elements: int = 8
    unit_stride_policy: str = "store_first"
    compute_bonus: bool = True


@dataclass
class HardwareParams:
    source: Literal["yaml", "sysfs"] = "yaml"
    fields: List[str] = field(
        default_factory=lambda: [
            "L1",
            "L2",
            "L3",
            "cache_line",
            "simd_bytes",
            "reg_bytes",
            "mem_bw",
            "mem_lat",
            "cores",
            "shared_kb",
            "banks",
            "page_kb",
        ]
    )
    values: Optional[List[float]] = None  # required if source="yaml"


@dataclass
class PackerParams:
    normalize: Dict[str, Any] = field(
        default_factory=lambda: {"method": "log1p_minmax", "clip": [0.0, 1.0]}
    )
    include_channels: List[str] = field(default_factory=lambda: ["loop"])
