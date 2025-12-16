import numpy as np
import json
import yaml
from ..configs import TrainConfig

def bin012(x: int) -> int:
    return 0 if x <= 0 else (1 if x == 1 else 2)

def bin3f(x: float) -> int:
    # 0..1 -> categorize as 3 buckets -> 0:[0,0.33), 1:[0.33,0.66), 2:[0.66,1]
    x = 0.0 if not np.isfinite(x) else max(0.0, min(1.0, float(x)))
    if x < 1/3: return 0
    if x < 2/3: return 1
    return 2

def bin3i(n: int, t1: int = 8, t2: int = 32) -> int:
    # 0: <=8, 1: <=32, 2: >32
    if n <= t1: return 0
    if n <= t2: return 1
    return 2


def array(data, dtype = None):
    if dtype is None:
        dtype = type(data[0])
    return np.asarray(data, dtype=dtype)

def get_col(data: np.ndarray, mask: np.ndarray, idx: int) -> np.ndarray:
    if data.ndim != 2 or idx < 0 or idx >= data.shape[1]:
        return np.zeros((int(mask.sum()) if mask.size else 0,), dtype=np.float32)
    return data[mask, idx]

def load_train_config(path: str) -> TrainConfig:
    with open(path, "r") as fh:
        if path.endswith(".json"):
            raw = json.load(fh)
        else:
            raw = yaml.safe_load(fh)
    return TrainConfig.from_dict(raw)
