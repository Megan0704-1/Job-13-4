import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Sequence, Dict, Any, Optional
import argparse

parser = argparse.ArgumentParser(description="input a result file to generate plot")

parser.add_argument("--result-file", required=True)

args = parser.parse_args()

COLOR_METRIC = "#f5a623" 
COLOR_MOVING = "#2e86de"
COLOR_BEST   = "#e74c3c"

def load_metrics_from_txt(path: str) -> np.ndarray:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    vals = [float(x) for x in re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", txt)]
    return np.asarray(vals, dtype=float)

def load_metrics_from_log(path: str) -> np.ndarray:
    vals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"Get return:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", line)
            if m:
                vals.append(float(m.group(1)))
    return np.asarray(vals, dtype=float)

def _moving_avg_same(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    out = np.empty_like(x, dtype=float)
    csum = np.cumsum(np.insert(x, 0, 0.0))
    for i in range(len(x)):
        start = max(0, i - window + 1)
        count = i - start + 1
        out[i] = (csum[i + 1] - csum[start]) / count
    return out

def plot_metric_over_time(metrics: Sequence[float],
                          window: int = 10,
                          title: str = "Metric over time",
                          ylabel: str = "Metric (speedup proxy)",
                          savepath: Optional[str] = None,
                          show: bool = True,
                          ax: Optional[Any] = None) -> Dict[str, np.ndarray]:
    y = np.asarray(metrics, dtype=float)
    x = np.arange(1, len(y) + 1, dtype=int)
    moving = _moving_avg_same(y, window)
    best = np.maximum.accumulate(y)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(x, y, label="metric", alpha=0.9, color=COLOR_METRIC, linewidth=1.8)
    ax.plot(x, moving, label=f"moving avg ({window})", color=COLOR_MOVING, linewidth=2.2)
    ax.plot(x, best, label="best so far", color=COLOR_BEST, linewidth=2.2)

    ax.set_title(title)
    ax.set_xlabel("Attempt")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    if show:
        plt.show()

    return {"x": x, "metric": y, "moving": moving, "best": best}

if __name__ == "__main__":
    metrics = load_metrics_from_txt(f"{args.result_file}")  
    plot_metric_over_time(metrics, window=10, title="Metric over time",
                          savepath="metric_over_time.png")

