import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def plot_improvement(df, out_png, out_pdf, title):
    df = df.dropna(subset=["baseline_ms (-O3)", "optimized_ms (Your approach)"]).copy()
    if df.empty:
        print("No data to plot yet. Fill in the CSV first.")
        return
    
    df["improvement_%"] = (df["baseline_ms (-O3)"] - df["optimized_ms (Your approach)"]) / df["baseline_ms (-O3)"] * 100.0
    df["speedup_x"] = df["baseline_ms (-O3)"] / df["optimized_ms (Your approach)"]
    df = df.sort_values("improvement_%", ascending=False)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(df["benchmark"], df["improvement_%"])
    for rect, val in zip(bars, df["improvement_%"]):
        height = rect.get_height()
        label = f"{val:.0f}%"
        if val >= 50:
            label += " ★"
        plt.text(rect.get_x() + rect.get_width()/2.0, height, label, ha="center", va="bottom", fontsize=8)
    plt.title(title)
    plt.ylabel("Runtime reduction vs -O3 (%)")
    plt.xlabel("Benchmark")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()

def main():
    in_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmark_results.csv")
    df = pd.read_csv(in_csv)
    png_path = Path("performance_improvement_bar.png")
    pdf_path = Path("performance_improvement_bar.pdf")
    plot_improvement(df, png_path, pdf_path, "Performance improvement per benchmark")
    print(f"Saved: {png_path} and {pdf_path}")

if __name__ == "__main__":
    main()
