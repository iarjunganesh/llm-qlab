"""
compare_quants.py — Compare quantization benchmark results.

Reads results/benchmark_results.csv, produces comparison bar charts
(tokens/sec and VRAM usage), saves them to results/comparison.png,
and prints a markdown-formatted comparison table to stdout.

Usage:
    python compare_quants.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "benchmark_results.csv"
OUTPUT_PNG = RESULTS_DIR / "comparison.png"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_results() -> pd.DataFrame:
    if not CSV_PATH.exists():
        print(f"[error] Results file not found: {CSV_PATH}")
        print("Run benchmark.py first to generate results.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("[error] Results file is empty. Run benchmark.py first.")
        sys.exit(1)

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # Aggregate: mean per quant_type
    agg = (
        df.groupby("quant_type", sort=False)
        .agg(gen_tps=("gen_tps", "mean"), vram_mb=("vram_mb", "mean"))
        .reset_index()
    )

    quant_labels = agg["quant_type"].tolist()
    x = range(len(quant_labels))
    bar_width = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("LLM Quantization Comparison — llm-qlab", fontsize=14, fontweight="bold")

    # --- Chart 1: Tokens/sec ---
    axes[0].bar(x, agg["gen_tps"], width=bar_width, color="#4C9BE8", edgecolor="white")
    axes[0].set_title("Generation Speed (tokens/sec)")
    axes[0].set_xlabel("Quantization Format")
    axes[0].set_ylabel("Tokens / second")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(quant_labels)
    for i, v in enumerate(agg["gen_tps"]):
        axes[0].text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    # --- Chart 2: VRAM usage ---
    axes[1].bar(x, agg["vram_mb"], width=bar_width, color="#E8844C", edgecolor="white")
    axes[1].set_title("VRAM Usage (MB)")
    axes[1].set_xlabel("Quantization Format")
    axes[1].set_ylabel("VRAM (MB)")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(quant_labels)
    for i, v in enumerate(agg["vram_mb"]):
        axes[1].text(i, v + 5, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"Chart saved to {OUTPUT_PNG}")


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def print_markdown_table(df: pd.DataFrame) -> None:
    agg = (
        df.groupby("quant_type", sort=False)
        .agg(
            gen_tps=("gen_tps", "mean"),
            prompt_tps=("prompt_tps", "mean"),
            vram_mb=("vram_mb", "mean"),
            load_time_s=("load_time_s", "mean"),
            model_size_mb=("model_size_mb", "mean"),
        )
        .reset_index()
    )

    header = (
        "| Quant | Gen t/s | Prompt t/s | VRAM (MB) | Load (s) | Size (MB) |"
    )
    separator = "|-------|---------|------------|-----------|----------|-----------|"

    print("\n## Benchmark Results\n")
    print(header)
    print(separator)
    for _, row in agg.iterrows():
        print(
            f"| {row['quant_type']} "
            f"| {row['gen_tps']:.2f} "
            f"| {row['prompt_tps']:.2f} "
            f"| {row['vram_mb']:.0f} "
            f"| {row['load_time_s']:.2f} "
            f"| {row['model_size_mb']:.0f} |"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_results()
    plot_comparison(df)
    print_markdown_table(df)


if __name__ == "__main__":
    main()
