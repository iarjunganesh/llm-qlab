"""
compare_quants.py — Compare quantization benchmark results.

Reads results/benchmark_results.csv, produces comparison bar charts
(decode throughput and VRAM usage), saves them to results/comparison.png,
and prints a markdown-formatted comparison table to stdout.

Usage:
    python compare_quants.py
    python compare_quants.py --group-by model_family

Rows whose metrics are unmeasured (-1) are excluded from aggregates rather
than averaged in as zeros — a missing measurement must never quietly drag a
reported number down. Rows written before the prefill-timing fix carry
timing_source="legacy_invalid" and are reported separately.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from results_schema import CSV_FIELDS, CSV_PATH, LEGACY_MARKER, normalize_row

RESULTS_DIR = Path("results")
OUTPUT_PNG = RESULTS_DIR / "comparison.png"
OUTPUT_PNG_FAMILY = RESULTS_DIR / "comparison_by_family.png"

# Schema handling is shared with benchmark.py — see results_schema.py.


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _load_resilient() -> pd.DataFrame:
    with open(CSV_PATH, "r", newline="") as f:
        all_rows = list(csv.reader(f))
    if len(all_rows) <= 1:
        return pd.DataFrame(columns=CSV_FIELDS)
    rows = [r for r in (normalize_row(raw) for raw in all_rows[1:]) if r is not None]
    return pd.DataFrame(rows, columns=CSV_FIELDS)


def load_results() -> pd.DataFrame:
    if not CSV_PATH.exists():
        print(f"[error] Results file not found: {CSV_PATH}")
        print("Run benchmark.py first to generate results.")
        sys.exit(1)

    try:
        df = pd.read_csv(CSV_PATH)
        if not set(CSV_FIELDS).issubset(df.columns):
            df = _load_resilient()
    except Exception:
        df = _load_resilient()

    if df.empty:
        print("[error] Results file is empty. Run benchmark.py first.")
        sys.exit(1)

    stale = df[df["timing_source"] == LEGACY_MARKER]
    if not stale.empty:
        print(
            f"[warn] {len(stale)} row(s) predate the prefill-timing fix and have no "
            "valid throughput. They are excluded from charts and tables — re-run "
            "benchmark.py for those configurations."
        )

    return df


def _valid(series: pd.Series) -> pd.Series:
    """Drop unmeasured sentinels so they cannot skew an aggregate."""
    return series[series > 0]


def _median_or_nan(series: pd.Series) -> float:
    usable = _valid(series)
    return usable.median() if not usable.empty else float("nan")


def _aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    metrics = [
        "decode_tps", "decode_tps_std", "prefill_tps", "prefill_tps_std",
        "ttft_ms", "ttft_ms_std", "vram_delta_mb", "load_time_s", "model_size_mb",
    ]
    return (
        df.groupby(keys, sort=False)
        .agg({m: _median_or_nan for m in metrics})
        .reset_index()
    )


def _fmt(value: float, decimals: int = 2) -> str:
    return "n/a" if pd.isna(value) else f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    agg = _aggregate(df, ["quant_type"])

    labels = agg["quant_type"].tolist()
    x = range(len(labels))
    bar_width = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("LLM Quantization Comparison — llm-qlab", fontsize=14, fontweight="bold")

    axes[0].bar(
        x, agg["decode_tps"], width=bar_width, color="#4C9BE8", edgecolor="white",
        yerr=agg["decode_tps_std"].fillna(0), capsize=4,
    )
    axes[0].set_title("Decode Speed (tokens/sec)")
    axes[0].set_xlabel("Quantization Format")
    axes[0].set_ylabel("Tokens / second")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels)
    for i, v in enumerate(agg["decode_tps"]):
        if not pd.isna(v):
            axes[0].text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(x, agg["vram_delta_mb"], width=bar_width, color="#E8844C", edgecolor="white")
    axes[1].set_title("VRAM Attributable to Model (MB)")
    axes[1].set_xlabel("Quantization Format")
    axes[1].set_ylabel("VRAM (MB)")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels)
    for i, v in enumerate(agg["vram_delta_mb"]):
        if not pd.isna(v):
            axes[1].text(i, v + 5, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"Chart saved to {OUTPUT_PNG}")


def plot_comparison_by_family(df: pd.DataFrame) -> None:
    """Grouped bar chart: model families on X-axis, one bar per quant type."""
    RESULTS_DIR.mkdir(exist_ok=True)
    agg = _aggregate(df, ["model_family", "quant_type"])

    families = agg["model_family"].unique().tolist()
    quants = agg["quant_type"].unique().tolist()
    bar_width = 0.8 / len(quants)
    x = range(len(families))

    fig, ax = plt.subplots(figsize=(max(8, len(families) * 2), 5))
    fig.suptitle("Decode t/s by Model Family — llm-qlab", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for i, quant in enumerate(quants):
        subset = agg[agg["quant_type"] == quant]
        values, errors = [], []
        for fam in families:
            match = subset.loc[subset["model_family"] == fam, "decode_tps"]
            err = subset.loc[subset["model_family"] == fam, "decode_tps_std"]
            values.append(0.0 if match.empty or pd.isna(match.iloc[0]) else match.iloc[0])
            errors.append(0.0 if err.empty or pd.isna(err.iloc[0]) else err.iloc[0])

        offset = (i - len(quants) / 2) * bar_width + bar_width / 2
        bars = ax.bar(
            [xi + offset for xi in x], values, width=bar_width, label=quant,
            color=colors[i % len(colors)], edgecolor="white",
            yerr=errors, capsize=3,
        )
        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Model Family")
    ax.set_ylabel("Tokens / second")
    ax.set_title("Decode Speed by Model Family")
    ax.set_xticks(list(x))
    ax.set_xticklabels(families)
    ax.legend(title="Quant type")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG_FAMILY, dpi=150)
    print(f"Chart saved to {OUTPUT_PNG_FAMILY}")


# ---------------------------------------------------------------------------
# Markdown tables
# ---------------------------------------------------------------------------

def print_markdown_table(df: pd.DataFrame) -> None:
    agg = _aggregate(df, ["quant_type"])
    print("\n## Benchmark Results\n")
    print("| Quant | Decode t/s | Prefill t/s | TTFT (ms) | VRAM (MB) | Load (s) | Size (MB) |")
    print("|-------|------------|-------------|-----------|-----------|----------|-----------|")
    for _, row in agg.iterrows():
        print(
            f"| {row['quant_type']} "
            f"| {_fmt(row['decode_tps'])} "
            f"| {_fmt(row['prefill_tps'])} "
            f"| {_fmt(row['ttft_ms'])} "
            f"| {_fmt(row['vram_delta_mb'], 0)} "
            f"| {_fmt(row['load_time_s'])} "
            f"| {_fmt(row['model_size_mb'], 0)} |"
        )
    print()


def print_markdown_table_by_family(df: pd.DataFrame) -> None:
    agg = _aggregate(df, ["model_family", "quant_type"])
    print("\n## Benchmark Results by Model Family\n")
    print("| Model Family | Quant | Decode t/s | Prefill t/s | TTFT (ms) | VRAM (MB) | Size (MB) |")
    print("|--------------|-------|------------|-------------|-----------|-----------|-----------|")
    for _, row in agg.iterrows():
        print(
            f"| {row['model_family']} "
            f"| {row['quant_type']} "
            f"| {_fmt(row['decode_tps'])} "
            f"| {_fmt(row['prefill_tps'])} "
            f"| {_fmt(row['ttft_ms'])} "
            f"| {_fmt(row['vram_delta_mb'], 0)} "
            f"| {_fmt(row['model_size_mb'], 0)} |"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare quantization benchmark results.")
    parser.add_argument(
        "--group-by",
        choices=["quant_type", "model_family"],
        default="quant_type",
        help="Group results by 'quant_type' (default) or 'model_family'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_results()
    df = df[df["timing_source"] != LEGACY_MARKER]
    if df.empty:
        print("[error] No rows with valid timing data. Re-run benchmark.py.")
        sys.exit(1)

    if args.group_by == "model_family":
        plot_comparison_by_family(df)
        print_markdown_table_by_family(df)
    else:
        plot_comparison(df)
        print_markdown_table(df)


if __name__ == "__main__":
    main()
