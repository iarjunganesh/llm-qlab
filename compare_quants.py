"""
compare_quants.py — Compare quantization benchmark results.

Reads results/benchmark_results.csv, produces comparison bar charts
(tokens/sec and VRAM usage), saves them to results/comparison.png,
and prints a markdown-formatted comparison table to stdout.

Usage:
    python compare_quants.py
    python compare_quants.py --group-by model_family
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "benchmark_results.csv"
OUTPUT_PNG = RESULTS_DIR / "comparison.png"
OUTPUT_PNG_FAMILY = RESULTS_DIR / "comparison_by_family.png"

CSV_FIELDS = [
    "model_name",
    "model_family",
    "quant_type",
    "prompt_tokens",
    "generated_tokens",
    "prompt_tps",
    "gen_tps",
    "vram_mb",
    "load_time_s",
    "ttft_ms",
    "model_size_mb",
]

LEGACY_FIELDS_V1 = [
    "model_name",
    "quant_type",
    "prompt_tokens",
    "generated_tokens",
    "prompt_tps",
    "gen_tps",
    "vram_mb",
    "load_time_s",
    "model_size_mb",
]

LEGACY_FIELDS_V2 = [
    "model_name",
    "model_family",
    "quant_type",
    "prompt_tokens",
    "generated_tokens",
    "prompt_tps",
    "gen_tps",
    "vram_mb",
    "load_time_s",
    "model_size_mb",
]


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_row(values: list[str]) -> dict | None:
    if len(values) == len(CSV_FIELDS):
        row_map = dict(zip(CSV_FIELDS, values))
    elif len(values) == len(LEGACY_FIELDS_V2):
        row_map = dict(zip(LEGACY_FIELDS_V2, values))
        row_map["ttft_ms"] = "-1"
    elif len(values) == len(LEGACY_FIELDS_V1):
        row_map = dict(zip(LEGACY_FIELDS_V1, values))
        row_map["model_family"] = "unknown"
        row_map["ttft_ms"] = "-1"
    else:
        return None

    return {
        "model_name": row_map.get("model_name", "unknown"),
        "model_family": row_map.get("model_family", "unknown") or "unknown",
        "quant_type": row_map.get("quant_type", "unknown") or "unknown",
        "prompt_tokens": _safe_int(row_map.get("prompt_tokens", 0), 0),
        "generated_tokens": _safe_int(row_map.get("generated_tokens", 0), 0),
        "prompt_tps": _safe_float(row_map.get("prompt_tps", 0.0), 0.0),
        "gen_tps": _safe_float(row_map.get("gen_tps", 0.0), 0.0),
        "vram_mb": _safe_float(row_map.get("vram_mb", -1.0), -1.0),
        "load_time_s": _safe_float(row_map.get("load_time_s", 0.0), 0.0),
        "ttft_ms": _safe_float(row_map.get("ttft_ms", -1.0), -1.0),
        "model_size_mb": _safe_float(row_map.get("model_size_mb", 0.0), 0.0),
    }


def _load_results_resilient() -> pd.DataFrame:
    rows = []
    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    if len(all_rows) <= 1:
        return pd.DataFrame(columns=CSV_FIELDS)

    for raw in all_rows[1:]:
        normalized = _normalize_row(raw)
        if normalized is not None:
            rows.append(normalized)

    return pd.DataFrame(rows, columns=CSV_FIELDS)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_results() -> pd.DataFrame:
    if not CSV_PATH.exists():
        print(f"[error] Results file not found: {CSV_PATH}")
        print("Run benchmark.py first to generate results.")
        sys.exit(1)

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        df = _load_results_resilient()
    if df.empty:
        print("[error] Results file is empty. Run benchmark.py first.")
        sys.exit(1)

    # Backward-compat: fill columns added in newer schema versions.
    if "model_family" not in df.columns:
        df["model_family"] = "unknown"
    if "ttft_ms" not in df.columns:
        df["ttft_ms"] = -1.0

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


def plot_comparison_by_family(df: pd.DataFrame) -> None:
    """Grouped bar chart: model families on X-axis, one bar per quant type."""
    RESULTS_DIR.mkdir(exist_ok=True)

    agg = (
        df.groupby(["model_family", "quant_type"], sort=False)
        .agg(gen_tps=("gen_tps", "mean"))
        .reset_index()
    )

    families = agg["model_family"].unique().tolist()
    quants = agg["quant_type"].unique().tolist()
    n_families = len(families)
    n_quants = len(quants)

    bar_width = 0.8 / n_quants
    x = range(n_families)

    fig, ax = plt.subplots(figsize=(max(8, n_families * 2), 5))
    fig.suptitle("Gen t/s by Model Family — llm-qlab", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for i, quant in enumerate(quants):
        subset = agg[agg["quant_type"] == quant]
        # Align values to the families list (missing → 0)
        values = [
            subset.loc[subset["model_family"] == fam, "gen_tps"].iloc[0]
            if fam in subset["model_family"].values else 0.0
            for fam in families
        ]
        offset = (i - n_quants / 2) * bar_width + bar_width / 2
        bars = ax.bar(
            [xi + offset for xi in x],
            values,
            width=bar_width,
            label=quant,
            color=colors[i % len(colors)],
            edgecolor="white",
        )
        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.3,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xlabel("Model Family")
    ax.set_ylabel("Tokens / second")
    ax.set_title("Generation Speed by Model Family")
    ax.set_xticks(list(x))
    ax.set_xticklabels(families)
    ax.legend(title="Quant type")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG_FAMILY, dpi=150)
    print(f"Chart saved to {OUTPUT_PNG_FAMILY}")


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def print_markdown_table(df: pd.DataFrame) -> None:
    has_ttft = (df["ttft_ms"] != -1.0).any()

    agg = (
        df.groupby("quant_type", sort=False)
        .agg(
            gen_tps=("gen_tps", "mean"),
            prompt_tps=("prompt_tps", "mean"),
            vram_mb=("vram_mb", "mean"),
            load_time_s=("load_time_s", "mean"),
            ttft_ms=("ttft_ms", "mean"),
            model_size_mb=("model_size_mb", "mean"),
        )
        .reset_index()
    )

    if has_ttft:
        header = "| Quant | Gen t/s | Prompt t/s | VRAM (MB) | Load (s) | TTFT (ms) | Size (MB) |"
        separator = "|-------|---------|------------|-----------|----------|-----------|-----------|"
    else:
        header = "| Quant | Gen t/s | Prompt t/s | VRAM (MB) | Load (s) | Size (MB) |"
        separator = "|-------|---------|------------|-----------|----------|-----------|"

    print("\n## Benchmark Results\n")
    print(header)
    print(separator)
    for _, row in agg.iterrows():
        ttft_col = f"| {row['ttft_ms']:.2f} " if has_ttft else ""
        print(
            f"| {row['quant_type']} "
            f"| {row['gen_tps']:.2f} "
            f"| {row['prompt_tps']:.2f} "
            f"| {row['vram_mb']:.0f} "
            f"| {row['load_time_s']:.2f} "
            f"{ttft_col}"
            f"| {row['model_size_mb']:.0f} |"
        )
    print()


def print_markdown_table_by_family(df: pd.DataFrame) -> None:
    has_ttft = (df["ttft_ms"] != -1.0).any()

    agg = (
        df.groupby(["model_family", "quant_type"], sort=False)
        .agg(
            gen_tps=("gen_tps", "mean"),
            prompt_tps=("prompt_tps", "mean"),
            vram_mb=("vram_mb", "mean"),
            ttft_ms=("ttft_ms", "mean"),
            model_size_mb=("model_size_mb", "mean"),
        )
        .reset_index()
    )

    if has_ttft:
        header = "| Model Family | Quant | Gen t/s | Prompt t/s | VRAM (MB) | TTFT (ms) | Size (MB) |"
        separator = "|--------------|-------|---------|------------|-----------|-----------|-----------|"
    else:
        header = "| Model Family | Quant | Gen t/s | Prompt t/s | VRAM (MB) | Size (MB) |"
        separator = "|--------------|-------|---------|------------|-----------|-----------|"

    print("\n## Benchmark Results by Model Family\n")
    print(header)
    print(separator)
    for _, row in agg.iterrows():
        ttft_col = f"| {row['ttft_ms']:.2f} " if has_ttft else ""
        print(
            f"| {row['model_family']} "
            f"| {row['quant_type']} "
            f"| {row['gen_tps']:.2f} "
            f"| {row['prompt_tps']:.2f} "
            f"| {row['vram_mb']:.0f} "
            f"{ttft_col}"
            f"| {row['model_size_mb']:.0f} |"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare quantization benchmark results."
    )
    parser.add_argument(
        "--group-by",
        choices=["quant_type", "model_family"],
        default="quant_type",
        help="Group results by 'quant_type' (default) or 'model_family'.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    df = load_results()

    if args.group_by == "model_family":
        plot_comparison_by_family(df)
        print_markdown_table_by_family(df)
    else:
        plot_comparison(df)
        print_markdown_table(df)


if __name__ == "__main__":
    main()
