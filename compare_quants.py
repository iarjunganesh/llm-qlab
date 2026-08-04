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

from results_schema import (
    CSV_FIELDS, CSV_PATH, LEGACY_MARKER, UNSTABLE_MARKER, normalize_row,
)

# Neither marker denotes a publishable measurement, for different reasons:
# legacy rows were timed by a broken path, unstable rows were timed correctly
# but without evidence the GPU held a consistent clock state.
EXCLUDED_SOURCES = (LEGACY_MARKER, UNSTABLE_MARKER)

RESULTS_DIR = Path("results")
OUTPUT_PNG = RESULTS_DIR / "comparison.png"
OUTPUT_PNG_FAMILY = RESULTS_DIR / "comparison_by_family.png"

# Categorical slots 1-3, validated for colourblind separation against the light
# chart surface (worst adjacent pair ΔE 9.2 deutan / 27.6 normal). The aqua slot
# sits below 3:1 contrast, so every bar carries a direct value label.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
SURFACE = "#fcfcfb"
INK = "#1a1a19"
MUTED = "#5c5c5a"

# Quantization is ordinal — keep charts in bit-depth order, not discovery order.
QUANT_ORDER = ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M",
               "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"]

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


CONFIG_KEY = ["model_name", "quant_type", "n_gpu_layers"]


def _keep_latest_per_config(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated measurements of one configuration to the newest.

    benchmark.py appends rather than replaces, so re-running a configuration —
    which is exactly what the harness tells you to do when a row comes back
    flagged — leaves the old row in the file. Both then reach the charts, and
    which one a lookup returns depends on row order rather than recency. That
    would let a superseded number be published after it had been re-measured.

    Rows are in append order, so the last occurrence is the newest.
    """
    if not set(CONFIG_KEY).issubset(df.columns):
        return df
    superseded = len(df) - len(df.drop_duplicates(subset=CONFIG_KEY, keep="last"))
    if superseded:
        print(f"[info] {superseded} superseded row(s) ignored — a newer "
              "measurement exists for the same configuration.")
    return df.drop_duplicates(subset=CONFIG_KEY, keep="last").reset_index(drop=True)


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

    df = _keep_latest_per_config(df)

    stale = df[df["timing_source"] == LEGACY_MARKER]
    if not stale.empty:
        print(
            f"[warn] {len(stale)} row(s) predate the prefill-timing fix and have no "
            "valid throughput. They are excluded from charts and tables — re-run "
            "benchmark.py for those configurations."
        )

    unstable = df[df["timing_source"] == UNSTABLE_MARKER]
    if not unstable.empty:
        print(
            f"[warn] {len(unstable)} row(s) were measured without a verified GPU "
            "clock state and are excluded from charts and tables. Re-run those "
            "configurations with the machine otherwise idle and the power profile "
            "set to its highest setting."
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


def _quant_order(agg: pd.DataFrame) -> list[str]:
    quants = [q for q in QUANT_ORDER if q in set(agg["quant_type"])]
    return quants + [q for q in agg["quant_type"].unique() if q not in quants]


def _split_plottable(agg: pd.DataFrame, value_col: str = "decode_tps") -> tuple[list[str], list[str]]:
    """Separate quant formats that have data from those that have none.

    A configuration that was refused or flagged carries no throughput, and
    plotting it as a zero-height bar renders an empty slot that reads as a
    measurement of zero rather than an absence of one. Such formats are named
    in a caption instead — the fact that Q8_0 does not fit is a finding, but it
    is not a bar.
    """
    plottable, empty = [], []
    for quant in _quant_order(agg):
        values = agg.loc[agg["quant_type"] == quant, value_col]
        (plottable if (values > 0).any() else empty).append(quant)
    return plottable, empty


def _caption_for_empty(fig, empty: list[str]) -> None:
    """Note omitted quant formats beneath the axes."""
    if not empty:
        return
    fig.text(
        0.5, 0.015,
        f"{', '.join(empty)} omitted — exceeds available VRAM on this hardware; "
        "see README, Known issues.",
        ha="center", fontsize=8, color=MUTED, style="italic",
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _family_colors(families: list[str]) -> dict[str, str]:
    """Map each family to a fixed categorical slot.

    Keyed on the sorted family name rather than row order, so filtering the
    data cannot repaint the surviving series — colour follows the entity.
    """
    return {fam: CATEGORICAL[i % len(CATEGORICAL)] for i, fam in enumerate(sorted(families))}


def _grouped_bars(ax, agg, families, quants, colors, value_col, err_col, fmt):
    """Draw one grouped-bar panel: quant on x, one coloured bar per family."""
    # 2px surface gap between adjacent bars at 150 dpi.
    slot = 0.8 / len(families)
    bar_width = slot - (2 / 150) / max(len(quants), 1)

    for i, fam in enumerate(families):
        values, errors = [], []
        for q in quants:
            sel = agg[(agg["model_family"] == fam) & (agg["quant_type"] == q)]
            v = float("nan") if sel.empty else sel[value_col].iloc[0]
            e = 0.0 if sel.empty or err_col is None or pd.isna(sel[err_col].iloc[0]) else sel[err_col].iloc[0]
            values.append(0.0 if pd.isna(v) else v)
            errors.append(e)

        offset = (i - len(families) / 2) * slot + slot / 2
        positions = [xi + offset for xi in range(len(quants))]
        ax.bar(
            positions, values, width=bar_width, label=fam,
            color=colors[fam], edgecolor=SURFACE, linewidth=0.5,
            yerr=errors if any(errors) else None, capsize=3,
            error_kw={"elinewidth": 1, "ecolor": "#5c5c5a"},
        )
        # Direct labels satisfy the relief requirement for the low-contrast slot.
        # Sit them above the error-bar cap, not the bar top, or they collide.
        span = max([v for v in values if v > 0], default=1.0)
        for px, v, e in zip(positions, values, errors):
            if v > 0:
                ax.text(px, v + e + span * 0.02, fmt.format(v),
                        ha="center", va="bottom", fontsize=7, color=INK)

    ax.set_xticks(range(len(quants)))
    ax.set_xticklabels(quants)
    ax.grid(axis="y", color="#e4e4e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c9c5")
    ax.tick_params(colors=MUTED, length=0)
    ax.margins(y=0.18)


def plot_comparison(df: pd.DataFrame) -> None:
    """Decode throughput and VRAM, every family shown side by side per quant."""
    RESULTS_DIR.mkdir(exist_ok=True)
    agg = _aggregate(df, ["model_family", "quant_type"])

    families = sorted(agg["model_family"].unique().tolist())
    quants, empty_quants = _split_plottable(agg)
    if not quants:
        print("[error] No quantization format has plottable data.")
        sys.exit(1)
    colors = _family_colors(families)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)
    fig.suptitle("LLM Quantization Comparison — llm-qlab", fontsize=14,
                 fontweight="bold", color=INK)

    for ax in axes:
        ax.set_facecolor(SURFACE)

    _grouped_bars(axes[0], agg, families, quants, colors, "decode_tps", "decode_tps_std", "{:.1f}")
    axes[0].set_title("Decode speed (tokens/sec)", color=INK, fontsize=11)
    axes[0].set_xlabel("Quantization format", color=MUTED)
    axes[0].set_ylabel("Tokens / second", color=MUTED)

    _grouped_bars(axes[1], agg, families, quants, colors, "vram_delta_mb", None, "{:.0f}")
    axes[1].set_title("VRAM attributable to model (MB)", color=INK, fontsize=11)
    axes[1].set_xlabel("Quantization format", color=MUTED)
    axes[1].set_ylabel("VRAM (MB)", color=MUTED)

    # Legend is mandatory for >= 2 series; identity is never colour-alone.
    # Figure-level and above the panels, so it never covers a bar.
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, title="Model family", frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=len(families), fontsize=9)

    _caption_for_empty(fig, empty_quants)
    plt.tight_layout(rect=(0, 0.04 if empty_quants else 0, 1, 0.88))
    plt.savefig(OUTPUT_PNG, dpi=150, facecolor=SURFACE)
    print(f"Chart saved to {OUTPUT_PNG}")


def plot_comparison_by_family(df: pd.DataFrame) -> None:
    """Grouped bar chart: model families on X-axis, one bar per quant type."""
    RESULTS_DIR.mkdir(exist_ok=True)
    agg = _aggregate(df, ["model_family", "quant_type"])

    families = sorted(agg["model_family"].unique().tolist())
    quants, empty_quants = _split_plottable(agg)
    if not quants:
        print("[error] No quantization format has plottable data.")
        sys.exit(1)
    bar_width = 0.8 / len(quants)
    x = range(len(families))

    fig, ax = plt.subplots(figsize=(max(8, len(families) * 2), 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    fig.suptitle("Decode t/s by Model Family — llm-qlab", fontsize=14,
                 fontweight="bold", color=INK)

    # Quant is ordinal here, so step one hue light->dark rather than cycling hues.
    colors = ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281"][-len(quants):]
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

    _caption_for_empty(fig, empty_quants)
    plt.tight_layout(rect=(0, 0.04 if empty_quants else 0, 1, 1))
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
    df = df[~df["timing_source"].isin(EXCLUDED_SOURCES)]
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
