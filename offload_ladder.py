"""
offload_ladder.py — Sweep --n-gpu-layers and plot VRAM vs generation speed.

For each step in the ladder, loads the model with the given n_gpu_layers,
runs a warmed-up multi-run benchmark, records metrics, and finally produces
a CSV and a dual-axis line plot.

Usage:
    python offload_ladder.py --model models/llama-2-7b-chat.Q4_K_M.gguf --quant-type Q4_K_M
    python offload_ladder.py --model models/llama-2-7b-chat.Q4_K_M.gguf --quant-type Q4_K_M --steps 0,16,32,99

Measurement is delegated to bench_core.benchmark_model, the same code path
benchmark.py uses — the two entry points cannot drift apart.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from bench_core import DEFAULT_PROMPT, benchmark_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "offload_ladder.csv"
PLOT_PATH = RESULTS_DIR / "offload_ladder.png"

LADDER_FIELDS = [
    "model_family",
    "quant_type",
    "n_gpu_layers",
    "n_runs",
    "decode_tps",
    "decode_tps_std",
    "prefill_tps",
    "prefill_tps_std",
    "ttft_ms",
    "ttft_ms_std",
    "vram_delta_mb",
    "vram_total_mb",
    "vram_residency",
    "offload_state",
    "load_time_s",
    "timing_source",
]


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def load_ladder_results() -> list[dict]:
    """Read previously swept families, tolerating the pre-family schema."""
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("model_family", "unknown")
        r.setdefault("quant_type", "unknown")
    return rows


def save_ladder_results(rows: list[dict]) -> list[dict]:
    """Merge this sweep into the CSV, replacing only its own (family, quant).

    Sweeping a second model used to overwrite the first, so the ladder could
    never hold more than one family. Rows are keyed so each family accumulates
    and a re-run of one family refreshes just that family.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    key = {(r["model_family"], r["quant_type"]) for r in rows}
    kept = [r for r in load_ladder_results() if (r["model_family"], r["quant_type"]) not in key]
    merged = kept + rows

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LADDER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    families = sorted({r["model_family"] for r in merged})
    print(f"\n[ladder] Results saved to {CSV_PATH} ({len(merged)} rows, families: {', '.join(families)})")
    return merged


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_ladder_summary(model_name: str, quant_type: str, rows: list[dict]) -> None:
    header = (
        f"{'n_gpu_layers':>12} | {'decode_tps':>16} | {'prefill_tps':>16} "
        f"| {'ttft_ms':>16} | {'vram_model_mb':>13}"
    )
    sep = "-" * len(header)
    print(f"\n{'=' * len(header)}")
    print(f"  GPU Offload Ladder — {model_name} {quant_type}")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)
    for row in rows:
        def cell(key: str) -> str:
            value, std = row[key], row[f"{key}_std"]
            if value < 0:
                return "n/a"
            return f"{value:.2f} ± {std:.2f}" if std else f"{value:.2f}"

        print(
            f"{row['n_gpu_layers']:>12} | {cell('decode_tps'):>16} | {cell('prefill_tps'):>16} "
            f"| {cell('ttft_ms'):>16} | {row['vram_delta_mb']:>13.1f}"
        )
    print(sep)

    if any(r["timing_source"] != "perf_counters" for r in rows):
        print("[warn] One or more steps fell back to wall-clock estimates.")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#1a1a19"
MUTED = "#5c5c5a"
# Same validated categorical slots the comparison charts use, so a family keeps
# one identity across every figure in the repo.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]


def plot_ladder(quant_type: str, rows: list[dict]) -> None:
    """Small multiples over a shared x-axis — one measure per panel.

    Every model family present in *rows* is drawn as its own series, so a
    single figure answers "does offload behave the same across architectures".

    Deliberately NOT a dual-axis chart. Throughput and memory have unrelated
    units, so overlaying them on twin y-scales makes the crossing point an
    artefact of whatever ranges matplotlib happened to choose, and readers
    infer a relationship that isn't in the data. Stacked panels sharing the
    x-axis show the same correlation without inventing one.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    families = sorted({r["model_family"] for r in rows})
    colors = {f: CATEGORICAL[i % len(CATEGORICAL)] for i, f in enumerate(families)}
    # Even spacing: the ladder steps are ordinal, and a linear axis squeezes
    # 0-32 into a third of the width just because the last step is 99.
    steps = sorted({int(r["n_gpu_layers"]) for r in rows})
    pos = {s: i for i, s in enumerate(steps)}

    panels = [
        ("Decode speed", "tokens/sec", "decode_tps", "decode_tps_std", "{:.1f}"),
        ("Time to first token", "milliseconds", "ttft_ms", "ttft_ms_std", "{:.0f}"),
        ("VRAM attributable to model", "MB", "vram_delta_mb", None, "{:.0f}"),
    ]

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(10, 10), sharex=True, facecolor=SURFACE
    )

    for ax, (title, unit, col, err_col, fmt) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        hi, lo = float("-inf"), float("inf")

        for fam in families:
            series = sorted((r for r in rows if r["model_family"] == fam),
                            key=lambda r: int(r["n_gpu_layers"]))
            xs = [pos[int(r["n_gpu_layers"])] for r in series]
            values = [float(r[col]) for r in series]
            errs = [float(r[err_col]) for r in series] if err_col else [0.0] * len(values)

            ax.errorbar(
                xs, values, yerr=errs if err_col else None, marker="o", markersize=7,
                linewidth=2, color=colors[fam], capsize=3, ecolor="#8a8a87",
                elinewidth=1, markeredgecolor=SURFACE, markeredgewidth=1.5, label=fam,
            )
            hi = max(hi, max(v + e for v, e in zip(values, errs)))
            lo = min(lo, min(v - e for v, e in zip(values, errs)))

            # Endpoint labels only, and only for a single series — with several
            # families the values share an x position and overprint each other.
            # Identity then comes from the legend, magnitude from the axis.
            if len(families) == 1:
                for idx in (0, len(values) - 1):
                    ax.annotate(fmt.format(values[idx]), (xs[idx], values[idx] + errs[idx]),
                                textcoords="offset points", xytext=(0, 7), ha="center",
                                fontsize=8, color=INK)

        span = (hi - lo) or 1.0
        ax.set_title(f"{title} ({unit})", color=INK, fontsize=11, loc="left")
        ax.grid(axis="y", color="#e4e4e1", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c9c5")
        ax.tick_params(colors=MUTED, length=0)
        # Bound the view to data *including* error bars — clipping a whisker
        # hides exactly the variance the error bar exists to show.
        ax.set_ylim(lo - span * 0.12, hi + span * 0.18)

    axes[-1].set_xticks(range(len(steps)))
    axes[-1].set_xticklabels([str(s) for s in steps])
    axes[-1].set_xlabel("n_gpu_layers (layers offloaded to GPU)", color=MUTED)

    if len(families) > 1:
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, title="Model family", frameon=False,
                   loc="upper center", bbox_to_anchor=(0.5, 0.945),
                   ncol=len(families), fontsize=9)

    n_runs = rows[0].get("n_runs", 0) if rows else 0
    fig.suptitle(
        f"GPU offload ladder — {quant_type} (median of {n_runs} runs)",
        fontsize=14, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92 if len(families) > 1 else 0.97))
    plt.savefig(PLOT_PATH, dpi=150, facecolor=SURFACE)
    print(f"[ladder] Plot saved to {PLOT_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep --n-gpu-layers and benchmark VRAM vs generation speed."
    )
    parser.add_argument("--model", required=True, help="Path to the GGUF model file.")
    parser.add_argument("--quant-type", default="unknown",
                        help="Label for the quantization type, e.g. Q4_K_M.")
    parser.add_argument("--model-family", default="unknown",
                        help="Model family label, e.g. llama2, mistral, qwen2.5. "
                             "Each family accumulates in the ladder CSV and plot.")
    parser.add_argument("--steps", default="0,8,16,24,32,99",
                        help="Comma-separated n_gpu_layers values to sweep (default: 0,8,16,24,32,99).")
    parser.add_argument("--n-predict", type=int, default=128,
                        help="Number of tokens to generate per step (default: 128).")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Measured runs per step; reports median ± stdev (default: 3).")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="Prompt to use for each benchmark step.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    try:
        steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
    except ValueError as exc:
        raise SystemExit("[error] --steps must be a comma-separated list of integers.") from exc

    model_name = Path(args.model).stem
    rows = []

    for n_layers in steps:
        print(f"\n[ladder] === n_gpu_layers={n_layers} ===")
        row = benchmark_model(
            args.model,
            n_gpu_layers=n_layers,
            prompt=args.prompt,
            n_predict=args.n_predict,
            n_runs=args.n_runs,
        )
        row["model_family"] = args.model_family
        row["quant_type"] = args.quant_type
        rows.append(row)

    print_ladder_summary(model_name, args.quant_type, rows)
    merged = save_ladder_results(rows)
    # Plot every family accumulated so far, not just the one just swept.
    plot_ladder(args.quant_type, merged)


if __name__ == "__main__":
    main()
