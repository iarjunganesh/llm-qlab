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
    "load_time_s",
    "timing_source",
]


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_ladder_results(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LADDER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[ladder] Results saved to {CSV_PATH}")


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

def plot_ladder(model_name: str, quant_type: str, rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    x = [r["n_gpu_layers"] for r in rows]
    decode_tps = [r["decode_tps"] for r in rows]
    decode_err = [r["decode_tps_std"] for r in rows]
    vram_mb = [r["vram_delta_mb"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_tps = "#4C9BE8"
    color_vram = "#E8844C"

    ax1.set_xlabel("n_gpu_layers")
    ax1.set_ylabel("Decode speed (tokens/sec)", color=color_tps)
    line1 = ax1.errorbar(
        x, decode_tps, yerr=decode_err, marker="o", color=color_tps,
        capsize=3, label="decode t/s",
    )
    ax1.tick_params(axis="y", labelcolor=color_tps)

    ax2 = ax1.twinx()
    ax2.set_ylabel("VRAM attributable to model (MB)", color=color_vram)
    line2 = ax2.plot(x, vram_mb, marker="s", color=color_vram, label="VRAM (MB)")
    ax2.tick_params(axis="y", labelcolor=color_vram)

    handles = [line1, line2[0]]
    ax1.legend(handles, [h.get_label() for h in handles], loc="upper left")

    n_runs = rows[0]["n_runs"] if rows else 0
    plt.title(f"GPU Offload Ladder — {model_name} {quant_type} (median of {n_runs} runs)")
    fig.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
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
    except ValueError:
        raise SystemExit("[error] --steps must be a comma-separated list of integers.")

    model_name = Path(args.model).stem
    rows = []

    for n_layers in steps:
        print(f"\n[ladder] === n_gpu_layers={n_layers} ===")
        rows.append(
            benchmark_model(
                args.model,
                n_gpu_layers=n_layers,
                prompt=args.prompt,
                n_predict=args.n_predict,
                n_runs=args.n_runs,
            )
        )

    print_ladder_summary(model_name, args.quant_type, rows)
    save_ladder_results(rows)
    plot_ladder(model_name, args.quant_type, rows)


if __name__ == "__main__":
    main()
