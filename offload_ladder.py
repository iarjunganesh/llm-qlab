"""
offload_ladder.py — Sweep --n-gpu-layers and plot VRAM vs generation speed.

For each step in the ladder, loads the model with the given n_gpu_layers,
runs an inference benchmark, records metrics, and finally produces a CSV
and a dual-axis line plot.

Usage:
    python offload_ladder.py --model models/llama-2-7b-chat.Q4_K_M.gguf --quant-type Q4_K_M
    python offload_ladder.py --model models/llama-2-7b-chat.Q4_K_M.gguf --quant-type Q4_K_M --steps 0,16,32,99
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt

from benchmark import get_model_size_mb, get_vram_usage_mb

try:
    from llama_cpp import Llama
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "[error] llama-cpp-python is not installed. "
        "Run: pip install llama-cpp-python"
    ) from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "offload_ladder.csv"
PLOT_PATH = RESULTS_DIR / "offload_ladder.png"

LADDER_FIELDS = [
    "n_gpu_layers",
    "gen_tps",
    "prompt_tps",
    "vram_mb",
    "ttft_ms",
    "load_time_s",
]

DEFAULT_PROMPT = (
    "Explain the difference between quantization and pruning in large language models."
)


# ---------------------------------------------------------------------------
# Single-step benchmark
# ---------------------------------------------------------------------------

def run_ladder_step(
    model_path: str,
    n_gpu_layers: int,
    n_predict: int,
    prompt: str,
) -> dict:
    """Load the model with *n_gpu_layers* and run one inference pass."""
    print(f"\n[ladder] n_gpu_layers={n_gpu_layers} — loading model …")

    load_start = time.perf_counter()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    load_time = time.perf_counter() - load_start
    print(f"[ladder] model loaded in {load_time:.2f}s")

    vram_before = get_vram_usage_mb()

    ttft_ms: float = -1.0
    infer_start = time.perf_counter()
    chunks = []
    for chunk in llm(prompt, max_tokens=n_predict, stream=True, echo=False):
        if ttft_ms < 0:
            ttft_ms = (time.perf_counter() - infer_start) * 1000.0
        chunks.append(chunk)
    infer_wall_time = time.perf_counter() - infer_start

    vram_after = get_vram_usage_mb()

    output = chunks[-1] if chunks else {}
    generated_tokens = output.get("usage", {}).get("completion_tokens", 0)
    prompt_eval_tokens = output.get("usage", {}).get("prompt_tokens", 0)
    # Fallback: count generated tokens by summing chunk text character lengths when
    # usage metadata is absent. Note: character count ≠ token count, so metrics
    # derived from this value will be approximate.
    if generated_tokens == 0 and chunks:
        generated_tokens = sum(
            len(c.get("choices", [{}])[0].get("text", "")) for c in chunks
        )

    timings = output.get("timings", {})
    prompt_ms = timings.get("prompt_ms", 0.0)
    predicted_ms = timings.get("predicted_ms", 0.0)

    if prompt_ms > 0 and predicted_ms > 0:
        prompt_tps = (prompt_eval_tokens / prompt_ms) * 1000.0
        gen_tps = (generated_tokens / predicted_ms) * 1000.0
    else:
        wall_time = timings.get("total_ms", 0.0) / 1000.0
        if wall_time <= 0:
            wall_time = infer_wall_time
        if wall_time <= 0:
            wall_time = 1e-6
        gen_tps = generated_tokens / wall_time
        prompt_tps = prompt_eval_tokens / wall_time

    if vram_before >= 0 and vram_after >= 0:
        vram_value = max(vram_before, vram_after)
    else:
        vram_value = -1.0

    # Explicitly unload to free VRAM before the next step.
    del llm

    return {
        "n_gpu_layers": n_gpu_layers,
        "gen_tps": round(gen_tps, 2),
        "prompt_tps": round(prompt_tps, 2),
        "vram_mb": round(vram_value, 1),
        "ttft_ms": round(ttft_ms, 2),
        "load_time_s": round(load_time, 2),
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def save_ladder_results(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LADDER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[ladder] Results saved to {CSV_PATH}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_ladder_summary(model_name: str, quant_type: str, rows: list[dict]) -> None:
    header = (
        f"{'n_gpu_layers':>12} | {'gen_tps':>8} | {'prompt_tps':>10} "
        f"| {'vram_mb':>8} | {'ttft_ms':>8} | {'load_time_s':>11}"
    )
    sep = "-" * len(header)
    print(f"\n{'=' * len(header)}")
    print(f"  GPU Offload Ladder — {model_name} {quant_type}")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)
    for row in rows:
        print(
            f"{row['n_gpu_layers']:>12} | {row['gen_tps']:>8.2f} | {row['prompt_tps']:>10.2f} "
            f"| {row['vram_mb']:>8.1f} | {row['ttft_ms']:>8.2f} | {row['load_time_s']:>11.2f}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_ladder(model_name: str, quant_type: str, rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    x = [r["n_gpu_layers"] for r in rows]
    gen_tps = [r["gen_tps"] for r in rows]
    vram_mb = [r["vram_mb"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_tps = "#4C9BE8"
    color_vram = "#E8844C"

    ax1.set_xlabel("n_gpu_layers")
    ax1.set_ylabel("Generation speed (tokens/sec)", color=color_tps)
    line1 = ax1.plot(x, gen_tps, marker="o", color=color_tps, label="gen t/s")
    ax1.tick_params(axis="y", labelcolor=color_tps)

    ax2 = ax1.twinx()
    ax2.set_ylabel("VRAM used (MB)", color=color_vram)
    line2 = ax2.plot(x, vram_mb, marker="s", color=color_vram, label="VRAM (MB)")
    ax2.tick_params(axis="y", labelcolor=color_vram)

    lines = line1 + line2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title(f"GPU Offload Ladder — {model_name} {quant_type}")
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
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the GGUF model file.",
    )
    parser.add_argument(
        "--quant-type",
        default="unknown",
        help="Label for the quantization type, e.g. Q4_K_M.",
    )
    parser.add_argument(
        "--steps",
        default="0,8,16,24,32,99",
        help="Comma-separated list of n_gpu_layers values to sweep (default: 0,8,16,24,32,99).",
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=128,
        help="Number of tokens to generate per step (default: 128).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to use for each benchmark step.",
    )
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
        row = run_ladder_step(
            model_path=args.model,
            n_gpu_layers=n_layers,
            n_predict=args.n_predict,
            prompt=args.prompt,
        )
        rows.append(row)

    print_ladder_summary(model_name, args.quant_type, rows)
    save_ladder_results(rows)
    plot_ladder(model_name, args.quant_type, rows)


if __name__ == "__main__":
    main()
