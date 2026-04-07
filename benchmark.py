"""
benchmark.py — Inference benchmark for GGUF models using llama-cpp-python.

Usage:
    python benchmark.py --model models/model_Q4_K_M.gguf --quant-type Q4_K_M

Results are saved to results/benchmark_results.csv.
"""

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path

from llama_cpp import Llama


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_vram_usage_mb() -> float:
    """Return current GPU VRAM used in MB via nvidia-smi, or -1 on failure."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return float(output.strip().splitlines()[0])
    except Exception:
        return -1.0


def get_model_size_mb(model_path: str) -> float:
    """Return model file size in MB."""
    return os.path.getsize(model_path) / (1024 * 1024)


def print_summary(result: dict) -> None:
    """Print a formatted summary table for a single benchmark run."""
    print("\n" + "=" * 52)
    print(f"  Benchmark Summary — {result['quant_type']}")
    print("=" * 52)
    print(f"  Model            : {result['model_name']}")
    print(f"  Quant type       : {result['quant_type']}")
    print(f"  Prompt tokens    : {result['prompt_tokens']}")
    print(f"  Generated tokens : {result['generated_tokens']}")
    print(f"  Prompt t/s       : {result['prompt_tps']:.2f}")
    print(f"  Generate t/s     : {result['gen_tps']:.2f}")
    print(f"  VRAM used (MB)   : {result['vram_mb']:.1f}")
    print(f"  Load time (s)    : {result['load_time_s']:.2f}")
    print(f"  Model size (MB)  : {result['model_size_mb']:.1f}")
    print("=" * 52 + "\n")


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> dict:
    model_path = args.model
    print(f"Loading model: {model_path}")

    load_start = time.perf_counter()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
    )
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    prompt = args.prompt

    # Warm-up: tokenise prompt to count tokens
    prompt_tokens = llm.tokenize(prompt.encode())
    n_prompt_tokens = len(prompt_tokens)

    vram_before = get_vram_usage_mb()

    # Prompt processing timing
    pp_start = time.perf_counter()
    output = llm(
        prompt,
        max_tokens=args.n_predict,
        echo=False,
    )
    pp_end = time.perf_counter()

    vram_after = get_vram_usage_mb()

    total_time = pp_end - pp_start
    generated_tokens = output["usage"]["completion_tokens"]
    prompt_eval_tokens = output["usage"]["prompt_tokens"]

    gen_tps = generated_tokens / total_time if total_time > 0 else 0.0
    prompt_tps = prompt_eval_tokens / total_time if total_time > 0 else 0.0

    model_name = Path(model_path).stem
    model_size_mb = get_model_size_mb(model_path)

    result = {
        "model_name": model_name,
        "quant_type": args.quant_type,
        "prompt_tokens": prompt_eval_tokens,
        "generated_tokens": generated_tokens,
        "prompt_tps": round(prompt_tps, 2),
        "gen_tps": round(gen_tps, 2),
        "vram_mb": round(max(vram_before, vram_after), 1),
        "load_time_s": round(load_time, 2),
        "model_size_mb": round(model_size_mb, 1),
    }

    return result


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "benchmark_results.csv"
CSV_FIELDS = [
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


def save_result(result: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(result)
    print(f"Result saved to {CSV_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a GGUF model with llama-cpp-python."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the GGUF model file.",
    )
    parser.add_argument(
        "--prompt",
        default="Explain the difference between quantization and pruning in large language models.",
        help="Prompt to use for benchmarking.",
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=128,
        help="Number of tokens to generate (default: 128).",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=99,
        help="Number of model layers to offload to GPU (default: 99 = all).",
    )
    parser.add_argument(
        "--quant-type",
        default="unknown",
        help="Label for the quantization type, e.g. Q4_K_M (used in results).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    print_summary(result)
    save_result(result)


if __name__ == "__main__":
    main()
