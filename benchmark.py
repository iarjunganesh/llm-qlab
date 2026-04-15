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
from typing import Any

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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _token_count(llm: Llama, text: str, *, add_bos: bool = False) -> int:
    if not text:
        return 0
    try:
        return len(llm.tokenize(text.encode("utf-8"), add_bos=add_bos))
    except Exception:
        return 0


def print_summary(result: dict) -> None:
    """Print a formatted summary table for a single benchmark run."""
    print("\n" + "=" * 52)
    print(f"  Benchmark Summary — {result['quant_type']}")
    print("=" * 52)
    print(f"  Model            : {result['model_name']}")
    print(f"  Model family     : {result['model_family']}")
    print(f"  Quant type       : {result['quant_type']}")
    print(f"  Prompt tokens    : {result['prompt_tokens']}")
    print(f"  Generated tokens : {result['generated_tokens']}")
    print(f"  Prompt t/s       : {result['prompt_tps']:.2f}")
    print(f"  Generate t/s     : {result['gen_tps']:.2f}")
    print(f"  VRAM used (MB)   : {result['vram_mb']:.1f}")
    print(f"  Load time (s)    : {result['load_time_s']:.2f}")
    print(f"  TTFT (ms)        : {result['ttft_ms']:.2f}")
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

    vram_before = get_vram_usage_mb()

    # Run inference in streaming mode to capture TTFT (time-to-first-token).
    ttft_ms: float = -1.0
    infer_start = time.perf_counter()
    chunks = []
    for chunk in llm(prompt, max_tokens=args.n_predict, stream=True, echo=False):
        if ttft_ms < 0:
            ttft_ms = (time.perf_counter() - infer_start) * 1000.0
        chunks.append(chunk)
    infer_wall_time = time.perf_counter() - infer_start

    vram_after = get_vram_usage_mb()

    generated_text = "".join(
        c.get("choices", [{}])[0].get("text", "") for c in chunks
    )

    # The final chunk carries usage counts in llama-cpp-python streaming responses.
    output = chunks[-1] if chunks else {}
    generated_tokens = output.get("usage", {}).get("completion_tokens", 0)
    prompt_eval_tokens = output.get("usage", {}).get("prompt_tokens", 0)

    # Fallback to tokenizer-based counts when usage metadata is absent.
    if prompt_eval_tokens == 0:
        prompt_eval_tokens = _token_count(llm, prompt, add_bos=True)
    if generated_tokens == 0 and chunks:
        generated_tokens = _token_count(llm, generated_text, add_bos=False)

    # Use per-phase timings when available (llama-cpp-python >= 0.2.x)
    timings = output.get("timings", {})
    prompt_ms = timings.get("prompt_ms", 0.0)
    predicted_ms = timings.get("predicted_ms", 0.0)

    if prompt_ms > 0 and predicted_ms > 0:
        # Accurate per-phase rates from llama.cpp internal timers
        prompt_tps = (prompt_eval_tokens / prompt_ms) * 1000.0
        gen_tps = (generated_tokens / predicted_ms) * 1000.0
    else:
        # Fallback: use measured wall-clock inference time.
        wall_time = timings.get("total_ms", 0.0) / 1000.0
        if wall_time <= 0:
            wall_time = infer_wall_time
        if wall_time <= 0:
            wall_time = 1e-6  # prevent division by zero
        gen_tps = generated_tokens / wall_time
        prompt_tps = prompt_eval_tokens / wall_time

    if vram_before >= 0 and vram_after >= 0:
        vram_value = max(vram_before, vram_after)
    else:
        vram_value = -1.0

    model_name = Path(model_path).stem
    model_size_mb = get_model_size_mb(model_path)

    result = {
        "model_name": model_name,
        "model_family": args.model_family,
        "quant_type": args.quant_type,
        "prompt_tokens": prompt_eval_tokens,
        "generated_tokens": generated_tokens,
        "prompt_tps": round(prompt_tps, 2),
        "gen_tps": round(gen_tps, 2),
        "vram_mb": round(vram_value, 1),
        "load_time_s": round(load_time, 2),
        "ttft_ms": round(ttft_ms, 2),
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


def _normalize_result_row(values: list[str]) -> dict | None:
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
        "prompt_tps": round(_safe_float(row_map.get("prompt_tps", 0.0), 0.0), 2),
        "gen_tps": round(_safe_float(row_map.get("gen_tps", 0.0), 0.0), 2),
        "vram_mb": round(_safe_float(row_map.get("vram_mb", -1.0), -1.0), 1),
        "load_time_s": round(_safe_float(row_map.get("load_time_s", 0.0), 0.0), 2),
        "ttft_ms": round(_safe_float(row_map.get("ttft_ms", -1.0), -1.0), 2),
        "model_size_mb": round(_safe_float(row_map.get("model_size_mb", 0.0), 0.0), 1),
    }


def ensure_csv_schema() -> None:
    """Rotate and normalize benchmark CSV when schema mismatch is detected."""
    if not CSV_PATH.exists():
        return

    with open(CSV_PATH, "r", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return

    needs_migration = rows[0] != CSV_FIELDS
    normalized_rows = []
    skipped_rows = 0

    for raw in rows[1:]:
        normalized = _normalize_result_row(raw)
        if normalized is None:
            skipped_rows += 1
            continue
        normalized_rows.append(normalized)

    if not needs_migration and skipped_rows == 0:
        return

    backup_path = CSV_PATH.with_name(
        f"{CSV_PATH.stem}.legacy.{int(time.time())}{CSV_PATH.suffix}"
    )
    CSV_PATH.replace(backup_path)

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(normalized_rows)

    print(f"[info] Migrated benchmark CSV to latest schema: {CSV_PATH}")
    print(f"[info] Legacy backup saved as: {backup_path}")
    if skipped_rows:
        print(f"[warn] Skipped {skipped_rows} malformed row(s) during migration.")


def save_result(result: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    ensure_csv_schema()
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
    parser.add_argument(
        "--model-family",
        default="unknown",
        help="Model family label, e.g. llama2, mistral, phi3, gemma (used in results).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    print_summary(result)
    save_result(result)


if __name__ == "__main__":
    main()
