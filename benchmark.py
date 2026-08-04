"""
benchmark.py — Inference benchmark for GGUF models using llama-cpp-python.

Usage:
    python benchmark.py --model models/model_Q4_K_M.gguf --quant-type Q4_K_M

Results are saved to results/benchmark_results.csv.

Measurement methodology lives in llm_qlab/bench_core.py — see the module docstring
there for why prefill and decode are timed via llama.cpp's perf counters
rather than wall clock. CSV schema and migration live in llm_qlab/results_schema.py.
"""

import argparse

from llm_qlab.bench_core import DEFAULT_PROMPT, DEFAULT_PROMPT_TOKENS, benchmark_model
from llm_qlab.results_schema import save_result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(value: float, std: float) -> str:
    if value < 0:
        return "not measured"
    return f"{value:.2f}" + (f" ± {std:.2f}" if std else "")


def print_summary(result: dict) -> None:
    print("\n" + "=" * 58)
    print(f"  Benchmark Summary — {result['quant_type']}")
    print("=" * 58)
    print(f"  Model            : {result['model_name']}")
    print(f"  Model family     : {result['model_family']}")
    print(f"  GPU layers       : {result['n_gpu_layers']}")
    print(f"  Runs (median of) : {result['n_runs']}")
    print(f"  Prompt tokens    : {result['prompt_tokens']}")
    print(f"  Generated tokens : {result['generated_tokens']}")
    print(f"  Prefill t/s      : {_fmt(result['prefill_tps'], result['prefill_tps_std'])}")
    print(f"  Decode t/s       : {_fmt(result['decode_tps'], result['decode_tps_std'])}")
    print(f"  TTFT (ms)        : {_fmt(result['ttft_ms'], result['ttft_ms_std'])}")
    print(f"  VRAM model (MB)  : {result['vram_delta_mb']:.1f}")
    print(f"  VRAM board (MB)  : {result['vram_total_mb']:.1f}")
    print(f"  Load time (s)    : {result['load_time_s']:.2f}")
    print(f"  Model size (MB)  : {result['model_size_mb']:.1f}")
    print(f"  Timing source    : {result['timing_source']}")
    print("=" * 58)
    # Two distinct failure modes, and conflating them misreports both: a
    # wall-clock estimate is a timing-path problem, an unstable-clock row was
    # timed correctly but under a power state that makes it unpublishable.
    if result["timing_source"] == "wall_clock_estimate":
        print(
            "  [warn] llama.cpp perf counters unavailable — throughput is a\n"
            "         wall-clock estimate. Prefill in particular reads low."
        )
        print("=" * 58)
    elif result["timing_source"] == "unstable_clocks":
        print(
            "  [warn] GPU clock state was not verified for this row. The\n"
            "         numbers above are excluded from charts and tables."
        )
        print("=" * 58)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a GGUF model with llama-cpp-python."
    )
    parser.add_argument("--model", required=True, help="Path to the GGUF model file.")
    parser.add_argument("--prompt", default=None,
                        help="Literal prompt to benchmark. Overrides --prompt-tokens, "
                             "which means prompt length will vary between tokenizers.")
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS,
                        help=f"Prefill length in tokens, held equal across models "
                             f"(default: {DEFAULT_PROMPT_TOKENS}). Too short a prompt "
                             f"measures per-call overhead instead of throughput.")
    parser.add_argument("--n-predict", type=int, default=128,
                        help="Number of tokens to generate (default: 128).")
    parser.add_argument("--n-gpu-layers", type=int, default=99,
                        help="Number of model layers to offload to GPU (default: 99 = all).")
    parser.add_argument("--quant-type", default="unknown",
                        help="Label for the quantization type, e.g. Q4_K_M (used in results).")
    parser.add_argument("--model-family", default="unknown",
                        help="Model family label, e.g. llama2, mistral, phi3, gemma.")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Measured runs to aggregate; reports median ± stdev (default: 3).")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip the discarded warmup run (not recommended — inflates TTFT).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = benchmark_model(
        args.model,
        n_gpu_layers=args.n_gpu_layers,
        prompt=args.prompt if args.prompt is not None else DEFAULT_PROMPT,
        target_prompt_tokens=0 if args.prompt is not None else args.prompt_tokens,
        n_predict=args.n_predict,
        n_runs=args.n_runs,
        warmup=not args.no_warmup,
        # Needed to size the host-resident embedding table, which decides
        # whether the model fits.
        quant_type=args.quant_type,
    )
    result["model_family"] = args.model_family
    result["quant_type"] = args.quant_type
    print_summary(result)
    save_result(result)


if __name__ == "__main__":
    main()
