"""
bench_core.py — Shared measurement logic for llm-qlab benchmarks.

Both benchmark.py and offload_ladder.py import from here so that timing
methodology stays identical between the two entry points.

Measurement notes
-----------------
Throughput is read from llama.cpp's own performance counters
(``llama_perf_context``), which report prefill and decode phases separately:

    t_p_eval_ms / n_p_eval  -> prompt (prefill) processing
    t_eval_ms   / n_eval    -> token generation (decode)

This matters. Earlier versions of this repo read a ``timings`` dict off the
final streaming chunk, but llama-cpp-python does not populate ``usage`` or
``timings`` on streamed responses — so the code always fell through to a
wall-clock fallback that divided *both* phases by the same total elapsed
time. That made the reported "prompt t/s" a restatement of generation speed
rather than a measurement of prefill.

If the perf-counter API is unavailable, we fall back to a wall-clock
decomposition around TTFT and flag the row via ``timing_source``, rather than
silently reporting a number that looks like a measurement but isn't.
"""

from __future__ import annotations

import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import llama_cpp
from llama_cpp import Llama

# Sentinel for "not measured". Kept out of statistics so a failed probe can
# never masquerade as a real reading.
UNKNOWN = -1.0

DEFAULT_PROMPT = (
    "Explain the difference between quantization and pruning in large language models."
)


# ---------------------------------------------------------------------------
# GPU memory
# ---------------------------------------------------------------------------

def get_vram_usage_mb() -> float:
    """Return VRAM currently in use on GPU 0, in MB, or UNKNOWN on failure.

    This is whole-board usage as reported by nvidia-smi, so it includes the
    desktop compositor and any other process on the GPU. Use
    :func:`vram_delta` to attribute usage to the model under test.
    """
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
        return UNKNOWN


def vram_delta(baseline_mb: float, peak_mb: float) -> float:
    """VRAM attributable to the model: peak minus pre-load baseline."""
    if baseline_mb < 0 or peak_mb < 0:
        return UNKNOWN
    return max(0.0, peak_mb - baseline_mb)


# ---------------------------------------------------------------------------
# llama.cpp performance counters
# ---------------------------------------------------------------------------

def _ctx_ptr(llm: Llama):
    """Best-effort access to the underlying llama_context pointer."""
    ctx = getattr(llm, "_ctx", None)
    return getattr(ctx, "ctx", None) if ctx is not None else None


def reset_perf_counters(llm: Llama) -> bool:
    """Zero llama.cpp's perf counters. Returns False if unavailable."""
    ptr = _ctx_ptr(llm)
    if ptr is None:
        return False
    try:
        llama_cpp.llama_perf_context_reset(ptr)
        return True
    except Exception:
        return False


def read_perf_counters(llm: Llama) -> dict[str, float] | None:
    """Read prefill/decode timings from llama.cpp, or None if unavailable."""
    ptr = _ctx_ptr(llm)
    if ptr is None:
        return None
    try:
        data = llama_cpp.llama_perf_context(ptr)
    except Exception:
        return None

    n_p_eval = int(getattr(data, "n_p_eval", 0))
    n_eval = int(getattr(data, "n_eval", 0))
    t_p_eval_ms = float(getattr(data, "t_p_eval_ms", 0.0))
    t_eval_ms = float(getattr(data, "t_eval_ms", 0.0))

    # A zero token count means the phase never ran (e.g. the prompt was served
    # from cache). Report UNKNOWN rather than dividing by zero.
    prefill_tps = (n_p_eval / t_p_eval_ms * 1000.0) if n_p_eval and t_p_eval_ms > 0 else UNKNOWN
    decode_tps = (n_eval / t_eval_ms * 1000.0) if n_eval and t_eval_ms > 0 else UNKNOWN

    return {
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "prompt_tokens": float(n_p_eval),
        "generated_tokens": float(n_eval),
    }


# ---------------------------------------------------------------------------
# State hygiene between runs
# ---------------------------------------------------------------------------

def clear_state(llm: Llama) -> None:
    """Drop KV cache and token state so the next run re-executes prefill.

    Without this, llama-cpp-python's prompt-prefix reuse skips prefill on
    repeat runs of the same prompt and the prefill measurement collapses.
    """
    try:
        llm.reset()
    except Exception:
        pass
    ctx = getattr(llm, "_ctx", None)
    if ctx is not None:
        try:
            ctx.kv_cache_clear()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def measure_run(llm: Llama, prompt: str, n_predict: int) -> dict[str, float]:
    """Run one inference pass and return its metrics.

    Streaming is used so time-to-first-token reflects what a caller actually
    waits for. Throughput still comes from the perf counters, since wall-clock
    timing around a Python generator includes detokenization and loop overhead.
    """
    clear_state(llm)
    have_counters = reset_perf_counters(llm)

    ttft_ms: float = UNKNOWN
    chunk_count = 0
    start = time.perf_counter()
    for _ in llm(prompt, max_tokens=n_predict, stream=True, echo=False):
        if ttft_ms < 0:
            ttft_ms = (time.perf_counter() - start) * 1000.0
        chunk_count += 1
    total_s = time.perf_counter() - start

    metrics: dict[str, float] = {
        "ttft_ms": ttft_ms,
        "total_s": total_s,
    }

    counters = read_perf_counters(llm) if have_counters else None
    if counters and counters["decode_tps"] > 0:
        metrics.update(counters)
        metrics["timing_source"] = 1.0  # llama.cpp perf counters
        return metrics

    # Fallback: decompose wall clock around TTFT. The first token is produced
    # during the TTFT window, so it is excluded from the decode rate.
    prompt_tokens = float(count_tokens(llm, prompt, add_bos=True))
    generated_tokens = float(chunk_count)
    decode_s = total_s - (ttft_ms / 1000.0) if ttft_ms > 0 else total_s
    metrics.update(
        {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            # Prefill is approximate here: TTFT covers prefill *and* the first
            # decode step, so this reads slightly low.
            "prefill_tps": (prompt_tokens / (ttft_ms / 1000.0)) if ttft_ms > 0 else UNKNOWN,
            "decode_tps": ((generated_tokens - 1) / decode_s) if decode_s > 0 and generated_tokens > 1 else UNKNOWN,
            "timing_source": 0.0,  # wall-clock estimate
        }
    )
    return metrics


def count_tokens(llm: Llama, text: str, *, add_bos: bool = False) -> int:
    if not text:
        return 0
    try:
        return len(llm.tokenize(text.encode("utf-8"), add_bos=add_bos))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _median(values: Sequence[float]) -> float:
    usable = [v for v in values if v > 0]
    return statistics.median(usable) if usable else UNKNOWN


def _stdev(values: Sequence[float]) -> float:
    usable = [v for v in values if v > 0]
    return statistics.stdev(usable) if len(usable) > 1 else 0.0


def benchmark_model(
    model_path: str,
    *,
    n_gpu_layers: int,
    prompt: str = DEFAULT_PROMPT,
    n_predict: int = 128,
    n_runs: int = 3,
    warmup: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Load a model, warm it up, run *n_runs* measured passes, and aggregate.

    Returns medians plus sample standard deviations. Median is used rather
    than mean so a single scheduling hiccup does not move the headline number.
    """
    baseline_vram = get_vram_usage_mb()

    if verbose:
        print(f"Loading model: {model_path} (n_gpu_layers={n_gpu_layers})")
    load_start = time.perf_counter()
    llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, verbose=False)
    load_time_s = time.perf_counter() - load_start
    if verbose:
        print(f"Model loaded in {load_time_s:.2f}s")

    # The first inference absorbs CUDA context creation and kernel autotuning.
    # Measuring it would inflate TTFT several-fold, which is exactly the
    # discrepancy that showed up between the old comparison and ladder tables.
    if warmup:
        if verbose:
            print("Warmup run (discarded) …")
        measure_run(llm, prompt, min(8, n_predict))

    runs: list[dict[str, float]] = []
    for i in range(n_runs):
        if verbose:
            print(f"Run {i + 1}/{n_runs} …")
        runs.append(measure_run(llm, prompt, n_predict))

    peak_vram = get_vram_usage_mb()

    prefill = [r.get("prefill_tps", UNKNOWN) for r in runs]
    decode = [r.get("decode_tps", UNKNOWN) for r in runs]
    ttft = [r.get("ttft_ms", UNKNOWN) for r in runs]

    # Wall-clock fallback anywhere in the set taints the whole aggregate.
    source = "perf_counters" if all(r.get("timing_source") == 1.0 for r in runs) else "wall_clock_estimate"

    result: dict[str, Any] = {
        "model_name": Path(model_path).stem,
        "n_gpu_layers": n_gpu_layers,
        "n_runs": n_runs,
        "prompt_tokens": int(_median([r.get("prompt_tokens", 0.0) for r in runs])),
        "generated_tokens": int(_median([r.get("generated_tokens", 0.0) for r in runs])),
        "prefill_tps": round(_median(prefill), 2),
        "prefill_tps_std": round(_stdev(prefill), 2),
        "decode_tps": round(_median(decode), 2),
        "decode_tps_std": round(_stdev(decode), 2),
        "ttft_ms": round(_median(ttft), 2),
        "ttft_ms_std": round(_stdev(ttft), 2),
        "vram_delta_mb": round(vram_delta(baseline_vram, peak_vram), 1),
        "vram_total_mb": round(peak_vram, 1),
        "load_time_s": round(load_time_s, 2),
        "model_size_mb": round(Path(model_path).stat().st_size / (1024 * 1024), 1),
        "timing_source": source,
    }

    del llm
    return result
