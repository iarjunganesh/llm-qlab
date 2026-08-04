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

import gc
import re
import threading
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

# Default prefill length. A 16-token prompt makes t_p_eval_ms a measurement of
# per-call fixed cost (kernel launch, graph setup) plus whatever remains of the
# GPU's clock ramp, not of prefill throughput -- which is why prefill and TTFT
# carried far wider spreads than decode in every earlier sweep. 256 tokens is
# 16x the work while still fitting the 512-token default context alongside 128
# generated tokens, so it costs no KV cache growth and leaves VRAM comparable.
DEFAULT_PROMPT_TOKENS = 256

# Warmup stops once WARMUP_STREAK consecutive discarded runs agree within this
# fraction. Two was not enough: on llama-2 Q5_K_M two consecutive warmup runs
# agreed to 0.1% (58.05 -> 57.99 t/s) and the following 5-run median came in at
# 47.09 — a plateau mid-ramp is indistinguishable from steady state when only
# two samples are compared. Every one of 18 runs in the 2026-08-04 sweep
# reported convergence and four then measured 13-27% away.
# The budget has to cover the memory clock's climb as well as the throughput's.
# Measured on this card the memory domain sits at P4 (9001 MHz) for the first
# several runs of a cold configuration before reaching P2, and warmup runs at
# P4 cannot start a streak — so a budget of 8 was consumed entirely by the ramp.
WARMUP_TOLERANCE = 0.05
WARMUP_STREAK = 3
MAX_WARMUP_RUNS = 16

# Decode is memory-bandwidth-bound, so measurement is gated on the memory clock
# rather than on the SM clock alone. Observed memory P-states on this card under
# load are P0 12101 / P2 11101 / P4 9001 MHz.
#
# The floor sits between P4 and P2 (9001/12101 = 0.744, 11101/12101 = 0.917)
# because P2->P4 is the transition that matters: its ratio is 1.233, and the
# decode bimodality being chased was 1.228-1.236 across three quantization
# formats. Demanding P0 outright was tried first and rejected every run in 12
# attempts — the driver migrates P0<->P2 continuously under WDDM and no run of
# useful length stays at P0 throughout. P0 vs P2 is a 1.09x spread, which the
# coherence check below bounds rather than eliminates.
MIN_MEM_CLOCK_FRACTION = 0.90

# Accepted runs must have executed at comparable clocks for their spread to
# describe the model rather than the power state. Runs are compared on mean
# memory clock; more than this relative spread means the set is not coherent
# and the aggregate is flagged rather than published.
CLOCK_COHERENCE_TOLERANCE = 0.02

# Escape hatch for the clock-coherence proxy, and deliberately tight. If the
# accepted runs' throughput agrees to within this, clock excursions among them
# did not affect the measurement and the aggregate stands. Set well below the
# 1.23x artifact this machinery exists to exclude, so it cannot readmit it.
THROUGHPUT_COHERENCE_TOLERANCE = 0.01

# nvidia-smi clocks_event_reasons bits.
THROTTLE_GPU_IDLE = 0x1
THROTTLE_APP_CLOCKS = 0x2
THROTTLE_SW_POWER_CAP = 0x4
THROTTLE_HW_SLOWDOWN = 0x8
THROTTLE_SYNC_BOOST = 0x10
THROTTLE_SW_THERMAL = 0x20
THROTTLE_HW_THERMAL = 0x40
THROTTLE_HW_POWER_BRAKE = 0x80

# Reasons that disqualify a run. SwPowerCap is deliberately absent: a boosting
# GPU sits at its power cap essentially all the time — that is how the boost
# algorithm works, not a fault — and treating it as disqualifying rejected
# every run in 12 attempts while the card was drawing 54 W of a 108 W limit at
# 62 C. Where power capping actually costs throughput it shows up as a memory
# clock dip, which the floor above already catches. These bits, by contrast,
# mean the card was forcibly slowed.
THROTTLE_DISQUALIFYING = (
    THROTTLE_HW_SLOWDOWN | THROTTLE_SW_THERMAL
    | THROTTLE_HW_THERMAL | THROTTLE_HW_POWER_BRAKE
)

# Attempts allowed per requested clean run. The card migrates P-states on its
# own, so some proportion of runs will always be discarded; this bounds how
# long a configuration may spend trying rather than failing silently.
CLOCK_ATTEMPT_BUDGET = 4

# Anything above this before the model loads means another process is on the GPU.
IDLE_UTILIZATION_PCT = 10.0

# Source text for token-exact prompts. Tokenizers disagree on how many tokens a
# given string becomes, so a fixed string would hand each model a different
# amount of prefill work and make the cross-model comparison invalid. This is
# truncated to an exact token count per model instead. Only needs to be longer
# than the target under the most efficient tokenizer in the set.
PROMPT_CORPUS = """
Quantization reduces the numeric precision used to store a model's weights.
A tensor held in sixteen-bit floating point can be mapped onto four- or
eight-bit integers together with a small number of scaling factors, so the
memory needed to hold the model shrinks by roughly the ratio of the two bit
widths. The mapping is lossy. Each weight is replaced by the nearest value
the smaller format can represent, and the residual difference is discarded.
Block-wise schemes limit the damage by giving every small group of weights
its own scale, so a single unusually large value distorts only its immediate
neighbours rather than an entire tensor.

The practical benefit is not only that the file on disk becomes smaller. A
single token of autoregressive decoding requires reading every weight the
model contains, so decoding speed on modern accelerators is bounded by memory
bandwidth rather than by arithmetic capability. Halving the number of bytes
that must cross the memory bus very nearly halves the time each token takes.
This is why a heavily quantized model can produce text faster than a more
precise copy of the same network running on identical hardware, even though
both perform the same number of multiplications.

Prefill behaves differently. Processing the tokens of an incoming prompt can
be batched, so many rows of the weight matrices are reused across positions
once they have been fetched. That shifts the bottleneck away from bandwidth
and towards raw arithmetic throughput, and it explains why measured prefill
rates are typically far higher than decode rates on the same model, and why
the two respond differently to a change in precision.

Memory capacity introduces a separate effect that is easy to mistake for a
property of the quantization format itself. When the weights very nearly fill
the available device memory, the runtime may place part of the model in host
memory and stream it across the system bus on demand. Throughput then falls
sharply, because the effective bandwidth of that path is an order of magnitude
below that of attached device memory. The transition is abrupt rather than
gradual, and a measurement taken near the boundary can depend on how much
memory other processes happened to be holding at the time.

Pruning attacks size from another direction. Instead of representing every
weight less precisely, it removes weights entirely, setting them to zero and
omitting them from storage. Unstructured pruning may zero individual entries
scattered anywhere in a tensor, which preserves accuracy well but yields an
irregular sparsity pattern that most hardware cannot exploit without special
support. Structured pruning removes whole rows, columns, attention heads, or
layers, producing a smaller dense network that runs faster on ordinary kernels
at the cost of a larger drop in quality for the same reduction in parameters.

The two techniques compose. A network can be pruned and then quantized, and
the savings multiply rather than merely add. What they do not share is a
recovery story: quantization is usually applied after training with no further
optimization, whereas aggressive pruning generally requires fine-tuning before
the network regains its previous accuracy.
""".strip()


# ---------------------------------------------------------------------------
# GPU memory
# ---------------------------------------------------------------------------

def get_gpu_utilization() -> float:
    """Return GPU busy percentage across all processes, or UNKNOWN on failure."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
        return float(output.strip().splitlines()[0])
    except Exception:
        return UNKNOWN


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


def wait_for_vram_release(target_mb: float, *, tolerance_mb: float = 400.0,
                          timeout_s: float = 30.0) -> float:
    """Block until VRAM returns near *target_mb*, or timeout.

    Freeing a Llama object does not immediately return device memory, so the
    next model can be handed a fragmented heap and spill to host memory even
    though it would fit on a clean device. That made results order-dependent:
    the same model measured fast or slow depending on what preceded it.
    """
    deadline = time.time() + timeout_s
    current = get_vram_usage_mb()
    while time.time() < deadline:
        current = get_vram_usage_mb()
        if current < 0 or current <= target_mb + tolerance_mb:
            return current
        time.sleep(0.5)
    return current


def _sample_peak(current_peak: float) -> float:
    """Fold a fresh VRAM reading into a running peak."""
    reading = get_vram_usage_mb()
    if reading < 0:
        return current_peak
    return reading if current_peak < 0 else max(current_peak, reading)


def get_vram_free_mb() -> float:
    """Free VRAM on GPU 0 in MB, or UNKNOWN if telemetry is unavailable."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True,
        )
        return float(output.strip().splitlines()[0])
    except Exception:
        return UNKNOWN


def check_vram_headroom(model_size_mb: float, *, kv_overhead: float = 1.15) -> tuple[bool, str]:
    """Decide whether the model can actually be held in free VRAM.

    Weights are not the whole cost — the KV cache, compute buffers and CUDA
    context add roughly 10-20% on top, so a model that "just fits" on paper
    does not fit in practice.

    This exists because a model larger than free VRAM still *runs*: the driver
    silently pages it, and the harness reports a throughput number that is
    really measuring PCIe transfer and whatever else happened to be on the GPU
    at that moment. Those numbers looked like measurements and drifted 20-60%
    between identical runs. Refusing to present them is the fix — a benchmark
    that cannot be run under controlled conditions should say so.
    """
    free = get_vram_free_mb()
    if free < 0:
        return True, "vram telemetry unavailable — proceeding unchecked"
    required = model_size_mb * kv_overhead
    if required > free:
        return False, (f"needs ~{required:.0f} MB (weights {model_size_mb:.0f} MB "
                       f"+ {int((kv_overhead - 1) * 100)}% runtime) but only {free:.0f} MB free")
    return True, f"{free:.0f} MB free for ~{required:.0f} MB required"


def read_clock_state() -> dict[str, Any]:
    """Snapshot the GPU's clock domain in one nvidia-smi call.

    Returns pstate ("P0".."P12" or "unknown"), current and maximum SM/memory
    clocks in MHz, and whichever throttle reasons are active.

    Memory clock is queried alongside SM clock because decode is
    memory-bandwidth-bound: it is the memory domain that sets decode
    throughput, and the two domains change P-state independently.
    """
    fields = ("pstate,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,"
              "clocks_event_reasons.active")
    blank = {"pstate": "unknown", "sm_mhz": UNKNOWN, "mem_mhz": UNKNOWN,
             "sm_max_mhz": UNKNOWN, "mem_max_mhz": UNKNOWN, "throttle": "unknown"}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
        parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
    except Exception:
        return blank
    if len(parts) < 6:
        return blank
    try:
        note_mem_clock(float(parts[2]))
    except ValueError:
        pass

    def _num(value: str) -> float:
        try:
            return float(value)
        except ValueError:
            return UNKNOWN

    # The active-reasons bitmask is hex. GpuIdle is expected between runs and
    # is not a throttle; the mask is kept intact so callers can distinguish
    # benign boost management from a forced slowdown.
    try:
        mask = int(parts[5], 16) & ~THROTTLE_GPU_IDLE
    except ValueError:
        mask = -1
    throttle = "unknown" if mask < 0 else ("none" if mask == 0 else hex(mask))

    return {
        "pstate": parts[0] or "unknown",
        "sm_mhz": _num(parts[1]),
        "mem_mhz": _num(parts[2]),
        "sm_max_mhz": _num(parts[3]),
        "mem_max_mhz": _num(parts[4]),
        "throttle": throttle,
    }


class ClockSampler:
    """Poll the clock domain in a background thread for the duration of a run.

    Sampling before and after a run is not sufficient and was actively
    misleading: the GPU drops to its idle P-state within a fraction of a second
    of the work stopping, so a reading taken between runs reports P5/810 MHz no
    matter how the run itself executed. What matters is the clock *while the
    kernels were resident*, which only concurrent sampling can observe.
    """

    def __init__(self, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ClockSampler":
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _poll(self) -> None:
        while not self._stop.is_set():
            state = read_clock_state()
            if state.get("mem_mhz", UNKNOWN) >= 0:
                self.samples.append(state)
            self._stop.wait(self.interval_s)

    def under_load(self) -> list[dict[str, Any]]:
        """Samples taken while the GPU was actually busy.

        Load ramps and tails off around the measured window, so leading and
        trailing idle samples are excluded rather than dragging the summary
        toward the idle clock.
        """
        return [s for s in self.samples if s.get("pstate") not in ("P5", "P8", "P12")]

    def min_mem_mhz(self) -> float:
        """Lowest memory clock observed under load — the worst moment governs.

        A run that dipped a P-state partway through is not a measurement at the
        higher clock, so the minimum decides admissibility, not the mean.
        """
        loaded = self.under_load()
        return min((s["mem_mhz"] for s in loaded), default=UNKNOWN)

    def max_mem_mhz(self) -> float:
        return max((s["mem_mhz"] for s in self.samples), default=UNKNOWN)

    def mean_mem_mhz(self) -> float:
        """Time-weighted mean memory clock under load.

        Sampling is at a fixed interval, so a plain mean is already time
        weighted. This is the figure that actually predicts decode throughput
        when the card splits a run between two P-states.
        """
        loaded = [s["mem_mhz"] for s in self.under_load()]
        return sum(loaded) / len(loaded) if loaded else UNKNOWN


def _disqualifying_throttles(samples: list[dict[str, Any]]) -> set[str]:
    """Throttle reasons among *samples* that invalidate a measurement.

    Boost-management reasons (SwPowerCap in particular) are excluded — see
    THROTTLE_DISQUALIFYING. Only forced slowdowns are returned.
    """
    blocking = set()
    for sample in samples:
        raw = sample.get("throttle", "none")
        if raw in ("none", "unknown"):
            continue
        try:
            mask = int(raw, 16)
        except ValueError:
            continue
        if mask & THROTTLE_DISQUALIFYING:
            blocking.add(hex(mask & THROTTLE_DISQUALIFYING))
    return blocking


def _relative_spread(values: list[float]) -> float:
    """Peak-to-peak spread as a fraction of the maximum."""
    usable = [v for v in values if v > 0]
    if len(usable) < 2:
        return 0.0
    return (max(usable) - min(usable)) / max(usable)


def _clocks_are_coherent(mean_clocks: list[float],
                         decode_tps: list[float] | None = None) -> tuple[bool, float]:
    """Were the accepted runs comparable enough to aggregate?

    Returns (coherent, clock_spread).

    Clock spread is a *proxy* for the thing that actually matters: whether the
    reported stdev describes the model or the power state. When the direct
    evidence is available and tight, the proxy must not override it. Measured
    on mistral Q5_K_M, five runs spanned 11101-11501 MHz mean (3.5%) and
    produced decode figures spanning 49.18-49.44 t/s (0.5%, stdev 0.11) — the
    clock excursions were brief and upward, and demonstrably did not move
    throughput at this magnitude.

    So a set is coherent if the clocks agree, *or* if the throughput agrees
    tightly enough that clock variation cannot be hiding in it. Every run has
    already cleared MIN_MEM_CLOCK_FRACTION individually, so this cannot admit a
    set that ran at a low clock throughout.
    """
    clock_spread = _relative_spread(mean_clocks)
    if clock_spread <= CLOCK_COHERENCE_TOLERANCE:
        return True, clock_spread
    if decode_tps and _relative_spread(decode_tps) <= THROUGHPUT_COHERENCE_TOLERANCE:
        return True, clock_spread
    return False, clock_spread


def _modal_pstate(samples: list[dict[str, Any]]) -> str:
    """The most common P-state across a run's clock samples."""
    states = [s.get("pstate", "unknown") for s in samples]
    return max(set(states), key=states.count) if states else "unknown"


def _summarize_throttles(samples: list[dict[str, Any]]) -> str:
    """Throttle reasons seen across a run's clock samples, or 'none'."""
    seen = {s.get("throttle", "unknown") for s in samples} - {"none", "unknown"}
    return "|".join(sorted(seen)) if seen else "none"


# Highest memory clock this process has actually seen. nvidia-smi's
# clocks.max.mem is a spec figure and under-reports the achieved P0 clock on
# this hardware (12001 reported, 12101 observed), so a threshold taken against
# it is measured from the wrong zero. The observed peak is the honest reference.
_observed_max_mem_mhz = UNKNOWN


def note_mem_clock(mhz: float) -> None:
    """Record a memory clock reading into the running observed maximum."""
    global _observed_max_mem_mhz
    if mhz > 0 and mhz > _observed_max_mem_mhz:
        _observed_max_mem_mhz = mhz


def reference_max_mem_mhz(state: dict[str, Any] | None = None) -> float:
    """Best available estimate of the card's top memory clock."""
    reported = (state or {}).get("mem_max_mhz", UNKNOWN)
    candidates = [c for c in (_observed_max_mem_mhz, reported) if c > 0]
    return max(candidates) if candidates else UNKNOWN


def mem_clock_fraction(state: dict[str, Any]) -> float:
    """Memory clock as a fraction of its maximum, or UNKNOWN without telemetry."""
    current = state.get("mem_mhz", UNKNOWN)
    maximum = reference_max_mem_mhz(state)
    if current < 0 or maximum <= 0:
        return UNKNOWN
    return current / maximum


def clocks_are_boosted(state: dict[str, Any], *, min_fraction: float = MIN_MEM_CLOCK_FRACTION) -> bool:
    """True if the card is in its top performance state.

    Two independent conditions, because either alone has a blind spot. P0 is
    the driver's own label but a card can sit in P0 with the memory domain
    still ramping; the clock fraction catches that. The clock fraction alone
    cannot be evaluated when telemetry is missing, where pstate still can be.
    """
    fraction = mem_clock_fraction(state)
    if state.get("pstate") == "unknown" and fraction < 0:
        return False  # no telemetry — never claim boosted
    if fraction >= 0 and fraction < min_fraction:
        return False
    return state.get("pstate") in ("P0", "unknown")


def wait_for_stable_clocks(*, samples: int = 3, tolerance_pct: float = 8.0,
                           timeout_s: float = 25.0) -> bool:
    """Block until the GPU reaches its top P-state and stops moving.

    An idle GPU sits at a low clock and ramps under load. The first model of a
    sweep was being measured mid-ramp and read ~20% low with wide variance.

    An earlier revision watched ``clocks.sm`` alone and returned True while the
    *memory* clock was still a P-state low. Decode is memory-bandwidth-bound,
    so that check could report success on a card running decode ~20% slow —
    which produced a bimodal throughput distribution with a constant ~1.23x
    ratio across every quantization format, the signature of a clock artifact
    rather than a property of any model. On this hardware the memory domain
    spans 810 MHz idle to 12001 MHz boost, so the blind spot was most of the
    dynamic range.

    Returns True only if the memory clock is both boosted and settled.
    """
    deadline = time.time() + timeout_s
    history: list[float] = []
    while time.time() < deadline:
        state = read_clock_state()
        mem = state.get("mem_mhz", UNKNOWN)
        if mem < 0:
            return False
        history.append(mem)
        if len(history) >= samples and clocks_are_boosted(state):
            window = history[-samples:]
            if max(window) > 0 and (max(window) - min(window)) / max(window) * 100 <= tolerance_pct:
                return True
        time.sleep(0.7)
    return False


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
# Tensor placement, straight from llama.cpp
# ---------------------------------------------------------------------------

# Where the weights actually landed is reported by llama.cpp at load time and
# nowhere else. An nvidia-smi delta cannot substitute for it: that delta is
# whole-board, includes the CUDA context and KV cache, and is measured against
# a GGUF file size that counts metadata the loader never uploads. Dividing one
# by the other produced a "residency" figure that could exceed 1.0 for a model
# that had spilled, which is why it failed to predict throughput.
_LOG_LINES: list[str] = []

_BUFFER_RE = re.compile(
    r"^\s*load_tensors:\s*(?P<buf>\S+)\s+model buffer size\s*=\s*(?P<mb>[\d.]+)\s*MiB",
    re.IGNORECASE,
)

_OFFLOAD_RE = re.compile(
    r"^\s*load_tensors:\s*offloaded\s+(?P<done>\d+)\s*/\s*(?P<total>\d+)\s+layers to GPU",
    re.IGNORECASE,
)


@llama_cpp.llama_log_callback
def _log_collector(level: int, text: bytes, user_data) -> None:  # noqa: ARG001
    try:
        _LOG_LINES.append(text.decode("utf-8", errors="replace"))
    except Exception:
        pass


@llama_cpp.llama_log_callback
def _log_discard(level: int, text: bytes, user_data) -> None:  # noqa: ARG001
    """Swallow llama.cpp logs without touching Python state."""


def install_log_capture() -> None:
    """Route llama.cpp's C-level log through us so load lines can be parsed."""
    llama_cpp.llama_log_set(_log_collector, None)


def uninstall_log_capture() -> None:
    """Stop collecting.

    Capture must not outlive model load. Left installed, every llama.cpp log
    message crosses into Python during generation: the first sweep run this
    way lost 2.5x of prefill throughput and doubled TTFT variance. An
    instrument that changes the measurement is worse than no instrument.
    """
    llama_cpp.llama_log_set(_log_discard, None)


def parse_tensor_placement(lines: Sequence[str]) -> dict[str, float] | None:
    """Split load-time buffer sizes into device and host bytes.

    llama.cpp emits one line per backend buffer, plus a layer tally::

        load_tensors: offloaded 33/33 layers to GPU
        load_tensors:        CUDA0 model buffer size =  3820.94 MiB
        load_tensors:   CPU_Mapped model buffer size =    70.31 MiB

    Note the two do not agree on "fully offloaded": at 33/33 the token
    embedding table still sits on the host, so byte residency tops out near
    0.98. The layer tally is therefore the authority on whether weights
    spilled; the byte split says how much.

    Returns None if no such lines were seen, so callers can fall back rather
    than invent a number.
    """
    gpu_mb = 0.0
    host_mb = 0.0
    seen = False
    layers_done = UNKNOWN
    layers_total = UNKNOWN
    for line in "".join(lines).splitlines():
        tally = _OFFLOAD_RE.match(line)
        if tally:
            layers_done = float(tally.group("done"))
            layers_total = float(tally.group("total"))
            continue
        match = _BUFFER_RE.match(line)
        if not match:
            continue
        seen = True
        size = float(match.group("mb"))
        if match.group("buf").upper().startswith("CPU"):
            host_mb += size
        else:
            gpu_mb += size
    if not seen:
        return None
    total = gpu_mb + host_mb
    return {
        "gpu_mb": gpu_mb,
        "host_mb": host_mb,
        "residency": (gpu_mb / total) if total > 0 else UNKNOWN,
        "layers_offloaded": layers_done,
        "layers_total": layers_total,
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

    # Suppress EOS so every run decodes exactly n_predict tokens. Without this
    # the model ends its reply when it likes -- one run produced 20 tokens
    # instead of 128 -- which both shrinks the decode sample and makes the
    # generated length depend on the prompt. llama-bench fixes n_gen the same
    # way. This is a throughput measurement, not a generation-quality one.
    eos_bias = {llm.token_eos(): -100.0}

    ttft_ms: float = UNKNOWN
    chunk_count = 0
    start = time.perf_counter()
    for _ in llm(prompt, max_tokens=n_predict, stream=True, echo=False,
                 logit_bias=eos_bias):
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


def build_prompt(llm: Llama, target_tokens: int, corpus: str = PROMPT_CORPUS) -> str:
    """Return a prompt that this model's tokenizer turns into *target_tokens*.

    Truncation happens in token space rather than by character count so that
    every model in a sweep prefills exactly the same number of tokens. The
    round trip through detokenize is not always exact, so the caller should
    treat the returned length as approximate and rely on the measured
    ``n_p_eval`` counter for the number that gets reported.
    """
    tokens = llm.tokenize(corpus.encode("utf-8"), add_bos=False)
    if len(tokens) < target_tokens:
        raise ValueError(
            f"prompt corpus yields only {len(tokens)} tokens for this tokenizer, "
            f"need at least {target_tokens}"
        )
    return llm.detokenize(tokens[:target_tokens]).decode("utf-8", errors="ignore")


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
    target_prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
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

    # Sampled before this process touches the GPU, so anything above idle is
    # someone else. On Windows the device is time-sliced with every compositor,
    # browser and chat client on the desktop; a busy neighbour has been seen to
    # move decode by 30% between otherwise identical invocations. The harness
    # cannot prevent that, so it at least refuses to report it as a clean number.
    baseline_util = get_gpu_utilization()
    if verbose and baseline_util > IDLE_UTILIZATION_PCT:
        print(f"[warn] GPU already {baseline_util:.0f}% busy before load — another "
              "process is competing; results will read low and vary")

    # Refuse configurations that cannot be held in VRAM. Without this the model
    # still loads, the driver pages it, and the harness emits a plausible number
    # that is really measuring PCIe traffic — the source of the 20-60% swings on
    # Q8_0 models sitting at 93-98% of an 8 GB card.
    model_size_mb = round(Path(model_path).stat().st_size / (1024 * 1024), 1)
    fits, headroom_note = check_vram_headroom(model_size_mb) if n_gpu_layers != 0 else (True, "cpu only")
    if not fits:
        if verbose:
            print(f"[skip] {Path(model_path).name}: {headroom_note}")
            print("       Close GPU-using applications or use a larger card; "
                  "a paged run is not a measurement.")
        return {
            "model_name": Path(model_path).stem,
            "n_gpu_layers": n_gpu_layers,
            "n_runs": 0,
            "prompt_tokens": 0, "generated_tokens": 0,
            "prefill_tps": UNKNOWN, "prefill_tps_std": 0.0,
            "decode_tps": UNKNOWN, "decode_tps_std": 0.0,
            "ttft_ms": UNKNOWN, "ttft_ms_std": 0.0,
            "vram_delta_mb": UNKNOWN, "vram_total_mb": UNKNOWN,
            "vram_residency": UNKNOWN, "offload_state": "insufficient_vram",
            "load_time_s": 0.0, "model_size_mb": model_size_mb,
            "timing_source": "skipped_insufficient_vram",
        }
    if verbose:
        print(f"VRAM check: {headroom_note}")
        print(f"Loading model: {model_path} (n_gpu_layers={n_gpu_layers})")
    install_log_capture()
    _LOG_LINES.clear()
    load_start = time.perf_counter()
    llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, verbose=False)
    load_time_s = time.perf_counter() - load_start
    placement = parse_tensor_placement(_LOG_LINES)
    uninstall_log_capture()
    _LOG_LINES.clear()
    if verbose:
        print(f"Model loaded in {load_time_s:.2f}s")
        if placement:
            print(f"Tensor placement: {placement['gpu_mb']:.0f} MB device / "
                  f"{placement['host_mb']:.0f} MB host")
        else:
            print("[warn] Could not read tensor placement from llama.cpp log")

    # Built after load because it depends on this model's tokenizer.
    if target_prompt_tokens > 0:
        prompt = build_prompt(llm, target_prompt_tokens)
        if verbose:
            print(f"Prompt: {count_tokens(llm, prompt, add_bos=True)} tokens "
                  f"(target {target_prompt_tokens} + BOS)")

    # Two separate warmups are needed, and conflating them was a bug.
    #   1. CUDA context creation and kernel autotuning — a short run covers it.
    #   2. The GPU climbing out of its idle clock state — a short run does NOT,
    #      because it ends before the clock ramp finishes. The first model of a
    #      sweep was therefore measured mid-ramp and read ~20% low with wide
    #      variance, while later models read correctly.
    # So: warm up at full generation length, then hold until clocks settle.
    #
    # One warmup is still not enough. Measured on llama-2 Q4_K_M, the first
    # *measured* run came in at 274 t/s prefill / 18.4 t/s decode while runs
    # 2-8 held 434-506 and 21.0-22.3 — after wait_for_stable_clocks had already
    # reported success. nvidia-smi's clock reading says nothing about whether
    # the caches and driver-side allocator have reached steady state, so warm
    # up until the throughput itself stops moving.
    # A streak of WARMUP_STREAK agreeing runs is required, not a single pair,
    # and every run in the streak must have been taken at a boosted memory
    # clock. Either condition alone was demonstrably insufficient.
    warmup_converged = True
    if warmup:
        if verbose:
            print("Warmup runs (discarded) …")
        warmup_converged = False
        previous = UNKNOWN
        streak = 0
        for attempt in range(MAX_WARMUP_RUNS):
            with ClockSampler() as sampler:
                observed = measure_run(llm, prompt, n_predict).get("decode_tps", UNKNOWN)
            floor = sampler.min_mem_mhz()
            ceiling = reference_max_mem_mhz()
            # Judged on the clock held *during* the run. Reading it afterwards
            # samples the idle state, which the card reaches almost immediately.
            boosted = floor > 0 and ceiling > 0 and floor >= ceiling * MIN_MEM_CLOCK_FRACTION
            agreed = (previous > 0 and observed > 0
                      and abs(observed - previous) / previous < WARMUP_TOLERANCE)
            streak = streak + 1 if (agreed and boosted) else 0
            if verbose:
                flag = "" if boosted else f"  [clock dipped to {floor:.0f} MHz]"
                print(f"  warmup {attempt + 1}: {observed:.2f} t/s  "
                      f"streak {streak}/{WARMUP_STREAK}{flag}")
            previous = observed
            if streak >= WARMUP_STREAK:
                warmup_converged = True
                if verbose:
                    print(f"Warmed up after {attempt + 1} runs "
                          f"({WARMUP_STREAK} consecutive within "
                          f"{WARMUP_TOLERANCE:.0%} at full memory clock)")
                break
        if not warmup_converged and verbose:
            print(f"[warn] decode rate still moving after {MAX_WARMUP_RUNS} warmup "
                  "runs — this row will be flagged unstable_clocks, not published")

    # Sampled after every run, not once at the end: a model that spills during
    # load and settles afterwards is invisible to a single trailing snapshot.
    peak_vram = _sample_peak(UNKNOWN)

    # Each measured run is bracketed by a clock reading. A run during which the
    # card left its top P-state, dropped memory clock, or hit a throttle reason
    # is not a measurement of the model — it is a measurement of the power
    # state — so it is discarded rather than averaged in.
    # The card cannot be pinned to P0 under WDDM — it migrates between memory
    # P-states on its own while the work is running. So rather than measure a
    # fixed number of runs and reject the dirty ones (which yields a variable
    # and often empty sample), keep running until n_runs clean samples exist or
    # the attempt budget is spent. Accepted runs are all at the same clock,
    # which is what makes their stdev meaningful.
    runs: list[dict[str, float]] = []
    rejected: list[str] = []
    clock_samples: list[dict[str, Any]] = []
    run_mean_clocks: list[float] = []
    max_attempts = max(n_runs * CLOCK_ATTEMPT_BUDGET, n_runs)
    attempt = 0
    while len(runs) < n_runs and attempt < max_attempts:
        attempt += 1
        if verbose:
            print(f"Run {len(runs) + 1}/{n_runs} (attempt {attempt}) …")
        with ClockSampler() as sampler:
            run = measure_run(llm, prompt, n_predict)
        peak_vram = _sample_peak(peak_vram)

        loaded = sampler.under_load()
        clock_samples.extend(loaded)
        floor = sampler.min_mem_mhz()
        ceiling = reference_max_mem_mhz()
        blocking = _disqualifying_throttles(loaded)

        if not loaded or floor < 0 or ceiling <= 0:
            reason = f"attempt {attempt}: no clock telemetry during run"
        elif floor < ceiling * MIN_MEM_CLOCK_FRACTION:
            reason = (f"attempt {attempt}: memory clock dipped to {floor:.0f} MHz "
                      f"({floor / ceiling:.0%} of {ceiling:.0f})")
        elif blocking:
            reason = f"attempt {attempt}: throttled ({', '.join(sorted(blocking))})"
        else:
            runs.append(run)
            run_mean_clocks.append(sampler.mean_mem_mhz())
            if verbose:
                print(f"  [accept] {run.get('decode_tps', UNKNOWN):.2f} t/s "
                      f"at {floor:.0f}-{sampler.max_mem_mhz():.0f} MHz "
                      f"(mean {sampler.mean_mem_mhz():.0f})")
            continue

        rejected.append(reason)
        if verbose:
            print(f"  [reject] {reason}")

    if verbose and len(runs) < n_runs:
        print(f"[warn] only {len(runs)} of {n_runs} clean runs collected in "
              f"{attempt} attempts")

    # Every run rejected means there is nothing to aggregate. Report the
    # configuration as unmeasured rather than inventing a number from zero
    # samples — the same rule the harness applies to unavailable counters.
    if not runs:
        if verbose:
            print(f"[warn] all {n_runs} runs rejected on clock state — "
                  "no measurement produced")
        del llm
        gc.collect()
        wait_for_vram_release(baseline_vram)
        return {
            "model_name": Path(model_path).stem,
            "n_gpu_layers": n_gpu_layers, "n_runs": 0,
            "prompt_tokens": 0, "generated_tokens": 0,
            "prefill_tps": UNKNOWN, "prefill_tps_std": 0.0,
            "decode_tps": UNKNOWN, "decode_tps_std": 0.0,
            "ttft_ms": UNKNOWN, "ttft_ms_std": 0.0,
            "vram_delta_mb": round(vram_delta(baseline_vram, peak_vram), 1),
            "vram_total_mb": round(peak_vram, 1),
            "vram_residency": round(placement["residency"], 3) if placement else UNKNOWN,
            "offload_state": "unknown",
            "pstate": _modal_pstate(clock_samples),
            "mem_clock_mhz": round(_median([s["mem_mhz"] for s in clock_samples]), 0),
            "throttle_reasons": _summarize_throttles(clock_samples),
            "load_time_s": round(load_time_s, 2),
            "model_size_mb": round(Path(model_path).stat().st_size / (1024 * 1024), 1),
            "timing_source": "unstable_clocks",
        }

    prefill = [r.get("prefill_tps", UNKNOWN) for r in runs]
    decode = [r.get("decode_tps", UNKNOWN) for r in runs]
    ttft = [r.get("ttft_ms", UNKNOWN) for r in runs]

    # Wall-clock fallback anywhere in the set taints the whole aggregate.
    source = "perf_counters" if all(r.get("timing_source") == 1.0 for r in runs) else "wall_clock_estimate"

    # A warmup that never converged, or a set thinned by clock rejections, does
    # not produce a publishable number. Earlier revisions only printed a warning
    # here, so an unstable row landed in the CSV indistinguishable from a good
    # one — the same defect class as the original streaming-timings bug: a value
    # that is not a measurement presented as though it were.
    # Rejections on their own are expected and fine — they are the mechanism
    # working. What disqualifies an aggregate is a warmup that never settled,
    # too few surviving runs, or surviving runs that ran at different clocks.
    coherent, clock_spread = _clocks_are_coherent(run_mean_clocks, decode)
    if not warmup_converged or len(runs) < n_runs or not coherent:
        source = "unstable_clocks"
        if verbose:
            reasons = []
            if not warmup_converged:
                reasons.append("warmup never converged")
            if len(runs) < n_runs:
                reasons.append(f"only {len(runs)}/{n_runs} clean runs")
            if not coherent:
                reasons.append(f"clock spread {clock_spread:.1%} across accepted runs")
            print(f"[warn] timing_source=unstable_clocks ({'; '.join(reasons)})")

    # Residency is the fraction of model weights llama.cpp placed on the device.
    # Reported by the loader itself, so it is a placement fact rather than an
    # inference drawn from whole-board memory readings.
    model_size_mb = round(Path(model_path).stat().st_size / (1024 * 1024), 1)
    delta = vram_delta(baseline_vram, peak_vram)
    residency = round(placement["residency"], 3) if placement else UNKNOWN

    # The layer tally, not the byte fraction, decides whether weights spilled:
    # a fully offloaded model still keeps its embedding table on the host.
    if placement is None:
        offload_state = "unknown"
    elif n_gpu_layers == 0:
        offload_state = "cpu_only"
    elif placement["layers_total"] < 0:
        offload_state = "unknown"
    elif placement["layers_offloaded"] >= placement["layers_total"]:
        offload_state = "resident"
    else:
        offload_state = "partial"

    if verbose and placement and offload_state == "partial":
        print(f"[warn] {placement['layers_offloaded']:.0f}/{placement['layers_total']:.0f} "
              f"layers on device, {placement['host_mb']:.0f} MB of weights on host "
              "— decode will be PCIe-bound")

    result: dict[str, Any] = {
        "model_name": Path(model_path).stem,
        "n_gpu_layers": n_gpu_layers,
        # The count of runs that survived clock rejection, not the count
        # requested — the stdev below is over these samples.
        "n_runs": len(runs),
        "prompt_tokens": int(_median([r.get("prompt_tokens", 0.0) for r in runs])),
        "generated_tokens": int(_median([r.get("generated_tokens", 0.0) for r in runs])),
        "prefill_tps": round(_median(prefill), 2),
        "prefill_tps_std": round(_stdev(prefill), 2),
        "decode_tps": round(_median(decode), 2),
        "decode_tps_std": round(_stdev(decode), 2),
        "ttft_ms": round(_median(ttft), 2),
        "ttft_ms_std": round(_stdev(ttft), 2),
        "vram_delta_mb": round(delta, 1),
        "vram_total_mb": round(peak_vram, 1),
        "vram_residency": residency,
        "offload_state": offload_state,
        # Recorded so a suspect row can be attributed after the fact. Without
        # these the 2026-08-04 sweep's bimodality was uninvestigable from the
        # CSV alone and had to be reconstructed from stdout logs.
        "pstate": _modal_pstate(clock_samples),
        # Mean across accepted runs only — the clock the published number was
        # actually produced at, not an average over rejected attempts too.
        "mem_clock_mhz": round(
            sum(run_mean_clocks) / len(run_mean_clocks) if run_mean_clocks else UNKNOWN, 0),
        "throttle_reasons": _summarize_throttles(clock_samples),
        "load_time_s": round(load_time_s, 2),
        "model_size_mb": model_size_mb,
        "timing_source": source,
    }

    # Release the device before returning. Freeing the Python object is not
    # enough: without waiting for the driver to hand memory back, the next
    # model in a sweep can be handed a fragmented heap and spill even though
    # it would fit on a clean device — which made results order-dependent.
    del llm
    gc.collect()
    released = wait_for_vram_release(baseline_vram)
    if verbose and released > baseline_vram + 400:
        print(f"[warn] VRAM did not fully release: {released:.0f} MB still held "
              f"vs {baseline_vram:.0f} MB baseline — next model may spill")
    return result
