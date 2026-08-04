# llm-qlab

> **How fast is a quantized LLM on a consumer GPU — and how would you know if your benchmark were lying to you?**

<!-- Row 1 — project -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows_11_native-0078D4?logo=windows&logoColor=white)](https://www.microsoft.com/windows)

<!-- Row 2 — the inference stack under measurement -->
[![CUDA](https://img.shields.io/badge/CUDA-13.3-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-CUDA_backend-222222)](https://github.com/ggerganov/llama.cpp)
[![llama-cpp-python](https://img.shields.io/badge/llama--cpp--python-0.3.20_source_build-3776AB?logo=python&logoColor=white)](https://github.com/abetlen/llama-cpp-python)
[![GGUF](https://img.shields.io/badge/GGUF-Q4__K__M_%C2%B7_Q5__K__M_%C2%B7_Q8__0-8A2BE2)](https://huggingface.co/docs/hub/gguf)

<!-- Row 3 — the hardware axis -->
[![GPU](https://img.shields.io/badge/GPU-RTX_5070_Laptop_8GB-1F2937?logo=nvidia&logoColor=76B900)](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/)
[![Arch](https://img.shields.io/badge/Blackwell-sm__120-1F2937?logo=nvidia&logoColor=76B900)](https://developer.nvidia.com/cuda-gpus)

<!-- Row 4 — how the numbers are produced -->
[![Timing](https://img.shields.io/badge/timing-llama.cpp_perf_counters-2ea44f)](#methodology)
[![Runs](https://img.shields.io/badge/reported-median_%C2%B1_stdev-2ea44f)](#methodology)
[![Warmup](https://img.shields.io/badge/warmup-discarded-2ea44f)](#methodology)
[![Clocks](https://img.shields.io/badge/GPU_clock-verified_per_run-2ea44f)](#throughput-is-only-comparable-at-a-comparable-clock)

---

## What Is This?

llm-qlab is a **benchmark harness for quantized LLM inference on consumer NVIDIA hardware**. It measures how GGUF quantization formats trade throughput against memory, and how performance changes as model layers move between CPU and GPU.

It drives inference through [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (GGUF via llama.cpp, CUDA-accelerated). It **measures and characterizes** kernels that llama.cpp provides — it does not implement CUDA kernels of its own.

The interesting part of this repo is not the numbers. It is the **methodology used to produce numbers you can defend** — and a worked example of what happens when that methodology is wrong.

---

## The Problem

Inference benchmarks fail quietly. A crash is obvious; a plausible wrong number is not.

The two phases of LLM inference have completely different performance characteristics:

| Phase | What happens | Bound by |
| --- | --- | --- |
| **Prefill** | The prompt is processed in parallel, all tokens at once | Compute |
| **Decode** | Tokens are generated one at a time, each depending on the last | Memory bandwidth |

Prefill is typically **an order of magnitude faster** than decode. A harness that reports them as a single "tokens/sec" number, or that accidentally divides both by the same elapsed time, produces figures that look entirely reasonable and are entirely meaningless.

This repo got that wrong, shipped the wrong numbers, and then caught it. [The post-mortem is below](#case-study-a-benchmark-that-lied).

---

## Methodology

Getting these numbers right is most of the work, so the approach is explicit and auditable.

### Throughput comes from llama.cpp's own counters

Not from wall clock. `llama_perf_context` reports the two phases independently:

| Counter | Phase measured |
| --- | --- |
| `t_p_eval_ms` / `n_p_eval` | Prompt processing (prefill) |
| `t_eval_ms` / `n_eval` | Token generation (decode) |

Wall-clock timing around a Python generator also captures detokenization and loop overhead, and cannot separate the phases at all.

### Every number is honest about its provenance

If the perf-counter API is unavailable, the harness falls back to a wall-clock decomposition around TTFT and **labels the row `wall_clock_estimate`**. It never presents an estimate as a measurement. Every result carries a `timing_source` column.

### Throughput is only comparable at a comparable clock

Decode is memory-bandwidth-bound, so the memory clock — not the SM clock — sets
decode throughput. This GPU migrates between memory P-states on its own while
work is running, and under WDDM the clock cannot be locked:

| P-state | Memory clock | Relative |
| --- | --- | --- |
| P0 | 12101 MHz | 1.00 |
| P2 | 11101 MHz | 0.92 |
| P4 | 9001 MHz | 0.74 |

Since the clock cannot be *controlled*, it is **observed and used to admit or
reject each run**:

- A background thread samples `pstate`, `clocks.mem` and the throttle-reason
  bitmask throughout every run. Sampling before and after is not enough — the
  card returns to its idle state within a fraction of a second of the work
  stopping, so a between-runs reading reports P5/810 MHz no matter how the run
  itself executed.
- A run whose memory clock dipped below 90% of the observed maximum is
  **discarded**, not averaged in. That floor sits between P4 and P2 because
  P2→P4 is the transition that matters.
- Runs continue until enough clean samples exist or an attempt budget is spent.
- The accepted runs must agree with each other, so their stdev describes the
  model rather than the power state.
- `pstate`, `mem_clock_mhz` and `throttle_reasons` are recorded per row, so any
  suspect result can be attributed afterwards instead of re-derived from logs.

Being at the power cap is *not* treated as a fault: a boosting GPU sits at its
power cap essentially all the time. Only forced slowdowns — hardware, thermal,
power-brake — disqualify a run.

### Four more rules the harness enforces

| Rule | Why |
| --- | --- |
| **Warmup run, discarded** | The first inference absorbs CUDA context creation and kernel autotuning. Measuring it inflates TTFT several-fold. |
| **N runs, median ± stdev** | One scheduling hiccup must not move the headline number. Variance is reported, not hidden. |
| **KV cache cleared between runs** | llama-cpp-python reuses matching prompt prefixes. Without a clear, prefill is skipped on repeat runs and the measurement collapses to nothing. |
| **VRAM as a delta** | A baseline is sampled *before* model load and subtracted from peak, so the figure reflects the model — not the desktop compositor sharing the GPU. |

Unmeasured values are recorded as `-1` and **excluded** from aggregates, never averaged in as zero.

A configuration that cannot be measured cleanly is **not published**. Rows carry
`timing_source = unstable_clocks` and are excluded from charts and tables, and a
model that does not fit in free VRAM is refused outright rather than measured
while the driver pages it.

---

## Case Study: A Benchmark That Lied

The published results of this repo were wrong for four months. Here is exactly how.

**The symptom, visible in the README the whole time:** prompt throughput read *lower* than decode throughput. For a 16-token prompt on a 7B model, prefill should be roughly an order of magnitude faster. It was reported as roughly four times slower.

**The cause:** `benchmark.py` read a `timings` dict off the final streaming chunk. llama-cpp-python does not populate `usage` or `timings` on streamed responses, so that dict was always empty and the code always took its wall-clock fallback — which divided *both* phases by the same total elapsed time.

**The proof:** every published row satisfied this identity exactly.

```text
prompt_tokens / prompt_tps  ==  generated_tokens / gen_tps
```

The reported "prompt t/s" was just `prompt_tokens / total_time`. It was a restatement of generation speed wearing a different label.

**A second defect it masked:** with no warmup run, the same configuration (llama2 Q4_K_M, full offload) reported TTFT of **86.94 ms** in one table and **24.14 ms** in another — a 3.6× disagreement between two tables in the same document, caused by CUDA context setup landing inside the first measurement.

**The fix:** read llama.cpp's counters directly, add a discarded warmup, report medians across repeated runs, clear KV state between them, and label any fallback explicitly. Legacy result rows are migrated but marked `legacy_invalid` with their throughput dropped — those numbers are not recoverable after the fact, and carrying them forward under new column names would have been worse than losing them.

> The history is documented rather than rewritten. A benchmark repo that quietly deletes its wrong numbers is less trustworthy than one that explains them.

---

## Case Study: Watching The Wrong Clock

The second defect was found by auditing results that had already passed every
check the harness had.

**The symptom.** A three-family sweep was run twice, back to back, to
demonstrate reproducibility. Six of nine configurations agreed to within 1.6%.
Three did not: two by ~23%, one by ~34%.

**The false lead.** The obvious story was that the divergent rows were the ones
near the VRAM ceiling, so the natural conclusion was memory pressure. Tensor
placement turned out to be byte-identical across both passes for all nine
configurations, which killed that explanation outright.

**The real tell.** Llama-2 was bimodal across *all three* quantization formats,
at a near-constant ratio:

| Quant | Low mode | High mode | Ratio |
| --- | --- | --- | --- |
| Q4_K_M | 53.19 | 65.75 | 1.236 |
| Q5_K_M | 46.92 | 57.99 | 1.236 |
| Q8_0 | 33.00 | 40.53 | 1.228 |

A constant multiplicative factor that survives changing the quantization format
is not a property of the model or its memory layout. It is a clock.

**The cause.** The warmup routine waited for `clocks.sm` to stabilize before
measuring. Decode is memory-bandwidth-bound, so the governing clock is
`clocks.mem`, which was never sampled. Polling the card under load showed it
cycling between three memory P-states — 12101, 11101 and 9001 MHz — and
**11101 / 9001 = 1.233**, matching the observed ratio to three significant
figures.

**The second failure underneath it.** The warmup stopped once two consecutive
runs agreed within 5%. All 18 runs across both passes reported convergence and
the "still moving" warning never fired once — yet four configurations then
measured 13–27% away from where warmup left them. The decisive case is
`llama2 Q5_K_M`: two consecutive warmup runs agreed to **0.1%** (58.05 → 57.99
t/s) and the following five-run median came in at 47.09. Two samples agreeing
cannot distinguish a plateau mid-ramp from steady state.

**Why the CSV could not answer this.** Neither clock was recorded, so the whole
investigation had to be reconstructed from stdout logs that happened to still be
on disk. That is why `pstate`, `mem_clock_mhz` and `throttle_reasons` are now
columns.

**The fix.** Sample the memory clock *during* each run from a background thread,
reject runs that left the boost state, require a streak of agreeing runs rather
than a pair, and flag any row that cannot be measured cleanly as
`unstable_clocks` so it is excluded from charts rather than merely warned about.
The effect on the same configuration:

| | Before | After |
| --- | --- | --- |
| llama2 Q4_K_M decode | 53.19 / 65.75 across passes | 61.08 ± 1.18 |
| Reported stdev | ± 1.21 | ± 0.17 on a verification run |
| Cross-pass drift | 23.6% | clock recorded per row |

The warmup trace caught the mechanism in the act — 49.2 t/s while pinned at
9001 MHz, stepping to 57 t/s on reaching 11101 MHz, in a single run sequence.

> Both defects share a shape: a value that was not a measurement was presented
> as though it were. The first divided two phases by the same clock; the second
> watched a clock that did not govern the workload. Neither crashed.

---

## Results

Measured 2026-08-04 on the hardware below, Armory Crate **Turbo** profile
(108 W GPU ceiling against a 55 W default), full GPU offload
(`--n-gpu-layers 99`), ~256-token prompt, 127 tokens generated, **median ±
sample stdev over 5 clock-verified runs** with a discarded warmup. Every row was
timed via llama.cpp perf counters and every run was verified to have held its
memory clock throughout — see [clock verification](#throughput-is-only-comparable-at-a-comparable-clock).

| Family | Quant | Decode t/s | Prefill t/s | TTFT (ms) | Mem clock | VRAM (MB) | Size (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama2 | Q4_K_M | 61.08 ± 1.18 | 1662.97 ± 54.06 | 156.59 ± 5.31 | 11101 | 4558 | 3892 |
| llama2 | Q5_K_M | 51.17 ± 0.37 | 1530.23 ± 12.79 | 170.51 ± 1.42 | 11101 | 5047 | 4562 |
| mistral | Q4_K_M | 59.89 ± 0.51 | 1648.19 ± 16.75 | 157.99 ± 1.63 | 11101 | 4492 | 4166 |
| mistral | Q5_K_M | 49.07 ± 0.13 | 1472.59 ± 5.03 | 176.77 ± 0.60 | 11141 | 5198 | 4894 |
| qwen2.5 | Q4_K_M | 60.63 ± 0.08 | 2112.14 ± 2.52 | 123.40 ± 0.14 | 11101 | 4704 | 4466 |
| qwen2.5 | Q5_K_M | 54.18 ± 0.07 | 2065.78 ± 4.32 | 126.17 ± 0.30 | 11545 | 5364 | 5193 |

All three Q8_0 configurations were **refused, not measured** — see
[the 8 GB ceiling](#q8_0-does-not-fit-an-8-gb-card).

![Quantization comparison](results/comparison.png)

**Prefill runs roughly 30x decode.** Prefill processes the prompt in parallel
and is compute-bound; decode emits one token at a time and is
memory-bandwidth-bound. An earlier revision of this README reported ~10x from a
16-token prompt, which is too short to saturate the GPU and understates prefill.
The original pre-fix numbers had the ratio *inverted* — see the case study above.

### Does the data behave like memory bandwidth?

If decode is bandwidth-bound, `decode_tps × model_size` should be roughly
constant within a family — the same bytes-per-second moving through the memory
system regardless of how the weights are quantized.

| Family | Q4_K_M | Q5_K_M | Ratio |
| --- | --- | --- | --- |
| llama2 | 237.7 GB/s | 233.4 GB/s | 0.98 |
| mistral | 249.5 GB/s | 240.1 GB/s | 0.96 |
| qwen2.5 | 270.8 GB/s | 281.4 GB/s | 1.04 |

Within 4% in every family. This is the check the previous sweep failed: Qwen2.5
Q5_K_M decoded *faster* than its own Q4_K_M despite being 16% larger, and did so
reproducibly across two passes. That inversion is gone, and it was a clock
artifact rather than a property of the model.

Qwen sustains ~15% more effective bandwidth than Llama-2 across both quants.
That is consistent with its placement: Qwen2.5-7B has 28 transformer layers to
Llama-2's 32 and keeps a much larger embedding table on the host
(`vram_residency` 0.93 versus 0.98), so less weight traffic crosses the memory
bus per token than file size alone suggests.

> **Not comparable to earlier revisions of this table.** Prompt length, the
> VRAM-release fix and clock verification all changed between sweeps. Decode is
> roughly double what this README reported on 2026-08-03, and most of that is a
> change in *measurement conditions*, not a speedup. The older numbers were
> averages over an unrecorded mix of memory P-states.

### GPU offload ladder — withheld pending re-measurement

The ladder was last measured under a harness revision with neither the
VRAM-release fix nor clock verification, so those numbers carry both defects
and are not republished. `results/offload_ladder.csv` is not shipped. The sweep
needs a full re-run.

Two structural findings from that work do survive, because they are properties
of the models and the loader rather than of the timing path:

- **The x-axis is not comparable across families.** Llama-2 and Mistral have 32
  transformer layers; Qwen2.5-7B has 28. So `n_gpu_layers=16` is half of one
  model and 57% of another, and Qwen is already fully offloaded by step 32
  while the others still have layers on the host.
- **llama.cpp counts the output layer separately.** For a 32-layer model,
  `n_gpu_layers=32` leaves that layer on CPU; only 33+ offloads everything.
  That is the 32 → 99 step visible for Llama-2 and Mistral (+~130 MB VRAM) and
  absent for Qwen.

---

## Known issues — measurements not yet trusted

This section exists because a benchmark that hides its unstable numbers is
worth less than one that names them.

### Q8_0 does not fit an 8 GB card

All three Q8_0 configurations are **refused before loading**, and this is the
answer rather than a gap in it:

| Family | Weights | Required (+14% runtime) | Free VRAM | Verdict |
| --- | --- | --- | --- | --- |
| llama2 | 6829 MB | 7854 MB | 7339 MB | refused |
| mistral | 7339 MB | 8440 MB | 7321 MB | refused |
| qwen2.5 | 7723 MB | 8882 MB | 7312 MB | refused |

The board has 8151 MB total with ~560 MB held by the desktop compositor. A 7B
model at Q8_0 needs its weights plus KV cache, compute buffers and CUDA context;
none of the three fits.

Earlier revisions of this harness *did* run them, and that is precisely the
problem. A model larger than free VRAM still executes — the driver pages it —
and returns a number that measures PCIe transfer rather than the model. Those
numbers looked plausible and were bimodal and irreproducible, which is what
started this whole investigation. Refusing to produce them is the fix.

To measure Q8_0 at 7B you need a larger card. To characterize this one, the
useful statement is the boundary itself.

### Qwen2.5 keeps more weight on the host

Qwen2.5-7B reports `vram_residency` of ~0.93 against ~0.98 for Llama-2 and
Mistral, because its 152k-token vocabulary makes for a large embedding table
that llama.cpp leaves on the host even at full offload. `offload_state` still
reads `resident` — the layer tally and the byte split disagree, and both are
recorded. Cross-family comparisons at equal file size are therefore not
comparisons at equal device-resident bytes.

### The remaining clock spread is bounded, not eliminated

The card cannot be pinned to P0 under WDDM. What the harness guarantees is that
every published run held at least P2 throughout and that runs aggregated
together agree with one another; it does not guarantee every run sat at exactly
the same clock. Two rows in the table above were measured slightly above P2
(11545 and 11141 MHz), so cross-*family* comparisons carry a residual
uncertainty of a few percent. Within-family comparisons at 11101 MHz do not.

### The offload ladder needs re-running

Ladder numbers predate both the VRAM-release fix and clock verification, so they
are withheld entirely rather than republished with caveats.

---

## Hardware & Environment

| Component | Details |
| --- | --- |
| **GPU** | NVIDIA GeForce RTX 5070 Laptop GPU, 8 GB VRAM (Blackwell, compute capability 12.0 / sm_120) |
| **CUDA** | 13.3 |
| **Driver** | 610.88 |
| **OS** | Windows 11 (native) |
| **Python** | 3.14 |
| **llama-cpp-python** | 0.3.20, built from source with `GGML_CUDA=on` |
| **Host compiler** | MSVC 14.44 (Visual Studio Build Tools 2022) |
| **Power profile** | Armory Crate Turbo — 108 W GPU ceiling (55 W default, 115 W max) |

> **Laptop GPUs need the highest power profile.** Under WDDM the memory clock
> cannot be locked, and a conservative profile leaves the card oscillating
> between P-states mid-run. The harness will tell you: rows measured at an
> unverified clock are flagged rather than published.
>
> **sm_120 requires a source build.** The PyPI wheel ships no kernels for the RTX 50-series. See [Quick Start](#quick-start).

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/iarjunganesh/llm-qlab
cd llm-qlab
python -m venv .venv
pip install -r requirements.txt
```

### 2. Build llama-cpp-python with CUDA for your architecture

The PyPI wheel does not include `sm_120` kernels, so the RTX 50-series needs a source build. This requires the CUDA toolkit and a C++ host compiler (MSVC on Windows, gcc on Linux).

```bash
# Windows — from a Developer Command Prompt (vcvars64.bat)
set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120
set FORCE_CMAKE=1
pip install llama-cpp-python==0.3.20 --no-binary llama-cpp-python

# Linux
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120" \
  pip install llama-cpp-python==0.3.20 --no-binary llama-cpp-python
```

Adjust `CMAKE_CUDA_ARCHITECTURES` for your GPU (`89` = Ada / RTX 40-series, `90` = Hopper, `120` = Blackwell). Verify:

```bash
python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"
```

### 3. Download GGUF models

```bash
python download_model.py --list                 # show presets
python download_model.py --model llama2-7b      # Llama-2-7B Q4_K_M (3.8 GB)

# any repo + file
python download_model.py --model TheBloke/Llama-2-7B-chat-GGUF \
                         --filename llama-2-7b-chat.Q8_0.gguf
```

### 4. Benchmark

```bash
python benchmark.py --model models/llama-2-7b-chat.Q4_K_M.gguf \
                    --quant-type Q4_K_M --model-family llama2 --n-gpu-layers 99

# tighter error bars for a publishable run
python benchmark.py --model models/llama-2-7b-chat.Q4_K_M.gguf \
                    --quant-type Q4_K_M --model-family llama2 --n-runs 10
```

### 5. Compare and sweep

```bash
python compare_quants.py                        # chart + table by quant format
python compare_quants.py --group-by model_family
python offload_ladder.py --model models/llama-2-7b-chat.Q4_K_M.gguf --quant-type Q4_K_M
python monitor_gpu.py --interval 1              # live GPU stats, separate terminal
```

---

## Script Reference

| Script | Purpose | Key arguments |
| --- | --- | --- |
| [`benchmark.py`](benchmark.py) | Run an inference benchmark | `--model` · `--quant-type` · `--model-family` · `--n-gpu-layers` · `--n-runs` · `--n-predict` · `--no-warmup` |
| [`offload_ladder.py`](offload_ladder.py) | Sweep GPU layer offload | `--model` · `--quant-type` · `--steps` · `--n-runs` |
| [`compare_quants.py`](compare_quants.py) | Charts + markdown tables | `--group-by` (`quant_type` \| `model_family`) |
| [`monitor_gpu.py`](monitor_gpu.py) | Live GPU stats logger | `--interval` · `--output` |
| [`download_model.py`](download_model.py) | Fetch GGUF models | `--model` · `--filename` · `--list` |
| [`bench_core.py`](bench_core.py) | Shared measurement logic | *imported, not run directly* |
| [`results_schema.py`](results_schema.py) | CSV schema + migration | *imported, not run directly* |
| [`sweep_pass.ps1`](sweep_pass.ps1) | One full 9-configuration sweep | *no arguments* |

**Two design constraints worth naming.** `benchmark.py` and `offload_ladder.py` both delegate to `bench_core.benchmark_model`, so the two entry points cannot drift apart in methodology — a bug fixed in one is fixed in both. And `results_schema.py` deliberately imports no `llama_cpp`, so `compare_quants.py` runs anywhere: you do not need a CUDA build just to plot results someone else measured.

---

## Results Schema

Results land in `results/benchmark_results.csv`.

| Column | Meaning |
| --- | --- |
| `model_name` · `model_family` · `quant_type` | Identity of the configuration |
| `n_gpu_layers` · `n_runs` | Configuration and sample size |
| `prompt_tokens` · `generated_tokens` | Token counts as reported by llama.cpp |
| `prefill_tps` · `prefill_tps_std` | Prompt processing throughput |
| `decode_tps` · `decode_tps_std` | Token generation throughput |
| `ttft_ms` · `ttft_ms_std` | Time to first token |
| `vram_delta_mb` | VRAM attributable to the model |
| `vram_total_mb` | Whole-board VRAM at peak |
| `vram_residency` · `offload_state` | Fraction of weight bytes on device, and whether every layer was offloaded |
| `pstate` · `mem_clock_mhz` | GPU performance state and mean memory clock during the accepted runs |
| `throttle_reasons` | Throttle bitmask observed under load, or `none` |
| `load_time_s` · `model_size_mb` | Load cost and file size |
| `timing_source` | `perf_counters` · `wall_clock_estimate` · `unstable_clocks` · `skipped_insufficient_vram` · `legacy_invalid` |

**Migration.** CSVs from earlier revisions are migrated automatically when new rows are appended, and the original is preserved as `benchmark_results.legacy.<timestamp>.csv`. Adding columns widens old rows rather than invalidating them — only layouts produced by the wall-clock timing bug are marked `legacy_invalid`. Rows excluded from charts are named in a warning rather than dropped silently.

**Re-runs supersede.** `benchmark.py` appends, so re-measuring a configuration leaves the earlier row in the file. Readers keep only the newest row per `(model_name, quant_type, n_gpu_layers)` and report how many were superseded, so a corrected measurement always wins over the one it replaced.

---

## Project Structure

```text
llm-qlab/
├── benchmark.py          # CLI: single-configuration benchmark
├── offload_ladder.py     # CLI: n_gpu_layers sweep
├── compare_quants.py     # CLI: charts + markdown tables
├── monitor_gpu.py        # CLI: live GPU stats
├── download_model.py     # CLI: GGUF fetcher
├── bench_core.py         # Measurement logic — shared by both benchmark CLIs
├── results_schema.py     # CSV schema + migration (no llama_cpp dependency)
├── sweep_pass.ps1        # Driver: one full 9-configuration sweep
├── test_clock_gate.py    # Clock-state admission and rejection
├── test_results_schema.py# Schema migration across every generation
├── test_dedup.py         # Superseded-row collapsing
├── requirements.txt
└── results/
    ├── benchmark_results.csv    # published measurements (tracked)
    ├── comparison.png           # generated chart
    ├── comparison_by_family.png # generated chart
    ├── logs/                    # per-run stdout (git-ignored)
    └── archive/                 # superseded sweeps (git-ignored)
```

The results CSV is tracked deliberately: the tables above should be
reproducible from the file that produced them, and per-row `timing_source`
makes every published number auditable. Raw logs and superseded sweeps stay
local.

---

## Roadmap

| Status | Item |
| --- | --- |
| ✅ | Prefill/decode separation via llama.cpp perf counters |
| ✅ | Warmup, repeated runs, variance reporting |
| ✅ | Model-attributable VRAM measurement |
| ✅ | Tensor placement read from llama.cpp's own loader accounting |
| ✅ | GPU clock-state verification, per-run rejection and recording |
| ◻️ | Re-run the GPU offload ladder under the current harness |
| ◻️ | Batch-size and context-length sweeps (throughput under concurrency) |
| ◻️ | Quality regression alongside speed — perplexity per quantization format |
| ◻️ | Second hardware axis: datacenter GPU comparison |
| ◻️ | CI smoke test on a CPU-only backend |

---

## Contributing

Results from other GPUs and models are welcome — the schema is hardware-agnostic. Please include the `timing_source` column so the methodology behind your numbers is auditable.

---

## License

MIT — see [LICENSE](LICENSE).
