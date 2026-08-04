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

The effect on llama-2 Q4_K_M, the configuration that first showed the split:

| | Before the fix | After, at P2 | After, at P0 |
| --- | --- | --- | --- |
| Decode t/s | 53.19 / 65.75 across two passes | 61.08 ± 1.18 | 74.43 ± 0.50 |
| Clock during measurement | unrecorded | 11101 MHz | 12101 MHz |
| Cross-pass drift | 23.6% | — | — |

The middle column is the same harness measuring the same model with the clock
verified but the card unable to reach P0, because it was also driving the
display. The right column is after that was fixed too. Neither is a speedup over
the left: all three are the same model on the same GPU, and the differences are
the measurement conditions the original harness could not see.

The warmup trace caught the mechanism in the act — 49.2 t/s while pinned at
9001 MHz, stepping to 57 t/s on reaching 11101 MHz, in a single run sequence.

> Both defects share a shape: a value that was not a measurement was presented
> as though it were. The first divided two phases by the same clock; the second
> watched a clock that did not govern the workload. Neither crashed.

---

## Results

Measured 2026-08-04 on the hardware below, Armory Crate **Turbo** profile
(115 W GPU ceiling) with the dGPU **not driving a display**, full GPU offload
(`--n-gpu-layers 99`), ~256-token prompt, 127 tokens generated, **median ±
sample stdev over 5 clock-verified runs** with a discarded warmup. Every row was
timed via llama.cpp perf counters, reported `offload_state = resident`, and was
verified to have held its memory clock throughout — see
[clock verification](#throughput-is-only-comparable-at-a-comparable-clock).

| Family | Quant | Decode t/s | Prefill t/s | TTFT (ms) | Mem clock | VRAM (MB) | Size (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama2 | Q4_K_M | 74.43 ± 0.50 | 2788.19 ± 9.14 | 93.86 ± 0.30 | 12101 | 4382 | 3892 |
| llama2 | Q5_K_M | 65.31 ± 0.37 | 2671.44 ± 9.19 | 97.91 ± 0.33 | 12101 | 5036 | 4562 |
| llama2 | Q8_0 | 45.78 ± 0.10 | 2829.01 ± 7.22 | 92.59 ± 0.23 | 11561 | 7256 | 6829 |
| mistral | Q4_K_M | 71.18 ± 0.06 | 2692.24 ± 10.52 | 97.28 ± 0.42 | 12101 | 4480 | 4166 |
| mistral | Q5_K_M | 62.17 ± 0.33 | 2585.53 ± 5.77 | 101.18 ± 0.26 | 12101 | 5192 | 4894 |
| mistral | Q8_0 | 43.17 ± 0.17 | 2708.80 ± 12.18 | 96.77 ± 0.42 | 11646 | 7590 | 7339 |
| qwen2.5 | Q4_K_M | 71.69 ± 0.22 | 3355.53 ± 9.92 | 78.32 ± 0.33 | 12101 | 4704 | 4466 |
| qwen2.5 | Q5_K_M | 63.15 ± 0.23 | 3269.73 ± 17.44 | 80.21 ± 0.57 | 12101 | 5364 | 5193 |
| qwen2.5 | Q8_0 | 43.91 ± 0.19 | 3493.31 ± 8.89 | 75.24 ± 0.26 | 11665 | 7700 | 7723 |

All nine configurations measured cleanly, including Q8_0 — which required
taking the display off the discrete GPU. See
[fitting Q8_0 on 8 GB](#fitting-q8_0-on-an-8-gb-card).

![Quantization comparison](results/comparison.png)

**Prefill runs 40-75x decode.** Prefill processes the prompt in parallel and is
compute-bound; decode emits one token at a time and is memory-bandwidth-bound.
An earlier revision of this README reported ~10x from a 16-token prompt, which
is too short to saturate the GPU and understates prefill. The original pre-fix
numbers had the ratio *inverted* — see the case study above.

### Does the data behave like memory bandwidth?

If decode is bandwidth-bound, `decode_tps × weights` should be roughly constant
within a family — the same bytes-per-second moving through the memory system
regardless of how the weights are quantized. Weights here are device-resident
bytes rather than file size, since the embedding table stays on the host.

| Family | Q4_K_M | Q5_K_M | Q8_0 | Spread |
| --- | --- | --- | --- | --- |
| llama2 | 284.4 GB/s | 292.3 GB/s | 306.6 GB/s | 7.2% |
| mistral | 291.5 GB/s | 298.9 GB/s | 311.1 GB/s | 6.3% |
| qwen2.5 | 299.2 GB/s | 305.4 GB/s | 314.9 GB/s | 5.0% |

Decode falls monotonically with size in every family — Q4_K_M > Q5_K_M > Q8_0,
nine rows, no exceptions. This is the check the previous sweep failed: Qwen2.5
Q5_K_M decoded *faster* than its own Q4_K_M despite being 16% larger, and did so
reproducibly across two passes. That inversion is gone; it was a clock artifact.

Effective bandwidth is not perfectly flat — it rises 5-7% from Q4_K_M to Q8_0.
Two candidate explanations, neither yet tested: fixed per-token cost (sampling,
kernel launch, the Python loop) is amortized better at 44 t/s than at 74, and
Q8_0 dequantization is arithmetically trivial next to a K-quant, so the larger
format sits closer to being purely bandwidth-bound. The trend is consistent in
direction and magnitude across all three families.

> **Not comparable to earlier revisions of this table.** Prompt length, the
> VRAM-release fix, clock verification, the power profile and the display mode
> all changed across sweeps. Decode is roughly triple what this README reported
> on 2026-08-03. Almost none of that is a speedup — it is measurement conditions
> that were previously uncontrolled and unrecorded.

### GPU offload ladder

Q4_K_M, `n_gpu_layers` swept 0 to 99, median of 3 clock-verified runs per step.
Full per-step data in `results/offload_ladder.csv`.

| Layers | llama2 | mistral | qwen2.5 | Weights on host (llama2) |
| --- | --- | --- | --- | --- |
| 0 | 11.93 ± 0.06 | 11.83 ± 0.05 | 11.69 ± 0.02 | everything |
| 8 | 14.18 ± 0.25 | 14.35 ± 0.06 | 15.54 ± 0.24 | 3055 MB |
| 16 | 19.12 ± 0.16 | 18.51 ± 0.03 | 20.78 ± 0.12 | 2829 MB |
| 24 | 27.17 ± 0.19 | 26.57 ± 0.21 | 35.55 ± 0.31 | 2720 MB |
| 32 | ⚠️ 47.5 ± 4.73 | 61.05 ± 0.16 | 71.06 ± 0.33 | 194 MB |
| 99 | 74.29 ± 0.13 | 70.91 ± 0.44 | ⚠️ 71.05 ± 0.47 | none |

⚠️ = not clock-verified, excluded from the chart. See below.

![GPU offload ladder](results/offload_ladder.png)

**The curve is sharply non-linear, and that is the finding.** Moving three
quarters of Llama-2 onto the GPU gets you from 11.9 to 27.2 t/s. Moving the last
quarter takes you to 74.3. Any layer left on the host forces a PCIe round trip
on *every token*, so a single straggler bottlenecks the whole pipeline — you do
not get proportional benefit, you get almost nothing until nearly everything is
resident. Full offload is **6.2x** CPU-only.

This is the practical answer to "my model nearly fits — how much do I lose
offloading part of it?" The answer is: much more than the fraction suggests.

Two structural findings, properties of the models and the loader rather than of
the timing path:

- **The x-axis is not comparable across families.** Llama-2 and Mistral have 32
  transformer layers; Qwen2.5-7B has 28. So `n_gpu_layers=16` is half of one
  model and 57% of another, and Qwen is already fully offloaded by step 32
  while the others still have layers on the host.
- **llama.cpp counts the output layer separately.** For a 32-layer model,
  `n_gpu_layers=32` leaves that layer on CPU; only 33+ offloads everything.
  Llama-2 gains 56% and Mistral 16% from that single step. Qwen2.5 is flat
  (71.06 → 71.05) because with 28 layers it is already complete at 32 — the
  clearest confirmation that the x-axis means different things per family.

### Two ladder steps are not published

**`llama2` at 32 layers varies 47–60 t/s across three separate runs.** Mistral
is stable at the same step (61.05 ± 0.16) with a comparable 203 MB left on host
against Llama-2's 194 MB, so the configuration shape is not the difference. The
likely cause is the KV cache: Llama-2 uses full multi-head attention and needs
256 MB where Mistral's grouped-query attention needs 64 MB, putting Llama-2
under real memory pressure at exactly the point where almost everything else is
resident. Reproducible instability, reported rather than averaged away.

**`qwen2.5` at 99 layers collected only one clean run** before the board went
busy. Its value agrees with both its own 32-layer step and the quantization
sweep's 71.70, so nothing here is in doubt — it simply is not verified to the
standard the other rows meet.

---

## Known issues — measurements not yet trusted

This section exists because a benchmark that hides its unstable numbers is
worth less than one that names them.

### Fitting Q8_0 on an 8 GB card

A 7B model at Q8_0 fits, but only just, and only if the discrete GPU is not also
drawing your desktop. On this laptop the dGPU was driving the panel directly
(ASUS "Ultimate" / MUX mode), which permanently held ~580 MB of VRAM for the
framebuffer and compositor. Switching to hybrid mode routes the display through
the integrated GPU and hands that memory back:

| | dGPU drives display | Display on iGPU |
| --- | --- | --- |
| VRAM in use at idle | 578-985 MB | **0 MB** |
| VRAM free | 7314 MB | **7891 MB** |
| Q8_0 configurations measurable | 0 of 3 | **3 of 3** |

The margins remain thin — 88 to 406 MB depending on family — so this is a
boundary worth respecting rather than a comfortable fit. `vram_residency` is
identical between each family's Q8_0 and its Q4_K_M row (0.98 / 0.98 / 0.93),
which is the evidence that nothing is being paged.

Two things make the difference between fitting and not, and neither is visible
in file size:

- **The KV cache is architectural, not proportional.** At 512 tokens it is
  256 MB for Llama-2 (full multi-head attention, 32 KV heads over 32 layers)
  and 28 MB for Qwen2.5 (4 KV heads over 28) — a 9x range at comparable file
  sizes. Llama-2 has the smallest Q8_0 file and the least headroom.
- **The embedding table never reaches the device.** llama.cpp keeps
  `token_embd` on the host at full offload: 133 MB for a 32k vocabulary,
  552 MB for Qwen2.5's 152k. Charging VRAM for it wrongly refused Qwen2.5 Q8_0
  on a card where it fits with 159 MB to spare.

Earlier revisions of this harness ran Q8_0 anyway, at a point where it did not
fit, and that is what started this investigation. A model larger than free VRAM
still executes — the driver pages it — and returns a number that measures PCIe
transfer. Those numbers were bimodal and irreproducible. The harness now refuses
rather than producing them.

### Q8_0 rows were measured at a slightly lower clock

Q4_K_M and Q5_K_M held P0 (12101 MHz) throughout. The Q8_0 rows averaged
11561-11665 MHz, drifting between P0 and P2 across their runs — admitted because
their throughput agreed to better than 1%, but a ~4% lower clock nonetheless.
Cross-*quant* comparisons therefore carry a few percent of clock confound that
within-quant comparisons do not, and the true Q4→Q8 falloff is marginally
steeper than the table shows.

### Qwen2.5 keeps more weight on the host

Qwen2.5-7B reports `vram_residency` of ~0.93 against ~0.98 for Llama-2 and
Mistral, because its 152k-token vocabulary makes for a large embedding table
that llama.cpp leaves on the host even at full offload. `offload_state` still
reads `resident` — the layer tally and the byte split disagree, and both are
recorded. Cross-family comparisons at equal file size are therefore not
comparisons at equal device-resident bytes.

### The remaining clock spread is bounded, not eliminated

The card cannot be pinned to P0 under WDDM. What the harness guarantees is that
every published run held at least P2 throughout, and that runs aggregated
together agree with one another; it does not guarantee every run sat at exactly
the same clock. Six of the nine rows above held 12101 MHz exactly; the three
Q8_0 rows did not, as noted above.

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
| **Power profile** | Armory Crate Turbo — 115 W GPU ceiling (55 W default) |
| **GPU mode** | Hybrid — display driven by the integrated GPU, dGPU compute-only |

> **Take the display off the discrete GPU.** On a MUX-equipped laptop, running
> the dGPU as the display adapter costs ~580 MB of VRAM permanently and put
> every Q8_0 configuration out of reach. Switching to hybrid mode (ASUS
> "Standard", not "Eco" — that disables the dGPU entirely) requires a reboot and
> costs nothing for compute: CUDA runs on the dGPU either way.
>
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
| [`llm_qlab/bench_core.py`](llm_qlab/bench_core.py) | Shared measurement logic | *imported, not run directly* |
| [`llm_qlab/results_schema.py`](llm_qlab/results_schema.py) | CSV schema + migration | *imported, not run directly* |
| [`sweep_pass.ps1`](sweep_pass.ps1) | Preflight + full sweep + ladder + charts | `-SkipLadder` · `-Runs` · `-Force` |

**Three design constraints worth naming.** `benchmark.py` and `offload_ladder.py` both delegate to `llm_qlab.bench_core.benchmark_model`, so the two entry points cannot drift apart in methodology — a bug fixed in one is fixed in both. `llm_qlab.results_schema` deliberately imports no `llama_cpp`, so `compare_quants.py` runs anywhere: you do not need a CUDA build just to plot results someone else measured. And the CLI entry points stay at the repository root rather than moving into the package, because every command in this README invokes them directly and relocating them would break the documented interface to gain nothing.

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
├── sweep_pass.ps1        # Driver: preflight, full sweep, offload ladder, charts
├── llm_qlab/
│   ├── bench_core.py     # Measurement logic — shared by both benchmark CLIs
│   └── results_schema.py # CSV schema + migration (no llama_cpp dependency)
├── tests/
│   ├── test_clock_gate.py     # Clock-state admission and rejection
│   ├── test_vram_estimate.py  # KV cache and embedding-table sizing
│   ├── test_results_schema.py # Schema migration across every generation
│   └── test_dedup.py          # Superseded-row collapsing
├── pyproject.toml        # pytest configuration
├── requirements.txt
├── LICENSE
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
