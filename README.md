# llm-qlab

> **LLM Quantization Benchmarks on Consumer GPUs**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![CUDA](https://img.shields.io/badge/CUDA-13.2-green?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-yellow)

A collection of Python scripts for benchmarking quantized large language models (LLMs) on consumer-grade NVIDIA GPUs. Track prefill and decode throughput, time-to-first-token, and VRAM usage across different quantization formats.

This repo drives inference through **llama-cpp-python** (GGUF via llama.cpp, CUDA-accelerated). It measures and characterizes kernels that llama.cpp provides — it does not implement CUDA kernels of its own.

---

## ⚠️ Results currently pending re-run

The benchmark tables previously published here were **withdrawn in the 2026-08-03 methodology fix** and have not yet been regenerated.

**What was wrong:** `benchmark.py` read a `timings` dict off the final streaming chunk, but llama-cpp-python does not populate `usage` or `timings` on streamed responses. The code therefore always fell through to a wall-clock fallback that divided *both* phases by the same total elapsed time. The published "Prompt (t/s)" column was actually `prompt_tokens / total_time` — a restatement of generation speed, not a measurement of prefill. Every row satisfied `prompt_tokens / prompt_tps == generated_tokens / gen_tps` exactly, which is the signature of the bug. The tell-tale symptom was prompt throughput reading *lower* than decode throughput; prefill should be roughly an order of magnitude faster.

Two further issues were fixed alongside it:

- **No warmup.** The first inference absorbed CUDA context creation and kernel autotuning. This is why the same configuration (llama2 Q4_K_M, `n_gpu_layers=99`) reported TTFT of 86.94 ms in the comparison table but 24.14 ms in the offload ladder — a 3.6× disagreement between two tables in this same README.
- **VRAM was whole-board, not model.** `nvidia-smi` total usage was sampled *after* model load, so the figure included the Windows desktop compositor and any other GPU process.

Numbers will reappear here once re-run on the reference hardware. Charts in `results/` are likewise stale until then.

---

## 🖥️ Hardware & Environment

| Component | Details |
|-----------|------|
| **GPU** | NVIDIA RTX 5070 Laptop GPU, 8 GB VRAM (compute capability 12.0) |
| **CUDA** | 13.2 |
| **Driver** | 595.97 |
| **OS** | Windows 11 (native) |
| **Python** | 3.14.3 |
| **llama-cpp-python** | 0.3.20 (built from source) |

---

## 📊 What This Repo Tracks

Benchmarks comparing the following quantization formats:

| Format | Description |
|--------|-------------|
| `Q4_K_M` | 4-bit K-quant (medium) — best speed, lowest VRAM |
| `Q5_K_M` | 5-bit K-quant (medium) — balance of speed and quality |
| `Q8_0` | 8-bit quantization — near-FP16 quality, highest VRAM |

Metrics captured per configuration, reported as **median ± sample stdev over N runs**:

- **Prefill throughput** (t/s) — prompt processing, timed separately
- **Decode throughput** (t/s) — token generation, timed separately
- **Time-to-first-token** (TTFT, ms)
- **VRAM attributable to the model** (MB), plus whole-board usage for reference
- Model load time (s) and file size (MB)

---

## 🔬 Methodology

Getting these numbers right is most of the work, so the approach is explicit:

**Throughput comes from llama.cpp's own performance counters** (`llama_perf_context`), which report the two phases independently:

| Counter | Phase |
|---------|-------|
| `t_p_eval_ms` / `n_p_eval` | prompt processing (prefill) |
| `t_eval_ms` / `n_eval` | token generation (decode) |

Wall-clock timing around a Python generator would also capture detokenization and loop overhead, and cannot separate prefill from decode at all. If the perf-counter API is unavailable, the harness falls back to a wall-clock decomposition around TTFT and **labels the row `wall_clock_estimate`** rather than silently reporting a number that looks like a measurement but isn't. Every result carries a `timing_source` column.

**A warmup run precedes every measurement** and is discarded, so CUDA context setup and autotuning do not land in the reported figures.

**Each configuration runs N times (default 3)** and reports the median, so one scheduling hiccup cannot move the headline number. Sample standard deviation is reported alongside and drawn as error bars.

**KV cache and token state are cleared before every run.** Without this, llama-cpp-python's prompt-prefix reuse skips prefill on repeat runs of the same prompt and the prefill measurement collapses to nothing.

**VRAM is a delta.** A baseline is sampled before model load and subtracted from peak, so the figure reflects the model rather than the whole board. Unmeasured values are recorded as `-1` and excluded from aggregates — never averaged in as zero.

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/iarjunganesh/llm-qlab
cd llm-qlab
pip install -r requirements.txt
```

> **Note — llama-cpp-python source build required for CUDA 13.2 / sm_120 (RTX 5070 series):**
> The PyPI wheel does not include sm_120 CUDA kernels. Build from source:
> ```bash
> git clone https://github.com/abetlen/llama-cpp-python --recursive
> cd llama-cpp-python
> set GGML_CUDA=on
> set FORCE_CMAKE=1
> pip install .
> ```
> After building, install remaining deps from the repo root: `pip install -r requirements.txt`

### 2. Download GGUF models from Hugging Face

Use the bundled `download_model.py` helper:

```bash
# List available presets
python download_model.py --list

# Download Llama-2-7B-Chat Q4_K_M (3.9 GB)
python download_model.py --model llama2-7b

# Download Q5_K_M (4.6 GB) or Q8_0 (6.8 GB)
python download_model.py --model TheBloke/Llama-2-7B-chat-GGUF --filename llama-2-7b-chat.Q5_K_M.gguf
python download_model.py --model TheBloke/Llama-2-7B-chat-GGUF --filename llama-2-7b-chat.Q8_0.gguf
```

Or use the Hugging Face CLI directly:

```bash
huggingface-cli download TheBloke/Llama-2-7B-chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir ./models
```

### 3. Run a benchmark

```bash
python benchmark.py --model models/llama-2-7b-chat.Q4_K_M.gguf --quant-type Q4_K_M --model-family llama2 --n-gpu-layers 99
```

Run all three quantization levels:

```bash
python benchmark.py --model models/llama-2-7b-chat.Q4_K_M.gguf --quant-type Q4_K_M --model-family llama2 --n-gpu-layers 99
python benchmark.py --model models/llama-2-7b-chat.Q5_K_M.gguf --quant-type Q5_K_M --model-family llama2 --n-gpu-layers 99
python benchmark.py --model models/llama-2-7b-chat.Q8_0.gguf   --quant-type Q8_0   --model-family llama2 --n-gpu-layers 99

# Example: benchmark a second family
python benchmark.py --model models/mistral-7b-instruct-v0.1.Q4_K_M.gguf --quant-type Q4_K_M --model-family mistral --n-gpu-layers 99

# Tighter error bars for a publishable run
python benchmark.py --model models/llama-2-7b-chat.Q4_K_M.gguf --quant-type Q4_K_M --model-family llama2 --n-runs 10
```

### 4. Monitor GPU in a separate terminal

```bash
python monitor_gpu.py --interval 1
```

### 5. Compare quantization results

```bash
python compare_quants.py
python compare_quants.py --group-by model_family
```

---

## 🔧 Script Reference

| Script | Purpose | Key Args |
|--------|---------|----------|
| `benchmark.py` | Run inference benchmark | `--model`, `--quant-type`, `--model-family`, `--n-predict`, `--n-gpu-layers`, `--n-runs`, `--no-warmup`, `--prompt` |
| `compare_quants.py` | Plot & compare results | `--group-by` (`quant_type` \| `model_family`); reads `results/benchmark_results.csv` |
| `offload_ladder.py` | Sweep n_gpu_layers and plot VRAM vs speed | `--model`, `--quant-type`, `--steps`, `--n-runs` |
| `monitor_gpu.py` | Live GPU stats logger | `--interval`, `--output` |
| `download_model.py` | Download GGUF models | `--model`, `--filename`, `--list` |
| `bench_core.py` | Shared measurement logic (imported, not run directly) | — |
| `results_schema.py` | CSV schema + migration (imported, not run directly) | — |

`benchmark.py` and `offload_ladder.py` both delegate measurement to `bench_core.benchmark_model`, so the two entry points cannot drift apart in methodology. `results_schema.py` deliberately avoids importing `llama_cpp`, so `compare_quants.py` runs on any machine — you do not need a CUDA build just to plot results.

---

## 📉 GPU Offload Ladder

`offload_ladder.py` systematically varies `--n-gpu-layers` across a configurable set of steps, benchmarks the model at each level, and produces:

- A summary table printed to stdout
- `results/offload_ladder.csv` with per-step metrics
- `results/offload_ladder.png` — dual-axis plot (decode t/s with error bars vs. model VRAM)

```bash
python offload_ladder.py --model models/llama-2-7b-chat.Q4_K_M.gguf --quant-type Q4_K_M
python offload_ladder.py --model models/llama-2-7b-chat.Q4_K_M.gguf --quant-type Q4_K_M --steps 0,16,32,99
```

---

## 🗃️ Results Schema

Results land in `results/benchmark_results.csv`:

| Column | Meaning |
|--------|---------|
| `model_name`, `model_family`, `quant_type` | Identity of the configuration |
| `n_gpu_layers`, `n_runs` | Configuration and sample size |
| `prompt_tokens`, `generated_tokens` | Token counts as reported by llama.cpp |
| `prefill_tps`, `prefill_tps_std` | Prompt processing throughput |
| `decode_tps`, `decode_tps_std` | Generation throughput |
| `ttft_ms`, `ttft_ms_std` | Time to first token |
| `vram_delta_mb` | VRAM attributable to the model |
| `vram_total_mb` | Whole-board VRAM at peak |
| `load_time_s`, `model_size_mb` | Load cost and file size |
| `timing_source` | `perf_counters`, `wall_clock_estimate`, or `legacy_invalid` |

**Schema migration.** CSVs from earlier revisions are migrated automatically when new rows are appended, and the original is preserved as `benchmark_results.legacy.<timestamp>.csv`. Rows written before the methodology fix are marked `legacy_invalid`; their throughput columns are set to `-1` rather than carried forward, because the underlying numbers are not recoverable after the fact. `compare_quants.py` excludes those rows from charts and tables and prints a warning naming how many it dropped.

---

## 📁 Repository Structure

```
llm-qlab/
├── README.md
├── requirements.txt
├── benchmark.py          # Main benchmark runner (CLI)
├── bench_core.py         # Shared measurement logic
├── results_schema.py     # CSV schema + migration (no llama_cpp dependency)
├── compare_quants.py     # Comparison plots & table
├── offload_ladder.py     # GPU offload ladder sweep
├── monitor_gpu.py        # Live GPU monitor
├── download_model.py     # GGUF model downloader
├── .gitignore
└── results/
    ├── benchmark_results.csv       # Benchmark output (ignored by git)
    ├── offload_ladder.csv          # Offload ladder output (ignored by git)
    ├── comparison.png              # Generated chart
    ├── comparison_by_family.png    # Family comparison chart (generated)
    └── offload_ladder.png          # Offload ladder plot (generated)
```

---

## 🤝 Contributing

PRs and issues welcome! If you have results from other GPUs or models, feel free to open a PR with your data — please include the `timing_source` column so methodology is auditable.

---

## 📄 License

MIT
