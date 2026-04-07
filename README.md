# llm-qlab

> **LLM Quantization Benchmarks on Consumer GPUs**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![CUDA](https://img.shields.io/badge/CUDA-13.2-green?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-yellow)

A collection of Python scripts for benchmarking quantized large language models (LLMs) on consumer-grade NVIDIA GPUs. Track inference speed, VRAM usage, and quality trade-offs across different quantization formats.

---

## 🖥️ Hardware & Environment

| Component | Details |
|-----------|---------|
| **GPU** | NVIDIA RTX 5070 8 GB |
| **CUDA** | 13.2 |
| **Driver** | 595.97 |
| **OS** | Linux (tested) / Windows WSL2 |

---

## 📊 What This Repo Tracks

Benchmarks comparing the following quantization formats using **llama.cpp** and **ExLlamaV2**:

| Format | Description |
|--------|-------------|
| `Q4_K_M` | 4-bit K-quant (medium) — sweet spot for 8 GB VRAM |
| `Q8_0` | 8-bit quantization — near FP16 quality |
| `FP16` | Full half-precision — baseline reference |

Metrics captured per run:
- Tokens / second (prompt processing & generation)
- VRAM usage (MB)
- Model load time (seconds)
- Model file size (MB)

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/iarjunganesh/llm-qlab
cd llm-qlab
pip install -r requirements.txt
```

> **Note:** Install PyTorch with CUDA 13.2 support separately if needed:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Run a benchmark

```bash
python benchmark.py \
  --model models/model_Q4_K_M.gguf \
  --quant-type Q4_K_M \
  --n-gpu-layers 99
```

### 3. Monitor GPU in a separate terminal

```bash
python monitor_gpu.py --interval 1
```

### 4. Compare quantization results

```bash
python compare_quants.py
```

---

## 🔧 Script Reference

| Script | Purpose | Key Args |
|--------|---------|----------|
| `benchmark.py` | Run inference benchmark | `--model`, `--quant-type`, `--n-predict`, `--n-gpu-layers` |
| `compare_quants.py` | Plot & compare results | reads `results/benchmark_results.csv` |
| `monitor_gpu.py` | Live GPU stats logger | `--interval`, `--output` |

---

## 📈 Benchmark Results (Placeholder)

> Replace this table with your actual results after running benchmarks.

| Model | Quant | Tokens/sec (gen) | VRAM (MB) | Load Time (s) | Size (MB) |
|-------|-------|-----------------|-----------|---------------|-----------|
| Llama-3-8B | Q4_K_M | — | — | — | — |
| Llama-3-8B | Q8_0 | — | — | — | — |
| Llama-3-8B | FP16 | — | — | — | — |
| Mistral-7B | Q4_K_M | — | — | — | — |
| Mistral-7B | Q8_0 | — | — | — | — |

---

## 📁 Repository Structure

```
llm-qlab/
├── README.md
├── requirements.txt
├── benchmark.py          # Main benchmark runner
├── compare_quants.py     # Comparison plots & table
├── monitor_gpu.py        # Live GPU monitor
├── .gitignore
└── results/
    └── .gitkeep          # Results saved here (ignored by git)
```

---

## 🤝 Contributing

PRs and issues welcome! If you have results from other GPUs or models, feel free to open a PR with your data.

---

## 📄 License

MIT
