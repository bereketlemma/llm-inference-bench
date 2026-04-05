# llm-inference-bench

[![CI](https://github.com/bereketlemma/llm-inference-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/bereketlemma/llm-inference-bench/actions/workflows/ci.yml)

A production-grade LLM inference benchmarking framework built on [vLLM](https://github.com/vllm-project/vllm). Measures throughput, latency percentiles (p50/p90/p99), and requests/sec across quantization formats (FP16, AWQ-INT4 Marlin) and batch sizes on open-source models. Includes an interactive Next.js dashboard for visualizing results.

**Live dashboard → [bench.bereketlemma.com](https://bench.bereketlemma.com/)**  
Deployed on Vercel · Custom domain via Cloudflare

---

## Live Benchmark Results

Benchmarks run on **NVIDIA L4 24GB** · **vLLM 0.16.0** · **April 5, 2026**  
Models: `mistralai/Mistral-7B-v0.1` (FP16) vs `TheBloke/Mistral-7B-v0.1-AWQ` (INT4-AWQ Marlin)  
10 measurement runs per configuration · 3 warmup runs · greedy decoding (temp=0.0)

### FP16 — Mistral-7B-v0.1

| Batch | Tokens | P50 (ms) | P99 (ms) | Throughput (tok/s) | RPS |
|------:|-------:|---------:|---------:|-------------------:|----:|
| 1 | 128 | 3587.0 | 3590.0 | 17.9 | 1.00 |
| 1 | 256 | 7169.7 | 7176.2 | 17.9 | 0.50 |
| 1 | 512 | 14353.3 | 14355.3 | 17.8 | 0.25 |
| 4 | 128 | 1872.0 | 1880.0 | 68.5 | 2.14 |
| 4 | 256 | 3740.0 | 3760.0 | 68.3 | 1.07 |
| 4 | 512 | 7520.0 | 7540.0 | 68.0 | 0.53 |
| 8 | 128 | 1920.0 | 1940.0 | 133.9 | 4.17 |
| 8 | 256 | 3840.0 | 3860.0 | 133.5 | 2.08 |
| 8 | 512 | 30590.4 | 30600.1 | 133.9 | 0.26 |

### INT4-AWQ Marlin — TheBloke/Mistral-7B-v0.1-AWQ

| Batch | Tokens | P50 (ms) | P99 (ms) | Throughput (tok/s) | RPS |
|------:|-------:|---------:|---------:|-------------------:|----:|
| 1 | 128 | 2084.1 | 2086.5 | 61.4 | 0.48 |
| 1 | 256 | 4201.2 | 4205.0 | 60.9 | 0.24 |
| 1 | 512 | 8460.7 | 8464.2 | 60.5 | 0.12 |
| 4 | 128 | 2180.1 | 2180.5 | 234.9 | 1.83 |
| 4 | 256 | 4393.6 | 4395.1 | 233.1 | 0.91 |
| 4 | 512 | 8990.1 | 8990.9 | 227.8 | 0.44 |
| 8 | 128 | 2264.0 | 2265.7 | **452.3** | 3.53 |
| 8 | 256 | 4590.0 | 4590.4 | 446.2 | 1.74 |
| 8 | 512 | 9544.8 | 9548.0 | 429.1 | 0.84 |

### Summary

| Metric | Value |
|---|---|
| INT4-AWQ throughput speedup (avg) | **3.35x** |
| P99 latency reduction (avg) | **37.5%** |
| Peak throughput (INT4-AWQ, BS=8, 128 tok) | **452.3 tok/s** |
| Peak throughput (FP16, BS=8, 128 tok) | 133.9 tok/s |

---

## Project Structure

```
llm-inference-bench/
├── benchmark/
│   ├── __init__.py
│   ├── config.py          # BenchmarkConfig dataclass with dev/production presets
│   ├── metrics.py         # p50/p90/p99, throughput, speedup calculations
│   └── runner.py          # vLLM model loading and benchmark orchestration
├── visualize/
│   ├── __init__.py
│   └── plot_results.py    # Matplotlib chart generation
├── frontend/              # Next.js 14 interactive dashboard
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx
│   │   │   └── api/benchmark-data/route.ts   # API route serving CSV data
│   │   ├── components/
│   │   │   └── Dashboard.tsx  # Recharts visualizations, PDF export, hacker UI
│   │   └── lib/
│   │       └── benchmark-data.ts   # Typed benchmark data and computed summaries
│   ├── package.json
│   └── next.config.js
├── results/
│   ├── Data/
│   │   ├── fp16_results.csv
│   │   └── awq_marlin_results.csv
│   └── charts/
├── tests/
│   ├── __init__.py
│   └── test_metrics.py    # 12 unit tests for metrics module
├── scripts/
│   └── run_benchmark.sh   # GPU pod helper — runs dev or prod benchmark suite
├── .dockerignore          # Excludes venv, frontend/, .git, results, editor files
├── Dockerfile             # NVIDIA CUDA 12.1 + Python 3.10
├── requirements.txt
└── main.py                # Entry point — runs full benchmark suite
```

---

## Setup

### Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (**required** — vLLM has no CPU mode)
- 24GB+ VRAM for Mistral-7B FP16 (L4, A10, A100, or equivalent)
- ~16GB VRAM for the AWQ-INT4 variant

> For local development and unit testing (no GPU), you can write and test the `benchmark/metrics.py`, `benchmark/config.py`, and `visualize/` modules without a GPU. The vLLM runner is only invoked at runtime. Use RunPod (~$0.20/hr A100), Lambda Labs, or Google Colab Pro for running actual benchmarks.

### Backend

```bash
git clone https://github.com/bereketlemma/llm-inference-bench.git
cd llm-inference-bench
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Frontend Dashboard

```bash
cd frontend
npm install
npm run dev         # http://localhost:3000
```

The dashboard reads benchmark data from `frontend/src/lib/benchmark-data.ts` as a fallback and from the `/api/benchmark-data` route when deployed.

---

## Usage

### Quick Validation (GPU, small model)

Runs `facebook/opt-125m` in FP16 with minimal batch/token configs — fast sanity check:

```bash
python main.py
# or using the shell helper:
bash scripts/run_benchmark.sh dev
```

### Production Benchmarks (GPU, Mistral-7B)

Edit `main.py` to enable the production config:

```python
config = BenchmarkConfig.production_config()
```

```bash
python main.py
# or using the shell helper:
bash scripts/run_benchmark.sh prod
```

The `run_benchmark.sh` script handles venv creation, dependency installation, and config switching automatically — useful when SSH-ing into a fresh RunPod or Lambda instance.

Output files are saved to `results/` with a timestamp:

- `results/benchmark_<timestamp>.json` — raw metrics per configuration
- `results/benchmark_<timestamp>.csv` — same data in tabular form
- `results/charts/` — latency, throughput, and batch-scaling charts (PNG)

### YAML Config

You can also drive the runner from a YAML file:

```python
config = BenchmarkConfig.from_yaml("my_config.yaml")
```

```yaml
batch_sizes: [1, 4, 8, 16]
output_token_lengths: [128, 256, 512]
num_runs: 10
warmup_runs: 3
quantization_formats:
  fp16:
    model: mistralai/Mistral-7B-v0.1
    quantization: null
  int4-awq:
    model: TheBloke/Mistral-7B-v0.1-AWQ
    quantization: awq
```

---

## Configuration

`BenchmarkConfig` is a dataclass with three built-in presets:

| Preset | Model | Batch sizes | Token lengths | Runs |
|---|---|---|---|---|
| `development_config()` | `facebook/opt-125m` | [1, 2] | [32, 64] | 3 |
| `production_config()` | `mistralai/Mistral-7B-v0.1` + AWQ | [1, 4, 8, 16] | [128, 256, 512] | 10 |
| Default | Mistral-7B FP16 + AWQ | [1, 4, 8, 16] | [128, 256, 512] | 10 |

---

## Frontend Dashboard

The `frontend/` directory contains a Next.js 14 dashboard with:

- **Latency charts** — p50/p90/p99 bar charts per quantization format
- **Throughput charts** — tokens/sec across batch sizes
- **Batch-size scaling** — FP16 vs INT4-AWQ throughput curve
- **Stat cards** — animated speedup, peak throughput, P99 reduction
- **PDF export** — one-click report download via jsPDF
- **Run history** — compare multiple benchmark runs side-by-side
- **Hacker terminal UI** — JetBrains Mono, green-on-black aesthetic

Stack: Next.js 14, TypeScript, Recharts, TailwindCSS, lucide-react, jsPDF

```bash
cd frontend
npm run build   # static export → out/
npm run start
```

### Deployment

The dashboard is deployed at **[bench.bereketlemma.com](https://bench.bereketlemma.com/)** via:

- **Vercel** — connected to the `frontend/` directory of this repo; auto-deploys on every push to `main`
- **Cloudflare** — custom domain `bereketlemma.com` is managed through Cloudflare DNS; a `CNAME` record points `bench` to the Vercel deployment URL

To deploy your own fork:

1. Import the repo into [Vercel](https://vercel.com) and set the **Root Directory** to `frontend`
2. Add your custom domain in the Vercel project settings
3. In Cloudflare DNS, add a `CNAME` record: `bench` → `cname.vercel-dns.com` (proxy off / DNS-only)

---

## Methodology

| Setting | Value |
|---|---|
| Warmup runs | 3 (before measurement, eliminates cold-start bias) |
| Measurement runs | 10 per configuration |
| Decoding | Greedy (temperature=0.0) for reproducibility |
| Quantization | Pre-quantized AWQ checkpoints (not runtime quantization) |
| Kernel | AWQ-Marlin (fused dequant + GEMM, faster than standard AWQ) |
| Metrics | p50, p90, p99 latency · tokens/sec · requests/sec |

---

## Docker

```bash
# Build (requires NVIDIA CUDA driver on host)
docker build -t llm-inference-bench .

# Run with GPU passthrough
docker run --gpus all --rm llm-inference-bench
```

The image is based on `nvidia/cuda:12.1.1-runtime-ubuntu22.04` with Python 3.10.

---

## Testing

```bash
pytest tests/              # runs 12 unit tests (no GPU required)
```

Tests cover:
- `calculate_metrics` — latency conversion, throughput calculation, percentile ordering
- `compute_speedup` — throughput and latency reduction ratios
- Edge cases for batch size, run counts, and output token tracking

CI does not install vLLM (GPU dependency). Tests run on every push via GitHub Actions.

---

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push:
1. `pytest tests/` — unit tests
2. `flake8` — lint

---

## Results Data

Raw CSV files are in `results/Data/`:
- `fp16_results.csv` — 9 configurations (batch × tokens), 10 runs each
- `awq_marlin_results.csv` — 9 configurations with full p50/p90/p99 metrics

These files are loaded by the Next.js API route (`frontend/src/app/api/benchmark-data/route.ts`) and served to the dashboard.

---

## License

MIT — see [LICENSE](LICENSE).

## Models Tested

- `facebook/opt-125m` — development validation
- `mistralai/Mistral-7B-v0.1` — production benchmarks (FP16)
- `TheBloke/Mistral-7B-v0.1-AWQ` — production benchmarks (INT4-AWQ)

## Troubleshooting

### vLLM import/install issues
- vLLM requires CUDA and an NVIDIA GPU for real benchmark execution.
- For non-GPU development, focus on tests and non-vLLM modules.

### Docker build is slow
- First build can be long due to large ML dependencies.
- Use: `docker build --progress=plain -t llm-inference-bench .` to see detailed progress.

### Container run without GPU is slow or hangs
- Running full `main.py` without GPU is not a meaningful benchmark path.
- Use containerized tests instead: `docker run --rm llm-inference-bench pytest tests/ -v`.

## License

MIT License. See the LICENSE file for full text.
