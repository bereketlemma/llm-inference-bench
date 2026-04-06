# llm-inference-bench

Benchmarking framework for LLM inference using [vLLM](https://github.com/vllm-project/vllm).

I built this to measure how much faster INT4 quantization actually is compared to full-precision FP16 on real hardware, and to have a structured way to test different configurations (batch size, sequence length, quantization format).

**Dashboard:** [bench.bereketlemma.com](https://bench.bereketlemma.com/)

---

## What I found

I ran Mistral-7B on an NVIDIA L4 (24GB) through 18 different configurations varying batch size, sequence length, and quantization format. Each config got 10 measurement runs after 3 warmup iterations, all with greedy decoding for reproducibility.

**INT4 AWQ-Marlin is 3.35x faster than FP16 on average.** Peak throughput hit 452 tok/s at batch=8.

### FP16 baseline

| Batch | Tokens | P50 (ms) | P99 (ms) | Tok/s | Req/s |
|------:|-------:|---------:|---------:|------:|------:|
| 1 | 128 | 3,587 | 3,590 | 17.9 | 1.00 |
| 1 | 256 | 7,170 | 7,176 | 17.9 | 0.50 |
| 1 | 512 | 14,353 | 14,355 | 17.8 | 0.25 |
| 4 | 128 | 1,872 | 1,880 | 68.5 | 2.14 |
| 4 | 256 | 3,740 | 3,760 | 68.3 | 1.07 |
| 4 | 512 | 7,520 | 7,540 | 68.0 | 0.53 |
| 8 | 128 | 1,920 | 1,940 | 133.9 | 4.17 |
| 8 | 256 | 3,840 | 3,860 | 133.5 | 2.08 |
| 8 | 512 | 30,590 | 30,600 | 133.9 | 0.26 |

### INT4 AWQ-Marlin

| Batch | Tokens | P50 (ms) | P99 (ms) | Tok/s | Req/s |
|------:|-------:|---------:|---------:|------:|------:|
| 1 | 128 | 2,084 | 2,087 | 61.4 | 0.48 |
| 1 | 256 | 4,201 | 4,205 | 60.9 | 0.24 |
| 1 | 512 | 8,461 | 8,464 | 60.5 | 0.12 |
| 4 | 128 | 2,180 | 2,181 | 234.9 | 1.83 |
| 4 | 256 | 4,394 | 4,395 | 233.1 | 0.91 |
| 4 | 512 | 8,990 | 8,991 | 227.8 | 0.44 |
| 8 | 128 | 2,264 | 2,266 | **452.3** | 3.53 |
| 8 | 256 | 4,590 | 4,590 | 446.2 | 1.74 |
| 8 | 512 | 9,545 | 9,548 | 429.1 | 0.84 |

### The takeaway

| | |
|---|---|
| Throughput speedup (avg) | **3.35x** |
| P99 latency reduction (avg) | **37.5%** |
| Peak throughput (INT4, BS=8) | **452.3 tok/s** |
| Peak throughput (FP16, BS=8) | 133.9 tok/s |

The Marlin kernel makes a real difference here. I initially tried the standard AWQ kernel and it was actually *slower* than FP16 — turns out vLLM falls back to a naive dequantization path unless you explicitly use `awq_marlin`. That's the kind of thing you only learn by running the benchmarks yourself.

---

## How it works

The Python backend loads models through vLLM, runs them through a matrix of configs, and collects latency/throughput metrics. The Next.js frontend turns those numbers into something a human can actually understand.

```
llm-inference-bench/
├── benchmark/
│   ├── config.py          # dev/production presets, YAML loading
│   ├── metrics.py         # percentile math, speedup calculations
│   └── runner.py          # vLLM model loading + benchmark loop
├── visualize/
│   └── plot_results.py    # matplotlib charts
├── frontend/              # Next.js dashboard (bench.bereketlemma.com)
│   ├── src/app/           # pages + API route
│   ├── src/components/    # Dashboard with Recharts
│   └── src/lib/           # typed benchmark data
├── results/Data/          # raw CSV files from actual runs
├── tests/                 # 12 unit tests, no GPU needed
├── Dockerfile             # NVIDIA CUDA 12.1 + Python 3.10
├── main.py                # entry point
└── requirements.txt
```

---

## Running it yourself

### You need

- Python 3.10+
- An NVIDIA GPU with CUDA (vLLM doesn't do CPU)
- 24GB VRAM for FP16 Mistral-7B, ~16GB for AWQ variant

If you don't have a local GPU, I used a Google Cloud L4 instance ($0.87/hr)  and the whole run cost about 3 dollars.

If you are a student like me you can get $300 in free credits [here](https://cloud.google.com/free). which can be used to run benchmarks on different hardware.

### Backend setup

```bash
git clone https://github.com/bereketlemma/llm-inference-bench.git
cd llm-inference-bench
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Quick test (GPU, small model)

```bash
python main.py
```

This runs OPT-125M with minimal configs — takes about a minute. Good for making sure everything works.

### Full benchmarks

Switch to production config in `main.py`:

```python
config = BenchmarkConfig.production_config()
```

Then run it. Takes 1-2 hours depending on GPU. The script saves results to `results/` with timestamps.

You can also use the shell helper on cloud instances:

```bash
bash scripts/run_benchmark.sh prod
```

### Frontend

```bash
cd frontend
npm install
npm run dev   
```

---

## The dashboard

[bench.bereketlemma.com](https://bench.bereketlemma.com/) has tabs for latency breakdown, throughput by sequence length, batch scaling curves, a head-to-head comparison tool, and a raw data explorer where you can filter by batch size and token count. There's also a PDF export if someone wants a snapshot.

Built with Next.js 14, TypeScript, Recharts, and Tailwind. The terminal aesthetic matches [my portfolio](https://bereketlemma.com).

Deployed on Vercel with auto-deploy on push. Domain managed through Cloudflare.

---

## How I ran the benchmarks

| | |
|---|---|
| GPU | NVIDIA L4 24GB (Google Cloud, us-west1-a) |
| Engine | vLLM 0.16.0 |
| Warmup | 3 iterations before measurement |
| Measurement | 10 runs per config |
| Decoding | Greedy (temp=0.0) |
| Quantization | Pre-quantized AWQ checkpoint + Marlin kernel |
| FP16 model | `mistralai/Mistral-7B-v0.1` |
| INT4 model | `TheBloke/Mistral-7B-v0.1-AWQ` |

I used the Marlin fused dequant+GEMM kernel for INT4. It's significantly faster than the default AWQ path in vLLM. 

You have to specify `quantization="awq_marlin"` explicitly, otherwise vLLM uses the slower implementation even though it logs a message saying Marlin is available.

---

## Docker

```bash
docker build -t llm-inference-bench .
docker run --gpus all --rm llm-inference-bench
```

Needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host.

Pre-configured on RunPod, Lambda, and GCP Deep Learning VMs.

---

## Tests

```bash
pytest tests/ -v
```

12 tests covering the metrics module latency conversion, throughput calculation, percentile ordering, speedup ratios, edge cases.

 No GPU needed. Runs in CI on every push.

---

## Config presets

| Preset | Model | Batch sizes | Tokens | Runs |
|---|---|---|---|---|
| `development_config()` | OPT-125M | 1, 2 | 32, 64 | 3 |
| `production_config()` | Mistral-7B + AWQ | 1, 4, 8 | 128, 256, 512 | 10 |

Or load from YAML:

```python
config = BenchmarkConfig.from_yaml("my_config.yaml")
```

---

## Raw data

CSVs are in `results/Data/`. The AWQ file has full p50/p90/p99 with sub-millisecond precision from 10 runs per config. 

The FP16 file has the same structure. Both get served to the dashboard through a Next.js API route.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
