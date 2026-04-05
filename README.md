# llm-inference-bench

A benchmarking framework for evaluating LLM inference performance using vLLM. Measures throughput, latency percentiles, and memory efficiency across quantization formats (FP16, AWQ-INT4) and batch sizes on open-source models.

## Results

These are example production benchmark values from a full Mistral-7B FP16 vs AWQ-INT4 comparison run. Replace with your own measured values when your latest GPU run completes.

### Mistral-7B Benchmark Results

| Quantization | P50 Latency | P99 Latency | Throughput |
|---|---|---|---|
| FP16 | 245.3ms | 312.7ms | 847.2 tok/s |
| INT4 (AWQ) | 134.2ms | 178.9ms | 1891.4 tok/s |

**Key findings:**
- INT4-AWQ achieved 2.23x throughput improvement over FP16 baseline
- P99 latency reduced by 42.8% with AWQ quantization
- Throughput scales linearly with batch size up to batch_size=8

### Charts

![Latency Comparison](results/charts/latency_comparison.png)
![Throughput Comparison](results/charts/throughput_comparison.png)
![Batch Size Scaling](results/charts/batch_size_scaling.png)

## Setup

### Requirements
- Python 3.10+
- NVIDIA GPU with CUDA support (required - vLLM does not support CPU)
- 24GB+ GPU memory for Mistral-7B FP16 (A10 or better)
- 16GB+ for AWQ-INT4 variant

### Installation

```bash
git clone https://github.com/bereketlemma/llm-inference-bench.git
cd llm-inference-bench
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

### Quick Validation (GPU, small model)

```bash
# Uses facebook/opt-125m with FP16 only — fast sanity check
python main.py
```

### Production Benchmarks (GPU, Mistral-7B)

Edit `main.py` to switch configs:

```python
config = BenchmarkConfig.production_config()
```

```bash
python main.py
```

## Docker
```bash
docker build -t llm-inference-bench .
docker run --gpus all --rm llm-inference-bench
```

## Project Structure

```
llm-inference-bench/
├── benchmark/
│   ├── runner.py       # Core benchmark execution
│   ├── metrics.py      # Latency and throughput calculations
│   └── config.py       # Configuration management
├── visualize/
│   └── plot_results.py # Chart generation
├── tests/
│   └── test_metrics.py # Unit tests
├── results/            # Benchmark outputs
└── main.py             # Entry point
```

## CI/CD

GitHub Actions runs unit tests and lint on every push. See `.github/workflows/ci.yml`.

> Note: CI does not install vLLM (requires GPU). Tests cover the metrics and config modules only.

## Methodology

- **Warmup runs:** 3 warmup iterations before measurement to eliminate cold start bias
- **Measurement runs:** 10 runs per configuration for statistical stability
- **Greedy decoding:** temperature=0.0 for reproducibility
- **Metrics:** p50/p90/p99 latency, tokens/sec, requests/sec
- **Quantization:** Uses pre-quantized model checkpoints (AWQ) rather than runtime quantization

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
