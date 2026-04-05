"""
Main entry point for llm-inference-bench.
Run this file to execute a full benchmark suite.
"""

import json
import os
import logging
import pandas as pd
from datetime import datetime

from benchmark.config import BenchmarkConfig
from benchmark.runner import run_benchmark_suite
from benchmark.metrics import compute_speedup
from visualize.plot_results import (
    plot_latency_comparison,
    plot_throughput_comparison,
    plot_batch_size_scaling,
    generate_summary_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting LLM Inference Benchmark Suite")
    logger.info("=" * 60)

    # Choose config based on environment
    # Use development_config() for quick validation with opt-125m
    # Use production_config() for full benchmarks with Mistral-7B
    config = BenchmarkConfig.development_config()

    logger.info(f"Quantization formats: {list(config.quantization_formats.keys())}")
    logger.info(f"Batch sizes: {config.batch_sizes}")
    logger.info(f"Output token lengths: {config.output_token_lengths}")

    # Run benchmarks
    results = run_benchmark_suite(config)

    if not results:
        logger.error("No results collected. Check your configuration.")
        return

    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)

    # Save raw results
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"results/benchmark_{timestamp}.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Raw results saved to: {results_path}")

    # Save CSV
    csv_path = f"results/benchmark_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV saved to: {csv_path}")

    # Generate visualizations
    logger.info("Generating charts...")
    plot_latency_comparison(df)
    plot_throughput_comparison(df)
    plot_batch_size_scaling(df)

    # Print summary table
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    summary = generate_summary_table(df)
    print(summary.to_string())

    # Compute speedup if multiple quantization formats were tested
    quant_formats = df["quantization"].unique()
    if "fp16" in quant_formats and "int4-awq" in quant_formats:
        # Aggregate across all batch/token configs for a fair comparison
        fp16_agg = df[df["quantization"] == "fp16"][
            ["tokens_per_second", "p99_latency_ms", "p50_latency_ms"]
        ].mean().to_dict()
        int4_agg = df[df["quantization"] == "int4-awq"][
            ["tokens_per_second", "p99_latency_ms", "p50_latency_ms"]
        ].mean().to_dict()
        speedup = compute_speedup(fp16_agg, int4_agg)

        print("\n" + "=" * 60)
        print("INT4-AWQ vs FP16 SPEEDUP")
        print("=" * 60)
        print(f"Throughput speedup:      {speedup['throughput_speedup']:.2f}x")
        print(f"P99 latency reduction:   {speedup['p99_latency_reduction']*100:.1f}%")
        print(f"P50 latency reduction:   {speedup['p50_latency_reduction']*100:.1f}%")

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
```

---

## Step 8 — Write Unit Tests

Open `tests/test_metrics.py` and add:

```python
"""
Unit tests for metrics calculation module.
Run with: pytest tests/
"""

import pytest
import numpy as np
from benchmark.metrics import calculate_metrics, compute_speedup


class TestCalculateMetrics:

    def test_basic_metrics_shape(self):
        """Test that calculate_metrics returns expected keys."""
        latencies = [0.1, 0.12, 0.11, 0.09, 0.13]
        token_counts = [100, 102, 98, 101, 99]

        result = calculate_metrics(
            latencies=latencies,
            token_counts=token_counts,
            batch_size=1,
            output_tokens=100,
        )

        required_keys = [
            "p50_latency_ms", "p90_latency_ms", "p99_latency_ms",
            "mean_latency_ms", "std_latency_ms", "min_latency_ms", "max_latency_ms",
            "tokens_per_second", "tokens_per_second_min", "requests_per_second",
            "batch_size", "output_tokens", "num_runs", "mean_tokens_per_run",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_latency_conversion_to_ms(self):
        """Test that latencies are correctly converted from seconds to ms."""
        latencies = [1.0, 1.0, 1.0]  # 1 second each
        token_counts = [100, 100, 100]

        result = calculate_metrics(
            latencies=latencies,
            token_counts=token_counts,
            batch_size=1,
            output_tokens=100,
        )

        assert abs(result["p50_latency_ms"] - 1000.0) < 1e-6
        assert abs(result["mean_latency_ms"] - 1000.0) < 1e-6

    def test_throughput_calculation(self):
        """Test tokens per second calculation."""
        latencies = [1.0, 1.0, 1.0]  # 1 second each
        token_counts = [100, 100, 100]  # 100 tokens each run

        result = calculate_metrics(
            latencies=latencies,
            token_counts=token_counts,
            batch_size=1,
            output_tokens=100,
        )

        assert abs(result["tokens_per_second"] - 100.0) < 1e-6

    def test_p99_higher_than_p50(self):
        """P99 latency should always be >= P50 latency."""
        latencies = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0]
        token_counts = [100] * 10

        result = calculate_metrics(
            latencies=latencies,
            token_counts=token_counts,
            batch_size=1,
            output_tokens=100,
        )

        assert result["p99_latency_ms"] >= result["p50_latency_ms"]

    def test_min_throughput_lower_than_mean(self):
        """Minimum throughput should be <= mean throughput."""
        latencies = [0.1, 0.2, 0.3, 0.4, 0.5]
        token_counts = [100, 100, 100, 100, 100]

        result = calculate_metrics(
            latencies=latencies,
            token_counts=token_counts,
            batch_size=1,
            output_tokens=100,
        )

        assert result["tokens_per_second_min"] <= result["tokens_per_second"]

    def test_batch_size_stored_correctly(self):
        """Batch size should be stored in results."""
        latencies = [0.1, 0.1, 0.1]
        token_counts = [100, 100, 100]

        result = calculate_metrics(
            latencies=latencies,
            token_counts=token_counts,
            batch_size=8,
            output_tokens=128,
        )

        assert result["batch_size"] == 8
        assert result["output_tokens"] == 128

    def test_requests_per_second(self):
        """Requests per second = batch_size / mean_latency."""
        latencies = [2.0, 2.0, 2.0]  # 2 seconds each
        token_counts = [100, 100, 100]

        result = calculate_metrics(
            latencies=latencies,
            token_counts=token_counts,
            batch_size=4,
            output_tokens=100,
        )

        # 4 requests / 2 seconds = 2 req/s
        assert abs(result["requests_per_second"] - 2.0) < 1e-6


class TestComputeSpeedup:

    def test_throughput_speedup(self):
        """Test throughput speedup calculation."""
        baseline = {"tokens_per_second": 100.0, "p99_latency_ms": 200.0, "p50_latency_ms": 100.0}
        optimized = {"tokens_per_second": 200.0, "p99_latency_ms": 100.0, "p50_latency_ms": 50.0}

        speedup = compute_speedup(baseline, optimized)

        assert abs(speedup["throughput_speedup"] - 2.0) < 1e-6
        assert abs(speedup["p99_latency_reduction"] - 0.5) < 1e-6
        assert abs(speedup["p50_latency_reduction"] - 0.5) < 1e-6

    def test_no_speedup_identical_configs(self):
        """Identical configs should produce 1.0x speedup."""
        metrics = {"tokens_per_second": 100.0, "p99_latency_ms": 200.0, "p50_latency_ms": 100.0}

        speedup = compute_speedup(metrics, metrics)

        assert abs(speedup["throughput_speedup"] - 1.0) < 1e-6
        assert abs(speedup["p99_latency_reduction"] - 0.0) < 1e-6

    def test_regression_detected(self):
        """If optimized is slower, speedup should be < 1.0."""
        baseline = {"tokens_per_second": 200.0, "p99_latency_ms": 100.0, "p50_latency_ms": 50.0}
        optimized = {"tokens_per_second": 100.0, "p99_latency_ms": 200.0, "p50_latency_ms": 100.0}

        speedup = compute_speedup(baseline, optimized)

        assert speedup["throughput_speedup"] < 1.0
        assert speedup["p99_latency_reduction"] < 0.0