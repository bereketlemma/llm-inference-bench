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
        print(f"P99 latency reduction:   {speedup['p99_latency_reduction'] * 100:.1f}%")
        print(f"P50 latency reduction:   {speedup['p50_latency_reduction'] * 100:.1f}%")

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()