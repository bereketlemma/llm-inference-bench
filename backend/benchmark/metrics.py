import numpy as np
from typing import List


def calculate_metrics(
    latencies: List[float],
    token_counts: List[int],
    batch_size: int,
    output_tokens: int,
) -> dict:
    """
    Calculate comprehensive inference metrics from raw measurements.

    Args:
        latencies: List of end-to-end latencies in seconds
        token_counts: List of total tokens generated per run
        batch_size: Number of concurrent requests
        output_tokens: Target output tokens per request

    Returns:
        Dictionary containing all computed metrics
    """
    latencies_ms = np.array(latencies) * 1000  # convert to milliseconds
    token_counts_arr = np.array(token_counts)
    latencies_arr = np.array(latencies)

    # Tokens per second = total tokens generated / elapsed time
    tokens_per_second_per_run = token_counts_arr / latencies_arr

    return {
        # Latency metrics
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p90_latency_ms": float(np.percentile(latencies_ms, 90)),
        "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "std_latency_ms": float(np.std(latencies_ms)),
        "min_latency_ms": float(np.min(latencies_ms)),
        "max_latency_ms": float(np.max(latencies_ms)),

        # Throughput metrics
        "tokens_per_second": float(np.mean(tokens_per_second_per_run)),
        "tokens_per_second_min": float(np.min(tokens_per_second_per_run)),
        "requests_per_second": float(batch_size / np.mean(latencies_arr)),

        # Configuration info
        "batch_size": batch_size,
        "output_tokens": output_tokens,
        "num_runs": len(latencies),

        # Efficiency
        "mean_tokens_per_run": float(np.mean(token_counts_arr)),
    }


def compute_speedup(baseline_metrics: dict, optimized_metrics: dict) -> dict:
    """
    Compute speedup ratios between two configurations.
    Typically used to compare FP16 (baseline) vs quantized (optimized).

    Args:
        baseline_metrics: Metrics from baseline configuration (e.g., FP16)
        optimized_metrics: Metrics from optimized configuration (e.g., INT4)

    Returns:
        Dictionary of speedup ratios
    """
    return {
        "throughput_speedup": (
            optimized_metrics["tokens_per_second"]
            / baseline_metrics["tokens_per_second"]
        ),
        "p99_latency_reduction": (
            1
            - optimized_metrics["p99_latency_ms"]
            / baseline_metrics["p99_latency_ms"]
        ),
        "p50_latency_reduction": (
            1
            - optimized_metrics["p50_latency_ms"]
            / baseline_metrics["p50_latency_ms"]
        ),
    }