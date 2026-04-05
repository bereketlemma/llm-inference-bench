"""
Core benchmark runner for LLM inference performance measurement.
Supports FP16 and AWQ (INT4) quantization formats via vLLM.
"""

import time
import logging
from typing import Optional, List

import torch
from vllm import LLM, SamplingParams

from .metrics import calculate_metrics
from .config import BenchmarkConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_name: str, quantization: Optional[str] = None) -> LLM:
    """
    Load an LLM using vLLM with optional quantization.

    Args:
        model_name: HuggingFace model name or path.
                     For quantized models, use the pre-quantized checkpoint
                     (e.g., "TheBloke/Mistral-7B-v0.1-AWQ" for AWQ).
        quantization: One of None (fp16) or "awq" (int4).
                      Must match the model checkpoint format.

    Returns:
        Loaded LLM instance
    """
    logger.info(f"Loading model: {model_name} with quantization: {quantization or 'fp16'}")

    kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        "max_model_len": 2048,
    }

    if quantization:
        kwargs["quantization"] = quantization
    else:
        kwargs["dtype"] = "float16"

    llm = LLM(**kwargs)

    logger.info("Model loaded successfully")
    return llm


def run_single_benchmark(
    llm: LLM,
    batch_size: int,
    output_tokens: int,
    num_runs: int = 10,
    warmup_runs: int = 3,
) -> dict:
    """
    Run a single benchmark configuration and collect metrics.

    Args:
        llm: Loaded vLLM model
        batch_size: Number of concurrent requests
        output_tokens: Number of tokens to generate per request
        num_runs: Number of measurement runs
        warmup_runs: Number of warmup runs before measurement

    Returns:
        Dictionary of metrics for this configuration
    """
    prompts = [
        "Explain the transformer attention mechanism in detail, "
        "covering self-attention, multi-head attention, and positional encoding."
    ] * batch_size

    sampling_params = SamplingParams(
        max_tokens=output_tokens,
        temperature=0.0,  # greedy decoding for reproducibility
    )

    # Warmup runs to avoid cold start bias
    logger.info(f"Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        llm.generate(prompts, sampling_params)

    # Measurement runs
    logger.info(f"Running {num_runs} measurement iterations...")
    latencies: List[float] = []
    token_counts: List[int] = []

    for run_idx in range(num_runs):
        start_time = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        latencies.append(elapsed)

        total_tokens = sum(
            len(output.outputs[0].token_ids)
            for output in outputs
        )
        token_counts.append(total_tokens)

        logger.debug(f"Run {run_idx + 1}: {elapsed:.3f}s, {total_tokens} tokens")

    return calculate_metrics(
        latencies=latencies,
        token_counts=token_counts,
        batch_size=batch_size,
        output_tokens=output_tokens,
    )


def run_benchmark_suite(config: BenchmarkConfig) -> List[dict]:
    """
    Run the full benchmark suite across all configurations.

    Args:
        config: BenchmarkConfig instance with all settings

    Returns:
        List of result dictionaries for all configurations
    """
    all_results = []

    for quantization_label, model_and_quant in config.quantization_formats.items():
        model_name = model_and_quant["model"]
        quant_arg = model_and_quant["quantization"]

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Benchmarking quantization: {quantization_label}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'=' * 50}")

        try:
            llm = load_model(model_name, quant_arg)

            for batch_size in config.batch_sizes:
                for output_tokens in config.output_token_lengths:
                    logger.info(
                        f"Config: batch_size={batch_size}, "
                        f"output_tokens={output_tokens}"
                    )

                    result = run_single_benchmark(
                        llm=llm,
                        batch_size=batch_size,
                        output_tokens=output_tokens,
                        num_runs=config.num_runs,
                        warmup_runs=config.warmup_runs,
                    )

                    result["quantization"] = quantization_label
                    result["model"] = model_name
                    all_results.append(result)

                    logger.info(
                        f"Result: p50={result['p50_latency_ms']:.1f}ms, "
                        f"p99={result['p99_latency_ms']:.1f}ms, "
                        f"throughput={result['tokens_per_second']:.1f} tok/s"
                    )

            # Free GPU memory between quantization formats
            del llm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to benchmark {quantization_label}: {e}")
            continue

    return all_results