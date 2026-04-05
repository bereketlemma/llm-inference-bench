"""
Configuration management for benchmark runs.
Supports YAML config files and programmatic configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
import yaml


@dataclass
class BenchmarkConfig:
    """
    Configuration for a benchmark run.

    quantization_formats maps a display label to a dict containing:
      - "model": the HuggingFace model name (must be a pre-quantized checkpoint for quantized formats)
      - "quantization": the vLLM quantization argument (None for FP16, "awq" for AWQ-INT4)
    """

    # Quantization formats to test
    # Each entry maps label -> {"model": ..., "quantization": ...}
    quantization_formats: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "fp16": {
            "model": "mistralai/Mistral-7B-v0.1",
            "quantization": None,
        },
        "int4-awq": {
            "model": "TheBloke/Mistral-7B-v0.1-AWQ",
            "quantization": "awq",
        },
    })

    # Test configurations
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16])
    output_token_lengths: List[int] = field(default_factory=lambda: [128, 256, 512])

    # Measurement settings
    num_runs: int = 10
    warmup_runs: int = 3

    # Output settings
    results_dir: str = "results"
    save_charts: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        """Load config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def development_config(cls) -> "BenchmarkConfig":
        """
        Lightweight config for quick validation on GPU.
        Uses a small model (OPT-125M) with FP16 only and minimal runs.
        Still requires a GPU — vLLM does not support CPU inference.
        """
        return cls(
            quantization_formats={
                "fp16": {
                    "model": "facebook/opt-125m",
                    "quantization": None,
                },
            },
            batch_sizes=[1, 2],
            output_token_lengths=[32, 64],
            num_runs=3,
            warmup_runs=1,
        )

    @classmethod
    def production_config(cls) -> "BenchmarkConfig":
        """
        Full benchmark config for GPU environments.
        Uses Mistral-7B in FP16 and its AWQ-INT4 variant.
        Requires an A10 (24GB) or better GPU.
        """
        return cls(
            quantization_formats={
                "fp16": {
                    "model": "mistralai/Mistral-7B-v0.1",
                    "quantization": None,
                },
                "int4-awq": {
                    "model": "TheBloke/Mistral-7B-v0.1-AWQ",
                    "quantization": "awq",
                },
            },
            batch_sizes=[1, 4, 8, 16],
            output_token_lengths=[128, 256, 512],
            num_runs=10,
            warmup_runs=3,
        )