"""
Visualization module for llm-inference-bench.
Generates latency, throughput, and batch-size scaling charts from benchmark results.
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logger = logging.getLogger(__name__)

CHART_DIR = "results/charts"
COLORS = {"fp16": "#4C72B0", "int4-awq": "#DD8452"}


def _ensure_chart_dir():
    os.makedirs(CHART_DIR, exist_ok=True)


def plot_latency_comparison(df: pd.DataFrame) -> None:
    """
    Bar chart comparing P50 and P99 latency across quantization formats,
    averaged across all batch sizes and token lengths.
    """
    _ensure_chart_dir()

    summary = (
        df.groupby("quantization")[["p50_latency_ms", "p99_latency_ms"]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(summary))
    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        summary["p50_latency_ms"],
        width,
        label="P50 Latency",
        color=[COLORS.get(q, "#888888") for q in summary["quantization"]],
        alpha=0.9,
    )
    ax.bar(
        [i + width / 2 for i in x],
        summary["p99_latency_ms"],
        width,
        label="P99 Latency",
        color=[COLORS.get(q, "#888888") for q in summary["quantization"]],
        alpha=0.5,
        hatch="//",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["quantization"], fontsize=11)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Comparison -- FP16 vs INT4-AWQ")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(CHART_DIR, "latency_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_throughput_comparison(df: pd.DataFrame) -> None:
    """
    Bar chart comparing average tokens/sec per quantization format.
    """
    _ensure_chart_dir()

    summary = (
        df.groupby("quantization")["tokens_per_second"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        summary["quantization"],
        summary["tokens_per_second"],
        color=[COLORS.get(q, "#888888") for q in summary["quantization"]],
        alpha=0.9,
        width=0.5,
    )

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{bar.get_height():,.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Tokens / second")
    ax.set_title("Throughput Comparison -- FP16 vs INT4-AWQ")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(CHART_DIR, "throughput_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_batch_size_scaling(df: pd.DataFrame) -> None:
    """
    Line chart showing tokens/sec vs batch size for each quantization format.
    """
    _ensure_chart_dir()

    scaling = (
        df.groupby(["quantization", "batch_size"])["tokens_per_second"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for quant, group in scaling.groupby("quantization"):
        group = group.sort_values("batch_size")
        ax.plot(
            group["batch_size"],
            group["tokens_per_second"],
            marker="o",
            label=quant,
            color=COLORS.get(quant, "#888888"),
            linewidth=2,
        )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Tokens / second")
    ax.set_title("Throughput Scaling by Batch Size")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(CHART_DIR, "batch_size_scaling.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame summarising key metrics grouped by quantization format
    and batch size, suitable for printing to stdout.
    """
    summary = (
        df.groupby(["quantization", "batch_size"])
        .agg(
            p50_ms=("p50_latency_ms", "mean"),
            p99_ms=("p99_latency_ms", "mean"),
            throughput=("tokens_per_second", "mean"),
            rps=("requests_per_second", "mean"),
        )
        .round(1)
        .reset_index()
    )
    return summary
