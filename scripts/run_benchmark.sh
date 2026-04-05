#!/usr/bin/env bash
# run_benchmark.sh — helper for running benchmarks on a GPU pod (RunPod, Lambda, etc.)
# Usage: bash scripts/run_benchmark.sh [dev|prod]
set -e

MODE=${1:-dev}

echo "========================================"
echo " llm-inference-bench"
echo " Mode: $MODE"
echo "========================================"

# Ensure venv exists
if [ ! -d "venv" ]; then
  echo "[setup] Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate
echo "[setup] Installing dependencies..."
pip install -q -r requirements.txt

# Switch config in main.py based on mode
if [ "$MODE" = "prod" ]; then
  echo "[bench] Running PRODUCTION benchmark (Mistral-7B FP16 + AWQ-INT4)..."
  # Temporarily patch main.py to use production_config
  sed -i 's/BenchmarkConfig.development_config()/BenchmarkConfig.production_config()/' main.py
  python main.py
  # Restore development_config
  sed -i 's/BenchmarkConfig.production_config()/BenchmarkConfig.development_config()/' main.py
else
  echo "[bench] Running DEVELOPMENT benchmark (OPT-125M, fast sanity check)..."
  python main.py
fi

echo "[done] Results saved to results/"
