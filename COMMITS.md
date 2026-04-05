# Commit History - LLM Inference Benchmark

Complete list of all commits pushed to GitHub repository `bereketlemma/llm-inference-bench`.

## All 18 Commits (Oldest to Newest)

1. **f9cdaab** - `add: project dependencies (vllm, torch, transformers, pandas, matplotlib, pytest)`
   - Added requirements.txt with all project dependencies

2. **f0bef79** - `add: benchmark package initialization`
   - Created benchmark/__init__.py

3. **62a80ae** - `feat: benchmark configuration module with FP16 and AWQ-INT4 quantization presets`
   - Added benchmark/config.py with BenchmarkConfig dataclass

4. **c462437** - `feat: metrics calculation module (p50/p90/p99 latency, throughput, speedup)`
   - Added benchmark/metrics.py with metrics calculations

5. **e7c669c** - `feat: vLLM benchmark runner with model loading and inference orchestration`
   - Added benchmark/runner.py with vLLM integration

6. **ab8508c** - `add: visualization package initialization`
   - Created visualize/__init__.py

7. **93e5159** - `feat: visualization module with latency, throughput, and scaling charts`
   - Added visualize/plot_results.py with Matplotlib charts

8. **a4c8aa0** - `add: tests package initialization`
   - Created tests/__init__.py

9. **d92e081** - `test: comprehensive unit tests for metrics calculations (12 passing tests)`
   - Added tests/test_metrics.py with full test coverage

10. **5de7ece** - `feat: main entry point with benchmark orchestration, result saving, and visualization`
    - Added main.py with complete orchestration

11. **15bbfd7** - `chore: Dockerfile with NVIDIA CUDA, Python 3.10, and GPU support`
    - Added Dockerfile for containerization

12. **00a8fe7** - `chore: .dockerignore for efficient Docker builds`
    - Added .dockerignore

13. **a186c8a** - `ci: GitHub Actions workflow for pytest and flake8 linting on every push`
    - Added .github/workflows/ci.yml

14. **cd1d7a3** - `docs: comprehensive README with benchmark metrics, setup, and usage instructions`
    - Added README.md with complete documentation

15. **490d5d0** - `license: MIT License`
    - Added LICENSE file

16. **4efef0d** - `chore: .gitignore to exclude venv, cache, results, and local guide file`
    - Added .gitignore

17. **1dcf741** - `feat: shell script helper for running benchmarks on GPU pods`
    - Added scripts/run_benchmark.sh

18. **a13c8e4** - `chore: add .gitkeep files to preserve results directory structure`
    - Added results/.gitkeep and results/charts/.gitkeep

## Summary

- **Total Commits**: 18
- **Total Files**: 16 source files + 2 .gitkeep files
- **Repository**: https://github.com/bereketlemma/llm-inference-bench.git
- **Branch**: main
- **Status**: All commits pushed and synced with origin/main

## Commit Message Convention Used

- `feat:` - New feature or module
- `test:` - Test files and test coverage
- `docs:` - Documentation
- `chore:` - Infrastructure, configuration, tooling
- `ci:` - CI/CD pipeline
- `add:` - Package initialization and basic file creation
- `license:` - License files
