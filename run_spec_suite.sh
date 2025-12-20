#!/usr/bin/env bash
set -euo pipefail

# Lightweight wrapper to run VLLM speculative decoding benchmarks.
# This script is intended to be called from CI or local perf harnesses.
# Adapted from lmdeploy/run_spec_suite.sh for VLLM compatibility.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${ROOT_DIR}/results_vllm_speculative"
mkdir -p "${OUTPUT_DIR}"

# Paths to models (adjust if needed or override via env)
MODEL_PATH="${MODEL_PATH:-/workspace/aimo/models/gpt-oss-120b}"
SPEC_MODEL_PATH="${SPEC_MODEL_PATH:-/workspace/aimo/models/gpt-oss-120b-eagle3}"

echo "[run_spec_suite] Environment:"
echo "  MODEL_PATH: ${MODEL_PATH}"
echo "  SPEC_MODEL_PATH: ${SPEC_MODEL_PATH}"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}"

# Activate conda environment
echo "[run_spec_suite] Activating conda vllm environment..."
source /workspace/aimo/miniconda/etc/profile.d/conda.sh
conda activate vllm

# Check if VLLM is available
if ! command -v vllm &> /dev/null; then
    echo "Error: VLLM not found in conda vllm environment"
    echo "Please ensure VLLM is installed: cd /workspace/aimo/LM/vllm && pip install -e ."
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

# 1. Run Baseline Scenarios (8K Context)
# This tests standard generation without speculation hooks active
echo ""
echo "[run_spec_suite] Running Baseline (8K Context)..."
python3 "${ROOT_DIR}/benchmark_speculative_serve.py" \
  --model-path "${MODEL_PATH}" \
  --spec-model-path "${SPEC_MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --method baseline \
  --scenario baseline \
  --warmup-runs "${SPEC_SUITE_WARMUP_RUNS:-1}" \
  --measurement-runs "${SPEC_SUITE_MEASUREMENT_RUNS:-1}"

# 2. Run Single-Batch 32K Scenarios (Baseline + Spec)
# This tests EAGLE3 and other speculative methods with single batch
echo ""
echo "[run_spec_suite] Running Single 32K Scenarios (All Methods)..."
python3 "${ROOT_DIR}/benchmark_speculative_serve.py" \
  --model-path "${MODEL_PATH}" \
  --spec-model-path "${SPEC_MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --method all \
  --scenario single \
  --warmup-runs "${SPEC_SUITE_WARMUP_RUNS:-1}" \
  --measurement-runs "${SPEC_SUITE_MEASUREMENT_RUNS:-1}"

# 3. Run Batch Scenarios (8K Context, various methods)
echo ""
echo "[run_spec_suite] Running Batch Scenarios (All Methods)..."
python3 "${ROOT_DIR}/benchmark_speculative_serve.py" \
  --model-path "${MODEL_PATH}" \
  --spec-model-path "${SPEC_MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --method all \
  --scenario batch \
  --warmup-runs "${SPEC_SUITE_WARMUP_RUNS:-1}" \
  --measurement-runs "${SPEC_SUITE_MEASUREMENT_RUNS:-1}"

# 4. Run Large Context Scenarios (16K Context)
echo ""
echo "[run_spec_suite] Running Large Context Scenarios (All Methods)..."
python3 "${ROOT_DIR}/benchmark_speculative_serve.py" \
  --model-path "${MODEL_PATH}" \
  --spec-model-path "${SPEC_MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --method all \
  --scenario large-context \
  --warmup-runs "${SPEC_SUITE_WARMUP_RUNS:-1}" \
  --measurement-runs "${SPEC_SUITE_MEASUREMENT_RUNS:-1}"

# 5. Run Stress Test Scenarios
echo ""
echo "[run_spec_suite] Running Stress Test Scenarios (All Methods)..."
python3 "${ROOT_DIR}/benchmark_speculative_serve.py" \
  --model-path "${MODEL_PATH}" \
  --spec-model-path "${SPEC_MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --method all \
  --scenario stress \
  --warmup-runs "${SPEC_SUITE_WARMUP_RUNS:-1}" \
  --measurement-runs "${SPEC_SUITE_MEASUREMENT_RUNS:-1}"

echo ""
echo "[run_spec_suite] All benchmarks completed. Results are in ${OUTPUT_DIR}"

# Summary of results
echo ""
echo "Generated result files:"
find "${OUTPUT_DIR}" -name "*.json" -exec echo "  {}" \; | sort

echo ""
echo "To analyze results:"
echo "  ls -la ${OUTPUT_DIR}/"
echo "  cat ${OUTPUT_DIR}/*.json | jq '.scenario.name + \" (\" + .scenario.method + \"): \" + (.throughput_tokens_per_sec.mean | tostring) + \" tok/s\"'"

echo ""
echo "[run_spec_suite] Benchmark suite completed successfully!"