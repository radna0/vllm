#!/usr/bin/env bash
set -euo pipefail

# Enhanced wrapper to run VLLM speculative decoding benchmarks.
# Supports 3 modes: baseline, vllm_native, arctic_inference
# COMPATIBLE with lmdeploy evaluation framework

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${ROOT_DIR}/results_vllm_speculative_enhanced"
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

# Check if Arctic Inference is available for enhanced mode
if ! python -c "import arctic_inference" 2>/dev/null; then
    echo "Warning: Arctic Inference not available, enhanced features will be limited"
    ARCTIC_AVAILABLE=false
else
    ARCTIC_AVAILABLE=true
fi

# Set PYTHONPATH
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

echo ""
echo "[run_spec_suite] Available modes:"
echo "  1. baseline     - Standard VLLM without speculation"
echo "  2. vllm_native - VLLM built-in speculative methods (EAGLE, EAGLE3, ngram, suffix)"
echo "  3. arctic_inference - Arctic Inference with LSTM Speculator"
echo ""

# Function to run enhanced benchmark
run_enhanced_benchmark() {
    local mode=$1
    local scenario=$2
    
    echo "[run_spec_suite] Running mode: $mode"
    echo "[run_spec_suite] Scenario: $scenario"
    
    case $mode in
        "baseline")
            python3 "${ROOT_DIR}/benchmark_speculative_enhanced.py" \
                --model-path "${MODEL_PATH}" \
                --spec-model-path "${SPEC_MODEL_PATH}" \
                --output-dir "${OUTPUT_DIR}" \
                --mode baseline \
                --scenario "$scenario" \
                --warmup-runs "${SPEC_SUITE_WARMUP_RUNS:-1}" \
                --measurement-runs "${SPEC_SUITE_MEASUREMENT_RUNS:-1}"
            ;;
        "vllm_native")
            python3 "${ROOT_DIR}/benchmark_speculative_enhanced.py" \
                --model-path "${MODEL_PATH}" \
                --spec-model-path "${SPEC_MODEL_PATH}" \
                --output-dir "${OUTPUT_DIR}" \
                --mode vllm_native \
                --scenario "$scenario" \
                --method "${SPEC_SUITE_VLLM_METHOD:-eagle3}" \
                --warmup-runs "${SPEC_SUITE_WARMUP_RUNS:-1}" \
                --measurement-runs "${SPEC_SUITE_MEASUREMENT_RUNS:-1}"
            ;;
        "arctic_inference")
            if [ "$ARCTIC_AVAILABLE" = true ]; then
                python3 "${ROOT_DIR}/benchmark_speculative_enhanced.py" \
                    --model-path "${MODEL_PATH}" \
                    --spec-model-path "${SPEC_MODEL_PATH}" \
                    --output-dir "${OUTPUT_DIR}" \
                    --mode arctic_inference \
                    --scenario "$scenario" \
                    --warmup-runs "${SPEC_SUITE_WARMUP_RUNS:-1}" \
                    --measurement-runs "${SPEC_SUITE_MEASUREMENT_RUNS:-1}"
            else
                echo "Warning: Arctic Inference not available, skipping arctic_inference mode"
            ;;
        *)
            echo "Error: Unknown mode $mode"
            exit 1
            ;;
    esac
}

# Main execution based on mode argument
MODE="${1:-all}"
SCENARIO="${2:-all}"

echo "[run_spec_suite] Running with MODE: $MODE, SCENARIO: $SCENARIO"

case $MODE in
    "all")
        echo "[run_spec_suite] Running ALL MODES and ALL SCENARIOS"
        
        # Baseline mode
        echo ""
        echo "[run_spec_suite] ==================== BASELINE MODE ===================="
        for scenario in baseline single batch large-context stress; do
            run_enhanced_benchmark baseline "$scenario"
        done
        
        # VLLM Native mode
        echo ""
        echo "[run_spec_suite] ==================== VLLM NATIVE MODE ===================="
        for scenario in single batch large-context stress; do
            run_enhanced_benchmark vllm_native "$scenario"
        done
        
        # Arctic Inference mode
        if [ "$ARCTIC_AVAILABLE" = true ]; then
            echo ""
            echo "[run_spec_suite] ==================== ARCTIC INFERENCE MODE ===================="
            for scenario in single batch large-context stress; do
                run_enhanced_benchmark arctic_inference "$scenario"
            done
        fi
        ;;
    "baseline"|"vllm_native"|"arctic_inference")
        echo "[run_spec_suite] Running MODE: $MODE, ALL SCENARIOS"
        
        for scenario in single batch large-context stress; do
            run_enhanced_benchmark "$MODE" "$scenario"
        done
        ;;
    *)
        echo "Usage: $0 [all|baseline|vllm_native|arctic_inference] [all|baseline|single|batch|large-context|stress]"
        echo ""
        echo "Examples:"
        echo "  $0 all                    # Run all modes and all scenarios"
        echo "  $0 vllm_native single    # Run VLLM native speculative methods, single scenarios only"
        echo "  $0 arctic_inference batch  # Run Arctic Inference, batch scenarios only"
        exit 1
        ;;
esac

echo ""
echo "[run_spec_suite] All benchmarks completed. Results are in ${OUTPUT_DIR}"

# Summary of results
echo ""
echo "Generated result files:"
find "${OUTPUT_DIR}" -name "*.json" -exec echo "  {}" \; | sort

echo ""
echo "Results summary:"
if command -v jq &> /dev/null; then
    echo "=== THROUGHPUT COMPARISON ==="
    echo "Mode                | Throughput (tok/s) | Latency (ms/tok) | Best Scenario"
    echo "--------------------|-------------------|------------------|------------------"
    
    # Extract summary from JSON files using jq
    for mode in baseline vllm_native arctic_inference; do
        if [ -f "${OUTPUT_DIR}" ]; then
            best_throughput=0
            best_latency=999999
            best_scenario=""
            
            for json_file in "${OUTPUT_DIR}"/*"${mode}"*.json; do
                if [ -f "$json_file" ]; then
                    throughput=$(jq -r '.throughput_tokens_per_sec.mean // 0' "$json_file")
                    latency=$(jq -r '.latency_ms_per_token.mean // 999999' "$json_file")
                    scenario_name=$(jq -r '.scenario.name // "unknown"' "$json_file")
                    
                    # Track best performance
                    if (( $(echo "$throughput > $best_throughput" | bc -l))); then
                        best_throughput=$throughput
                        best_latency=$latency
                        best_scenario=$scenario_name
                    fi
                fi
            done
            
            if [ "$best_throughput" != "0" ]; then
                printf "%-18s | %-15s | %15.1f | %15.2f | %s\n" \
                    "$mode" "$best_throughput" "$best_latency" "$best_scenario"
            fi
        fi
    done
    
    echo ""
    echo "=== DETAILED RESULTS ==="
    echo "Directory: ${OUTPUT_DIR}"
    echo "Files:"
    ls -la "${OUTPUT_DIR}/" | grep "\.json$"
fi

echo ""
echo "[run_spec_suite] Enhanced benchmark suite completed successfully!"