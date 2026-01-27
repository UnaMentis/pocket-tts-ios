#!/bin/bash
#
# Run Pocket TTS latency benchmarks
#
# Usage:
#   ./scripts/run-latency-bench.sh                    # Run sync benchmark
#   ./scripts/run-latency-bench.sh --streaming        # Run streaming benchmark (TTFA measurement)
#   ./scripts/run-latency-bench.sh --all              # Run both modes
#   ./scripts/run-latency-bench.sh --quick            # Quick 3-iteration test
#
# Reference Baselines:
#   TTFA (Time To First Audio): ~200ms
#   RTF (Real-Time Factor): 3-4x
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/../kyutai-pocket-ios}"
OUTPUT_DIR="$PROJECT_DIR/benchmark-results"
ITERATIONS=5
WARMUP=1
MODE="sync"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║           Pocket TTS Latency Benchmark Runner                ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_baseline() {
    echo -e "${YELLOW}Reference Baselines:${NC}"
    echo "  TTFA (Time To First Audio): ~200ms"
    echo "  RTF (Real-Time Factor):     3-4x realtime"
    echo "  Per-latent generation:      ~50ms"
    echo ""
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --streaming, -s     Use streaming mode (measures TTFA)"
    echo "  --sync              Use synchronous mode (default)"
    echo "  --all               Run both streaming and sync modes"
    echo "  --quick             Quick test with 3 iterations"
    echo "  --iterations N      Number of iterations per phrase (default: 5)"
    echo "  --warmup N          Warmup iterations (default: 1)"
    echo "  --model-dir PATH    Path to model directory"
    echo "  --output-dir PATH   Directory for JSON results"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  MODEL_DIR           Path to model directory (default: ../kyutai-pocket-ios)"
    echo ""
}

# Parse arguments
RUN_ALL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --streaming|-s)
            MODE="streaming"
            shift
            ;;
        --sync)
            MODE="sync"
            shift
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --quick)
            ITERATIONS=3
            WARMUP=1
            shift
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

print_header
print_baseline

# Check model directory
if [[ ! -d "$MODEL_DIR" ]]; then
    echo -e "${RED}ERROR: Model directory not found: $MODEL_DIR${NC}"
    echo ""
    echo "Set MODEL_DIR environment variable or use --model-dir option"
    exit 1
fi

echo "Configuration:"
echo "  Model directory: $MODEL_DIR"
echo "  Iterations:      $ITERATIONS"
echo "  Warmup:          $WARMUP"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build release binary
echo -e "${BLUE}Building latency benchmark (release mode)...${NC}"
cd "$PROJECT_DIR"
cargo build --release --bin latency-bench 2>/dev/null || {
    echo -e "${RED}Build failed. Trying with verbose output:${NC}"
    cargo build --release --bin latency-bench
    exit 1
}
echo -e "${GREEN}Build complete.${NC}"
echo ""

# Run benchmarks
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_benchmark() {
    local mode=$1
    local json_file="$OUTPUT_DIR/latency_${mode}_${TIMESTAMP}.json"

    echo -e "${BLUE}Running $mode benchmark...${NC}"
    echo ""

    if [[ "$mode" == "streaming" ]]; then
        cargo run --release --bin latency-bench -- \
            --model-dir "$MODEL_DIR" \
            --iterations "$ITERATIONS" \
            --warmup "$WARMUP" \
            --streaming \
            --json "$json_file"
    else
        cargo run --release --bin latency-bench -- \
            --model-dir "$MODEL_DIR" \
            --iterations "$ITERATIONS" \
            --warmup "$WARMUP" \
            --json "$json_file"
    fi

    echo ""
    echo -e "${GREEN}Results saved to: $json_file${NC}"
}

if [[ "$RUN_ALL" == true ]]; then
    run_benchmark "sync"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    run_benchmark "streaming"
else
    run_benchmark "$MODE"
fi

echo ""
echo -e "${GREEN}Benchmark complete!${NC}"
echo ""
echo "View results:"
echo "  cat $OUTPUT_DIR/latency_*_${TIMESTAMP}.json | jq '.summary'"
echo ""
