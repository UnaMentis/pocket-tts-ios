#!/bin/bash
#
# Comprehensive Quality Check for Pocket TTS
#
# Runs waveform comparison and quality metrics analysis
#
# Usage:
#   ./run_quality_check.sh --reference python_ref.wav --rust rust_output.wav --text "Hello world"
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
REFERENCE=""
RUST=""
TEXT=""
BASELINE=""
CHECK_REGRESSION=false
SAVE_BASELINE=""
OUTPUT_DIR="validation/quality_reports"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --reference)
            REFERENCE="$2"
            shift 2
            ;;
        --rust)
            RUST="$2"
            shift 2
            ;;
        --text)
            TEXT="$2"
            shift 2
            ;;
        --baseline)
            BASELINE="$2"
            shift 2
            ;;
        --check-regression)
            CHECK_REGRESSION=true
            shift
            ;;
        --save-baseline)
            SAVE_BASELINE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --reference FILE --rust FILE --text TEXT [--baseline FILE] [--check-regression] [--save-baseline FILE]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$REFERENCE" ] || [ -z "$RUST" ]; then
    echo -e "${RED}Error: --reference and --rust are required${NC}" >&2
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_JSON="$OUTPUT_DIR/quality_report_$TIMESTAMP.json"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Pocket TTS Quality Check${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Run waveform comparison with quality metrics
echo -e "${YELLOW}[1/3] Running waveform comparison and quality metrics...${NC}"

QUALITY_ARGS=""
if [ -n "$TEXT" ]; then
    QUALITY_ARGS="--text '$TEXT'"
fi

python3 validation/compare_waveforms.py \
    --reference "$REFERENCE" \
    --rust "$RUST" \
    --quality-metrics \
    $QUALITY_ARGS \
    --output-json "$REPORT_JSON"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Quality metrics completed${NC}"
else
    echo -e "${RED}✗ Quality metrics failed${NC}"
    exit 1
fi
echo ""

# Step 2: Save baseline if requested
if [ -n "$SAVE_BASELINE" ]; then
    echo -e "${YELLOW}[2/3] Saving baseline...${NC}"
    python3 validation/baseline_tracker.py \
        --save "$SAVE_BASELINE" \
        --metrics "$REPORT_JSON"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Baseline saved to $SAVE_BASELINE${NC}"
    else
        echo -e "${RED}✗ Failed to save baseline${NC}"
        exit 1
    fi
    echo ""
fi

# Step 3: Check regression if baseline provided
if [ -n "$BASELINE" ]; then
    echo -e "${YELLOW}[3/3] Checking for regressions...${NC}"

    if [ "$CHECK_REGRESSION" = true ]; then
        # Fail on regression
        python3 validation/baseline_tracker.py \
            --check-regression \
            --baseline "$BASELINE" \
            --metrics "$REPORT_JSON"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ No regressions detected${NC}"
        else
            echo -e "${RED}✗ Regressions detected - failing${NC}"
            exit 1
        fi
    else
        # Just compare, don't fail
        python3 validation/baseline_tracker.py \
            --compare \
            --baseline "$BASELINE" \
            --metrics "$REPORT_JSON"
    fi
    echo ""
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Quality check completed successfully${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved to: $REPORT_JSON"

if [ -n "$BASELINE" ]; then
    echo "Compared against: $BASELINE"
fi
