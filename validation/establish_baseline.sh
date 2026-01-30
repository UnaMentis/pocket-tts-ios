#!/bin/bash
#
# Establish Quality Baseline for Pocket TTS
#
# Generates reference audio and creates initial baseline
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
TEST_PHRASE="Hello, this is a test."
RUST_OUTPUT="validation/baseline_test_rust.wav"
PYTHON_REF="validation/baseline_test_python.wav"
VERSION="v0.4.1"
BASELINE_FILE="validation/baselines/baseline_${VERSION}.json"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Establishing Baseline for ${VERSION}${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Generate Python reference
echo -e "${YELLOW}[1/4] Generating Python reference audio...${NC}"

if [ -f "$PYTHON_REF" ]; then
    echo "Reference already exists: $PYTHON_REF"
else
    python3 validation/generate_reference_audio.py \
        --text "$TEST_PHRASE" \
        --output "$PYTHON_REF" || {
        echo "Warning: Could not generate Python reference. Using existing reference files."
        PYTHON_REF="validation/reference_outputs/phrase_00.wav"
    }
fi
echo ""

# Step 2: Generate Rust output
echo -e "${YELLOW}[2/4] Generating Rust TTS output...${NC}"
./target/release/test-tts \
    --model-dir models/kyutai-pocket-ios \
    --text "$TEST_PHRASE" \
    --output "$RUST_OUTPUT"
echo -e "${GREEN}✓ Generated: $RUST_OUTPUT${NC}"
echo ""

# Step 3: Run quality metrics
echo -e "${YELLOW}[3/4] Running quality metrics...${NC}"
python3 validation/quality_metrics.py \
    --audio "$RUST_OUTPUT" \
    --text "$TEST_PHRASE" \
    --reference "$PYTHON_REF" \
    --whisper-model base \
    --output-json "validation/quality_reports/baseline_${VERSION}.json"
echo ""

# Step 4: Save baseline
echo -e "${YELLOW}[4/4] Saving baseline...${NC}"
python3 validation/baseline_tracker.py \
    --save "$BASELINE_FILE" \
    --metrics "validation/quality_reports/baseline_${VERSION}.json"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Baseline Established Successfully${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Baseline file: $BASELINE_FILE"
echo ""
echo "To compare future changes against this baseline:"
echo "  ./validation/run_quality_check.sh \\"
echo "    --reference $PYTHON_REF \\"
echo "    --rust <your_output.wav> \\"
echo "    --text \"$TEST_PHRASE\" \\"
echo "    --baseline $BASELINE_FILE \\"
echo "    --check-regression"
