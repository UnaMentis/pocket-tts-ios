#!/bin/bash
#
# Pocket TTS iOS - Master Test Runner
#
# This script runs the complete validation suite:
# 1. Build Rust (if needed)
# 2. Generate reference outputs (if needed)
# 3. Generate Rust outputs
# 4. Compare waveforms and latents
# 5. Run ASR round-trip tests (optional)
#
# Usage:
#   ./run_tests.sh              # Full test suite
#   ./run_tests.sh --quick      # Skip ASR tests
#   ./run_tests.sh --rebuild    # Force rebuild of Rust binary
#   ./run_tests.sh --regen-ref  # Regenerate Python reference outputs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_ROOT/models/kyutai-pocket-ios"
VENV_DIR="$SCRIPT_DIR/venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
QUICK_MODE=false
FORCE_REBUILD=false
REGEN_REFERENCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --regen-ref)
            REGEN_REFERENCE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick      Skip ASR round-trip tests (faster)"
            echo "  --rebuild    Force rebuild of Rust binary"
            echo "  --regen-ref  Regenerate Python reference outputs"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Pocket TTS iOS - Validation Suite  ${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Check model directory
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${RED}Error: Model directory not found: $MODEL_DIR${NC}"
    echo "Run: python3 scripts/download-model.py"
    exit 1
fi

echo -e "${GREEN}✓ Model directory found${NC}"

# Step 1: Build Rust
echo ""
echo -e "${YELLOW}Step 1: Building Rust binary...${NC}"

RUST_BINARY="$PROJECT_ROOT/target/release/test-tts"

if [ "$FORCE_REBUILD" = true ] || [ ! -f "$RUST_BINARY" ]; then
    echo "Building release binary..."
    cd "$PROJECT_ROOT"
    cargo build --release --bin test-tts 2>&1 | tail -10
    cd "$SCRIPT_DIR"
else
    echo "Binary exists, skipping build (use --rebuild to force)"
fi

if [ ! -f "$RUST_BINARY" ]; then
    echo -e "${RED}Error: Rust binary not found after build${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Rust binary ready${NC}"

# Step 2: Set up Python environment
echo ""
echo -e "${YELLOW}Step 2: Setting up Python environment...${NC}"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install requirements
echo "Checking Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

echo -e "${GREEN}✓ Python environment ready${NC}"

# Step 3: Generate reference outputs if needed
echo ""
echo -e "${YELLOW}Step 3: Checking reference outputs...${NC}"

REF_MANIFEST="$SCRIPT_DIR/reference_outputs/manifest.json"

if [ "$REGEN_REFERENCE" = true ] || [ ! -f "$REF_MANIFEST" ]; then
    echo "Generating Python reference outputs..."

    # Check if reference harness can import pocket_tts
    if python3 -c "import pocket_tts" 2>/dev/null; then
        python3 "$SCRIPT_DIR/reference_harness.py" --model-dir "$MODEL_DIR" --with-whisper
    else
        echo -e "${YELLOW}Warning: pocket_tts Python module not available${NC}"
        echo "Reference outputs must exist or be generated manually"

        if [ ! -f "$REF_MANIFEST" ]; then
            echo -e "${RED}Error: No reference outputs and cannot generate them${NC}"
            echo "Install the Python pocket_tts package or provide pre-generated outputs"
            exit 1
        fi
    fi
else
    echo "Reference outputs exist, skipping generation"
fi

if [ -f "$REF_MANIFEST" ]; then
    echo -e "${GREEN}✓ Reference outputs available${NC}"
else
    echo -e "${RED}Error: Reference outputs not found${NC}"
    exit 1
fi

# Step 4: Run Rust harness and compare
echo ""
echo -e "${YELLOW}Step 4: Running validation...${NC}"

# Build validation command
VALIDATE_ARGS="--model-dir $MODEL_DIR --json-report $SCRIPT_DIR/validation_report.json"

if [ "$QUICK_MODE" = true ]; then
    VALIDATE_ARGS="$VALIDATE_ARGS --skip-asr"
fi

# Run the validation script
python3 "$SCRIPT_DIR/validate.py" $VALIDATE_ARGS
VALIDATE_RESULT=$?

# Step 5: Summary
echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}           TEST SUMMARY               ${NC}"
echo -e "${BLUE}======================================${NC}"

if [ $VALIDATE_RESULT -eq 0 ]; then
    echo -e "${GREEN}All tests PASSED!${NC}"
else
    echo -e "${RED}Some tests FAILED${NC}"
    echo ""
    echo "See validation_report.json for details"
    echo ""
    echo "Next steps for debugging:"
    echo "  1. Check latent comparison with: python3 compare_intermediates.py"
    echo "  2. Export debug outputs: $RUST_BINARY --model-dir $MODEL_DIR --export-latents latents.npy"
    echo "  3. Compare with Python: python3 dump_intermediates.py"
fi

# Deactivate venv
deactivate

exit $VALIDATE_RESULT
