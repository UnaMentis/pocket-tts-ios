#!/bin/bash
# Build pocket-tts-ios for iOS targets
#
# This script builds the Rust library for both iOS device and simulator,
# then packages them into an XCFramework for use in Xcode.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/target/xcframework"
FRAMEWORK_NAME="PocketTTS"

echo "Building pocket-tts-ios for iOS..."
echo "Project: $PROJECT_DIR"
echo "Output: $OUTPUT_DIR"

# Ensure iOS targets are installed
echo ""
echo "Checking Rust targets..."
rustup target add aarch64-apple-ios 2>/dev/null || true
rustup target add aarch64-apple-ios-sim 2>/dev/null || true

# Clean previous builds
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Build for iOS device (arm64)
echo ""
echo "Building for iOS device (aarch64-apple-ios)..."
cd "$PROJECT_DIR"
cargo build --release --target aarch64-apple-ios

# Build for iOS simulator (arm64)
echo ""
echo "Building for iOS simulator (aarch64-apple-ios-sim)..."
cargo build --release --target aarch64-apple-ios-sim

# Generate Swift bindings
echo ""
echo "Generating Swift bindings..."
cargo run --bin uniffi-bindgen generate \
    src/pocket_tts.udl \
    --language swift \
    --out-dir "$OUTPUT_DIR/bindings"

# Create header directory
HEADERS_DIR="$OUTPUT_DIR/headers"
mkdir -p "$HEADERS_DIR"

# Copy module map
cat > "$HEADERS_DIR/module.modulemap" << 'EOF'
framework module PocketTTS {
    umbrella header "PocketTTS.h"
    export *
    module * { export * }
}

module pocket_tts_iosFFI {
    header "pocket_tts_iosFFI.h"
    export *
}
EOF

# Create umbrella header
cat > "$HEADERS_DIR/PocketTTS.h" << 'EOF'
#ifndef PocketTTS_h
#define PocketTTS_h

#include "pocket_tts_iosFFI.h"

#endif /* PocketTTS_h */
EOF

# Copy FFI header from bindings
if [ -f "$OUTPUT_DIR/bindings/pocket_tts_iosFFI.h" ]; then
    cp "$OUTPUT_DIR/bindings/pocket_tts_iosFFI.h" "$HEADERS_DIR/"
fi

# Create XCFramework
echo ""
echo "Creating XCFramework..."

DEVICE_LIB="$PROJECT_DIR/target/aarch64-apple-ios/release/libpocket_tts_ios.a"
SIM_LIB="$PROJECT_DIR/target/aarch64-apple-ios-sim/release/libpocket_tts_ios.a"

if [ ! -f "$DEVICE_LIB" ]; then
    echo "Error: Device library not found at $DEVICE_LIB"
    exit 1
fi

if [ ! -f "$SIM_LIB" ]; then
    echo "Error: Simulator library not found at $SIM_LIB"
    exit 1
fi

xcodebuild -create-xcframework \
    -library "$DEVICE_LIB" \
    -headers "$HEADERS_DIR" \
    -library "$SIM_LIB" \
    -headers "$HEADERS_DIR" \
    -output "$OUTPUT_DIR/$FRAMEWORK_NAME.xcframework"

# Copy Swift bindings
echo ""
echo "Copying Swift bindings..."
cp "$OUTPUT_DIR/bindings/pocket_tts_ios.swift" "$OUTPUT_DIR/"

# Summary
echo ""
echo "Build complete!"
echo ""
echo "Output files:"
echo "  XCFramework: $OUTPUT_DIR/$FRAMEWORK_NAME.xcframework"
echo "  Swift file:  $OUTPUT_DIR/pocket_tts_ios.swift"
echo ""
echo "To use in Xcode:"
echo "  1. Drag $FRAMEWORK_NAME.xcframework into your Xcode project"
echo "  2. Add pocket_tts_ios.swift to your project"
echo "  3. Import and use PocketTTSEngine in your Swift code"
