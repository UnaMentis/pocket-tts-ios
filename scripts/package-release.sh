#!/bin/bash
# Package Pocket TTS iOS XCFramework for release
#
# Usage: ./scripts/package-release.sh <version>
# Example: ./scripts/package-release.sh 0.4.0
#
# This script packages the built XCFramework and Swift bindings
# into a release zip file ready for distribution.

set -euo pipefail

# Parse arguments: <version> [--allow-dirty]
VERSION=""
ALLOW_DIRTY=0
for arg in "$@"; do
    case "$arg" in
        --allow-dirty)
            ALLOW_DIRTY=1
            ;;
        -*)
            echo "Error: unknown option '$arg'"
            echo "Usage: $0 <version> [--allow-dirty]"
            exit 1
            ;;
        *)
            if [ -z "$VERSION" ]; then
                VERSION="$arg"
            else
                echo "Error: unexpected extra argument '$arg'"
                echo "Usage: $0 <version> [--allow-dirty]"
                exit 1
            fi
            ;;
    esac
done

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version> [--allow-dirty]"
    echo "Example: $0 0.5.0"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/release"
RELEASE_NAME="PocketTTS-v$VERSION"
XCFRAMEWORK_DIR="$PROJECT_DIR/target/xcframework"

# PROVENANCE GUARD: refuse to build a release artifact from a dirty working
# tree (this is exactly the unknown-provenance failure this script guards
# against). Only tracked modifications count as "dirty" here, matching
# `git describe --dirty` semantics — stray untracked files (e.g. local
# report notes) do not block a release. Override with --allow-dirty.
cd "$PROJECT_DIR"
if git rev-parse --git-dir >/dev/null 2>&1; then
    if ! git diff --quiet HEAD 2>/dev/null; then
        if [ "$ALLOW_DIRTY" -eq 1 ]; then
            echo "WARNING: working tree has uncommitted changes to tracked files;"
            echo "         proceeding because --allow-dirty was passed."
        else
            echo "Error: working tree is dirty (uncommitted changes to tracked files)."
            echo "A dirty release build produces an unknown-provenance artifact."
            echo "Commit/stash your changes, or pass --allow-dirty to override."
            exit 1
        fi
    fi
else
    echo "Warning: not inside a git repository; provenance metadata will be limited."
fi

echo "Packaging Pocket TTS iOS v$VERSION..."
echo "Project: $PROJECT_DIR"
echo "Output: $OUTPUT_DIR/$RELEASE_NAME.zip"

# Verify XCFramework exists
if [ ! -d "$XCFRAMEWORK_DIR/PocketTTS.xcframework" ]; then
    echo "Error: XCFramework not found at $XCFRAMEWORK_DIR/PocketTTS.xcframework"
    echo "Run ./scripts/build-ios.sh first"
    exit 1
fi

# Verify Swift bindings exist
if [ ! -f "$XCFRAMEWORK_DIR/pocket_tts_ios.swift" ]; then
    echo "Error: Swift bindings not found at $XCFRAMEWORK_DIR/pocket_tts_ios.swift"
    echo "Run ./scripts/build-ios.sh first"
    exit 1
fi

# Clean and create output directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/$RELEASE_NAME/Sources"

# Copy XCFramework
echo "Copying XCFramework..."
cp -r "$XCFRAMEWORK_DIR/PocketTTS.xcframework" "$OUTPUT_DIR/$RELEASE_NAME/"

# Copy Swift files
echo "Copying Swift bindings..."
cp "$XCFRAMEWORK_DIR/pocket_tts_ios.swift" "$OUTPUT_DIR/$RELEASE_NAME/Sources/"

# Copy high-level Swift wrapper if it exists
if [ -f "$PROJECT_DIR/swift/PocketTTSSwift.swift" ]; then
    cp "$PROJECT_DIR/swift/PocketTTSSwift.swift" "$OUTPUT_DIR/$RELEASE_NAME/Sources/"
fi

# Copy model files
echo "Copying model files..."
MODELS_DIR="$PROJECT_DIR/models/kyutai-pocket-ios"
if [ -d "$MODELS_DIR" ]; then
    mkdir -p "$OUTPUT_DIR/$RELEASE_NAME/Models"
    cp "$MODELS_DIR/model.safetensors" "$OUTPUT_DIR/$RELEASE_NAME/Models/"
    cp "$MODELS_DIR/tokenizer.model" "$OUTPUT_DIR/$RELEASE_NAME/Models/"
    if [ -d "$MODELS_DIR/voices" ]; then
        cp -r "$MODELS_DIR/voices" "$OUTPUT_DIR/$RELEASE_NAME/Models/"
    fi
else
    echo "Error: Model files not found at $MODELS_DIR"
    echo "Expected: model.safetensors, tokenizer.model, voices/"
    exit 1
fi

# Copy documentation
echo "Copying documentation..."
cp "$PROJECT_DIR/LICENSE" "$OUTPUT_DIR/$RELEASE_NAME/"
cp "$PROJECT_DIR/CHANGELOG.md" "$OUTPUT_DIR/$RELEASE_NAME/"

# Copy attribution (required — the code is derived work; the zip must carry it)
if [ -f "$PROJECT_DIR/ATTRIBUTION.md" ]; then
    cp "$PROJECT_DIR/ATTRIBUTION.md" "$OUTPUT_DIR/$RELEASE_NAME/"
else
    echo "Error: ATTRIBUTION.md not found at $PROJECT_DIR/ATTRIBUTION.md"
    echo "The release must ship attribution for the derived code and model weights."
    exit 1
fi

# Copy integration guide. FAIL CLOSED: never ship a fabricated README with
# broken example code. If the real integration guide is missing, stop.
if [ -f "$PROJECT_DIR/docs/INTEGRATION.md" ]; then
    cp "$PROJECT_DIR/docs/INTEGRATION.md" "$OUTPUT_DIR/$RELEASE_NAME/README.md"
else
    echo "Error: docs/INTEGRATION.md not found at $PROJECT_DIR/docs/INTEGRATION.md"
    echo "Refusing to ship a release without a real integration guide."
    exit 1
fi

# WEIGHTS LICENSE: the zip redistributes Kyutai model weights (v1,
# english_2026-01) under CC-BY-4.0. The MIT LICENSE covers only the code and
# does NOT cover the weights — state this explicitly next to the weights.
echo "Writing model weights license..."
cat > "$OUTPUT_DIR/$RELEASE_NAME/Models/WEIGHTS_LICENSE.txt" << 'EOF'
Model Weights License
=====================

The model weights in this directory (model.safetensors, tokenizer.model, and
the voice files under voices/) are the Kyutai Pocket TTS weights
(v1, english_2026-01).

  Copyright (c) Kyutai
  Licensed under Creative Commons Attribution 4.0 International (CC-BY-4.0)
  License text: https://creativecommons.org/licenses/by/4.0/
  Source:       https://huggingface.co/kyutai/pocket-tts-without-voice-cloning

IMPORTANT: The code license for this project (MIT, see ../LICENSE) applies ONLY
to the software. It does NOT cover these model weights. The weights are governed
solely by the CC-BY-4.0 license above. If you redistribute the weights you must
provide attribution to Kyutai as required by CC-BY-4.0.

See ../ATTRIBUTION.md for the full attribution chain.
EOF

# PROVENANCE: write BUILD_INFO.txt at the zip root so every artifact carries a
# verifiable record of exactly what commit / tree it was built from.
echo "Writing build provenance (BUILD_INFO.txt)..."
cd "$PROJECT_DIR"
if git rev-parse --git-dir >/dev/null 2>&1; then
    GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    GIT_DESCRIBE="$(git describe --tags --always --dirty 2>/dev/null || echo unknown)"
else
    GIT_SHA="unknown"
    GIT_DESCRIBE="unknown"
fi
BUILD_TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    BUILDER="CI (GitHub Actions)"
else
    BUILDER="local ($(whoami)@$(hostname -s 2>/dev/null || hostname))"
fi
cat > "$OUTPUT_DIR/$RELEASE_NAME/BUILD_INFO.txt" << EOF
Pocket TTS iOS — Build Provenance
=================================

version:         $VERSION
git_sha:         $GIT_SHA
git_describe:    $GIT_DESCRIBE
build_timestamp: $BUILD_TIMESTAMP
builder:         $BUILDER
EOF
echo "BUILD_INFO.txt:"
cat "$OUTPUT_DIR/$RELEASE_NAME/BUILD_INFO.txt"

# Create zip
echo "Creating zip archive..."
cd "$OUTPUT_DIR"
zip -r "$RELEASE_NAME.zip" "$RELEASE_NAME"

# Calculate checksum
echo ""
echo "Calculating checksum..."
shasum -a 256 "$RELEASE_NAME.zip" > "$RELEASE_NAME.zip.sha256"

# Summary
echo ""
echo "Package complete!"
echo ""
echo "Output files:"
echo "  Archive:  $OUTPUT_DIR/$RELEASE_NAME.zip"
echo "  Checksum: $OUTPUT_DIR/$RELEASE_NAME.zip.sha256"
echo ""
echo "Contents:"
unzip -l "$RELEASE_NAME.zip" | head -20
echo ""
echo "Size: $(du -h "$RELEASE_NAME.zip" | cut -f1)"
