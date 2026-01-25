#!/bin/bash
# Package Pocket TTS iOS XCFramework for release
#
# Usage: ./scripts/package-release.sh <version>
# Example: ./scripts/package-release.sh 0.4.0
#
# This script packages the built XCFramework and Swift bindings
# into a release zip file ready for distribution.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.4.0"
    exit 1
fi

VERSION="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/release"
RELEASE_NAME="PocketTTS-v$VERSION"
XCFRAMEWORK_DIR="$PROJECT_DIR/target/xcframework"

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

# Copy documentation
echo "Copying documentation..."
cp "$PROJECT_DIR/LICENSE" "$OUTPUT_DIR/$RELEASE_NAME/"
cp "$PROJECT_DIR/CHANGELOG.md" "$OUTPUT_DIR/$RELEASE_NAME/"

# Copy integration guide (or create a basic one)
if [ -f "$PROJECT_DIR/docs/INTEGRATION.md" ]; then
    cp "$PROJECT_DIR/docs/INTEGRATION.md" "$OUTPUT_DIR/$RELEASE_NAME/README.md"
else
    cat > "$OUTPUT_DIR/$RELEASE_NAME/README.md" << 'EOF'
# Pocket TTS iOS

Text-to-speech synthesis for iOS using the Kyutai Pocket TTS model.

## Requirements

- iOS 17.0+
- Xcode 15+

## Installation

1. Drag `PocketTTS.xcframework` into your Xcode project
2. Add Swift files from `Sources/` to your project
3. Download model files (see Model Files section)

## Quick Start

```swift
import PocketTTS

// Initialize engine
let engine = try PocketTTSEngine(modelPath: modelPath)

// Configure
let config = TTSConfig(voiceIndex: 0, temperature: 0.7, speed: 1.0)
try engine.configure(config: config)

// Synthesize
let result = try engine.synthesize(text: "Hello, world!")
// result.samples contains Float32 audio at 24kHz
```

## Model Files

Download model files and include in your app bundle:

```
Models/
├── model.safetensors     # Main model (~225MB)
├── tokenizer.model       # Tokenizer (~60KB)
└── voices/               # Voice embeddings (~4MB)
    ├── alba.safetensors
    ├── marius.safetensors
    └── ...
```

## Available Voices

0. Alba, 1. Marius, 2. Javert, 3. Jean, 4. Fantine, 5. Cosette, 6. Eponine, 7. Azelma

## License

MIT License - See LICENSE file
EOF
fi

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
