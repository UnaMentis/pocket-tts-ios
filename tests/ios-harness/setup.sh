#!/bin/bash
# Setup script for PocketTTS iOS Demo App
#
# This script creates an Xcode project that integrates the PocketTTS XCFramework
# and bundles the model files for testing.
#
# Usage: ./setup.sh [release-zip-path]
#
# If no release zip is provided, it will download the latest release from GitHub.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/PocketTTSDemo"
RELEASE_ZIP="${1:-}"

echo "=== PocketTTS iOS Demo Setup ==="
echo ""

# Create project directory structure
mkdir -p "$PROJECT_DIR/PocketTTSDemo"
mkdir -p "$PROJECT_DIR/Frameworks"
mkdir -p "$PROJECT_DIR/Resources/Models"

# Download or extract release
if [ -z "$RELEASE_ZIP" ]; then
    echo "Downloading latest release from GitHub..."
    TEMP_DIR=$(mktemp -d)
    gh release download --repo UnaMentis/pocket-tts-ios --pattern "*.zip" --dir "$TEMP_DIR"
    RELEASE_ZIP="$TEMP_DIR"/*.zip
fi

echo "Extracting release..."
EXTRACT_DIR=$(mktemp -d)
unzip -q "$RELEASE_ZIP" -d "$EXTRACT_DIR"
RELEASE_DIR=$(find "$EXTRACT_DIR" -maxdepth 1 -type d -name "PocketTTS-*" | head -1)

if [ -z "$RELEASE_DIR" ]; then
    echo "Error: Could not find release directory in zip"
    exit 1
fi

# Copy framework
echo "Copying XCFramework..."
cp -r "$RELEASE_DIR/PocketTTS.xcframework" "$PROJECT_DIR/Frameworks/"

# Copy Swift bindings
echo "Copying Swift bindings..."
cp "$RELEASE_DIR/Sources/pocket_tts_ios.swift" "$PROJECT_DIR/PocketTTSDemo/"
cp "$RELEASE_DIR/Sources/PocketTTSSwift.swift" "$PROJECT_DIR/PocketTTSDemo/"

# Copy model files
echo "Copying model files..."
cp -r "$RELEASE_DIR/Models/"* "$PROJECT_DIR/Resources/Models/"

# Create Info.plist
cat > "$PROJECT_DIR/PocketTTSDemo/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>$(DEVELOPMENT_LANGUAGE)</string>
    <key>CFBundleExecutable</key>
    <string>$(EXECUTABLE_NAME)</string>
    <key>CFBundleIdentifier</key>
    <string>$(PRODUCT_BUNDLE_IDENTIFIER)</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$(PRODUCT_NAME)</string>
    <key>CFBundlePackageType</key>
    <string>$(PRODUCT_BUNDLE_PACKAGE_TYPE)</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSRequiresIPhoneOS</key>
    <true/>
    <key>UIApplicationSceneManifest</key>
    <dict>
        <key>UIApplicationSupportsMultipleScenes</key>
        <true/>
    </dict>
    <key>UILaunchScreen</key>
    <dict/>
    <key>UIRequiredDeviceCapabilities</key>
    <array>
        <string>arm64</string>
    </array>
    <key>UISupportedInterfaceOrientations</key>
    <array>
        <string>UIInterfaceOrientationPortrait</string>
        <string>UIInterfaceOrientationLandscapeLeft</string>
        <string>UIInterfaceOrientationLandscapeRight</string>
    </array>
</dict>
</plist>
EOF

# Create Xcode project using xcodegen or manual project file
echo "Creating Xcode project..."

# Check if xcodegen is available
if command -v xcodegen &> /dev/null; then
    # Create project.yml for xcodegen
    cat > "$PROJECT_DIR/project.yml" << 'EOF'
name: PocketTTSDemo
options:
  bundleIdPrefix: com.unamentis
  deploymentTarget:
    iOS: "17.0"
  xcodeVersion: "15.0"

targets:
  PocketTTSDemo:
    type: application
    platform: iOS
    sources:
      - PocketTTSDemo
    resources:
      - path: Resources/Models
        buildPhase: resources
    settings:
      PRODUCT_BUNDLE_IDENTIFIER: com.unamentis.PocketTTSDemo
      INFOPLIST_FILE: PocketTTSDemo/Info.plist
      SWIFT_VERSION: "5.9"
      ENABLE_USER_SCRIPT_SANDBOXING: NO
    dependencies:
      - framework: Frameworks/PocketTTS.xcframework
        embed: true
EOF

    cd "$PROJECT_DIR"
    xcodegen generate
    echo "Xcode project created with xcodegen!"
else
    echo ""
    echo "Note: xcodegen not found. Creating project manually..."
    echo ""
    echo "To complete setup:"
    echo "1. Open Xcode and create a new iOS App project named 'PocketTTSDemo'"
    echo "2. Set the project location to: $PROJECT_DIR"
    echo "3. Add the Swift files from PocketTTSDemo/"
    echo "4. Drag Frameworks/PocketTTS.xcframework into the project"
    echo "5. Add Resources/Models as a folder reference to the target"
    echo ""
    echo "Or install xcodegen: brew install xcodegen"
    echo "Then re-run this script."
fi

# Cleanup
rm -rf "$EXTRACT_DIR"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Project location: $PROJECT_DIR"
echo ""
echo "Contents:"
echo "  - Frameworks/PocketTTS.xcframework"
echo "  - Resources/Models/ (model.safetensors, tokenizer.model, voices/)"
echo "  - PocketTTSDemo/ (Swift source files)"
echo ""
echo "To build and run:"
echo "  1. Open $PROJECT_DIR/PocketTTSDemo.xcodeproj in Xcode"
echo "  2. Select a simulator or device"
echo "  3. Press Cmd+R to build and run"
