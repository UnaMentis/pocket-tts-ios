#!/bin/bash
#
# Install git hooks for Pocket TTS iOS
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HOOKS_SOURCE="$PROJECT_DIR/.hooks"
HOOKS_TARGET="$PROJECT_DIR/.git/hooks"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Installing git hooks..."

# Ensure .git/hooks exists
mkdir -p "$HOOKS_TARGET"

# Install each hook
for hook in pre-commit pre-push; do
    if [ -f "$HOOKS_SOURCE/$hook" ]; then
        cp "$HOOKS_SOURCE/$hook" "$HOOKS_TARGET/$hook"
        chmod +x "$HOOKS_TARGET/$hook"
        echo -e "${GREEN}✓ Installed $hook${NC}"
    else
        echo -e "${YELLOW}⚠ Hook not found: $HOOKS_SOURCE/$hook${NC}"
    fi
done

# Create log directory
LOG_DIR="$HOME/.pocket-tts/hook-logs"
mkdir -p "$LOG_DIR"
echo -e "${GREEN}✓ Created log directory: $LOG_DIR${NC}"

echo ""
echo "Git hooks installed successfully!"
echo ""
echo "Optional dependencies for full functionality:"
echo "  brew install gitleaks    # Secrets detection"
echo ""
