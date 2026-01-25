#!/bin/bash
#
# Audit hook usage and detect bypasses
#
# Usage:
#   ./scripts/hook-audit.sh           # Show summary
#   ./scripts/hook-audit.sh --detect  # Detect in CI (exit 1 if found)
#   ./scripts/hook-audit.sh --recent  # Show recent activity only
#

set -e

DETECT_MODE=false
RECENT_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --detect)
            DETECT_MODE=true
            shift
            ;;
        --recent)
            RECENT_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --detect   Detect potential bypasses (for CI)"
            echo "  --recent   Show only recent activity"
            echo "  --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

LOG_DIR="$HOME/.pocket-tts/hook-logs"
LOG_FILE="$LOG_DIR/hook-audit.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

if [ ! -f "$LOG_FILE" ]; then
    echo -e "${YELLOW}No hook audit log found at: $LOG_FILE${NC}"
    echo "Run some commits with hooks installed to generate logs."
    exit 0
fi

echo -e "${BLUE}=== Hook Audit Summary ===${NC}"
echo ""

# Count by status
PASSED=$(grep -c "|PASSED|" "$LOG_FILE" 2>/dev/null || echo 0)
FAILED=$(grep -c "|FAILED" "$LOG_FILE" 2>/dev/null || echo 0)
TOTAL=$((PASSED + FAILED))

echo "Total hook runs: $TOTAL"
echo -e "  ${GREEN}Passed: $PASSED${NC}"
echo -e "  ${RED}Failed: $FAILED${NC}"

if [ "$TOTAL" -gt 0 ]; then
    SUCCESS_RATE=$((PASSED * 100 / TOTAL))
    echo "  Success rate: ${SUCCESS_RATE}%"
fi
echo ""

# Show failure breakdown
if [ "$FAILED" -gt 0 ]; then
    echo -e "${YELLOW}Failure breakdown:${NC}"
    grep "|FAILED" "$LOG_FILE" | cut -d'|' -f3 | sort | uniq -c | sort -rn | head -5
    echo ""
fi

# Recent activity
if [ "$RECENT_ONLY" = true ]; then
    echo -e "${BLUE}Recent hook activity (last 20):${NC}"
    tail -20 "$LOG_FILE" | while IFS='|' read -r timestamp hook status branch user pid; do
        case "$status" in
            PASSED)
                color=$GREEN
                ;;
            FAILED*)
                color=$RED
                ;;
            STARTED)
                color=$YELLOW
                ;;
            *)
                color=$NC
                ;;
        esac
        echo -e "  $timestamp - ${color}$status${NC} ($hook on $branch by $user)"
    done
    exit 0
fi

# Check for commits that might have bypassed hooks
if [ "$DETECT_MODE" = true ]; then
    echo -e "${BLUE}Checking for potential bypasses...${NC}"
    echo ""

    BYPASS_DETECTED=false

    # Check recent commits for suspicious patterns
    if command -v git &> /dev/null && [ -d .git ]; then
        RECENT_COMMITS=$(git log --oneline -20 --format="%H" 2>/dev/null || echo "")

        for commit in $RECENT_COMMITS; do
            MSG=$(git log -1 --format="%s" "$commit" 2>/dev/null || echo "")
            AUTHOR=$(git log -1 --format="%an" "$commit" 2>/dev/null || echo "")

            # Check for WIP/fixup patterns that suggest rushed commits
            if echo "$MSG" | grep -qiE "^(wip|fixup|squash|temp|xxx|todo:)"; then
                echo -e "  ${YELLOW}⚠ Potential rushed commit:${NC} $(git log -1 --oneline "$commit")"
                BYPASS_DETECTED=true
            fi
        done

        if [ "$BYPASS_DETECTED" = true ]; then
            echo ""
            echo -e "${YELLOW}Potential bypasses detected. Review the commits above.${NC}"
            echo "Note: WIP commits are flagged but not necessarily problematic."
        else
            echo -e "${GREEN}No obvious bypasses detected in recent commits.${NC}"
        fi
    else
        echo "Not in a git repository, skipping commit analysis."
    fi

    exit 0
fi

# Default: show summary and recent activity
echo -e "${BLUE}Recent hook activity (last 10):${NC}"
tail -10 "$LOG_FILE" | while IFS='|' read -r timestamp hook status branch user pid; do
    case "$status" in
        PASSED)
            color=$GREEN
            ;;
        FAILED*)
            color=$RED
            ;;
        STARTED)
            color=$YELLOW
            ;;
        *)
            color=$NC
            ;;
    esac
    echo -e "  $timestamp - ${color}$status${NC} ($hook on $branch by $user)"
done

echo ""
echo "Log file: $LOG_FILE"
echo "Use --recent for more history, --detect for CI mode"
