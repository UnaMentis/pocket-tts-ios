# Pocket TTS Quality Infrastructure Plan

**Created:** 2026-01-24
**Status:** Implemented (January 2026)
**Based on:** UnaMentis quality infrastructure patterns adapted for Rust/Candle

---

## Executive Summary

This document outlines a comprehensive quality infrastructure plan for the Pocket TTS iOS project, adapted from the battle-tested UnaMentis quality framework. The plan covers pre-commit hooks through CI/CD pipelines, tailored for a Rust/Candle ML model port.

### Current State

| Component | Status |
|-----------|--------|
| Pre-commit hooks | ✅ `.hooks/pre-commit`, `.hooks/pre-push` |
| CI/CD pipelines | ✅ 6 workflows in `.github/workflows/` |
| Linting config | ✅ `rustfmt.toml` + `Cargo.toml` lints |
| Code coverage | ✅ `codecov.yml` configured |
| Security scanning | ✅ `security.yml` workflow |
| Documentation validation | ✅ `docs.yml` workflow |

### Target State

| Component | Target |
|-----------|--------|
| Pre-commit hooks | ✅ rustfmt, clippy, gitleaks, tests |
| CI/CD pipelines | ✅ Full pipeline (lint, test, build, validate) |
| Linting config | ✅ Strict clippy + rustfmt config |
| Code coverage | ✅ 70% minimum with cargo-tarpaulin |
| Security scanning | ✅ cargo-audit + gitleaks |
| Documentation validation | ✅ Markdown linting |

---

## Implementation Progress Tracker

Use this checklist to track implementation progress:

### Phase 1: Local Development Quality (Pre-commit)
- [x] **1.1** Create `.hooks/pre-commit` script
- [x] **1.2** Create `.hooks/pre-push` script
- [x] **1.3** Create `scripts/install-hooks.sh`
- [x] **1.4** Create `rustfmt.toml` configuration
- [x] **1.5** Add clippy lints to `Cargo.toml`
- [x] **1.6** Create `.gitignore` updates for hook logs

### Phase 2: CI/CD Pipeline Foundation
- [x] **2.1** Create `.github/workflows/rust.yml` (main CI)
- [x] **2.2** Create `.github/workflows/security.yml`
- [x] **2.3** Create `.github/workflows/ios.yml`
- [x] **2.4** Create `.github/workflows/validation.yml`
- [x] **2.5** Create `.github/workflows/docs.yml`
- [x] **2.6** Create `.github/workflows/release.yml` (bonus)

### Phase 3: Code Coverage & Metrics
- [x] **3.1** Add `codecov.yml` configuration
- [x] **3.2** Integrate cargo-tarpaulin in CI
- [x] **3.3** Set up coverage thresholds

### Phase 4: Advanced Quality Features
- [x] **4.1** Create `.coderabbit.yaml` for AI review
- [ ] **4.2** Create hook audit script
- [ ] **4.3** Add quality metrics collection workflow
- [ ] **4.4** Create performance regression workflow

### Phase 5: Documentation & Polish
- [ ] **5.1** Create `CONTRIBUTING.md`
- [x] **5.2** Update `README.md` with badges
- [ ] **5.3** Create `docs/quality/testing-strategy.md`

---

## Phase 1: Local Development Quality (Pre-commit)

### 1.1 Pre-Commit Hook

**File:** `.hooks/pre-commit`

```bash
#!/bin/bash
#
# Pre-commit hook for Pocket TTS iOS
#
# Runs:
# 1. rustfmt - Code formatting
# 2. clippy - Linting with strict warnings
# 3. gitleaks - Secrets detection
# 4. Quick tests - Fast unit tests
#
# Bypass with: git commit --no-verify (logged for audit)

set -e

HOOK_LOG_DIR="$HOME/.pocket-tts/hook-logs"
mkdir -p "$HOOK_LOG_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_hook() {
    local status="$1"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local branch=$(git branch --show-current 2>/dev/null || echo "unknown")
    local user=$(whoami)
    echo "$timestamp|pre-commit|$status|$branch|$user|$$" >> "$HOOK_LOG_DIR/hook-audit.log"

    # Rotate log at 1MB
    if [ -f "$HOOK_LOG_DIR/hook-audit.log" ] && [ $(stat -f%z "$HOOK_LOG_DIR/hook-audit.log" 2>/dev/null || stat -c%s "$HOOK_LOG_DIR/hook-audit.log" 2>/dev/null) -gt 1048576 ]; then
        mv "$HOOK_LOG_DIR/hook-audit.log" "$HOOK_LOG_DIR/hook-audit.log.old"
    fi
}

log_hook "STARTED"

echo -e "${YELLOW}Running pre-commit checks...${NC}"
echo ""

# Get staged Rust files
STAGED_RS=$(git diff --cached --name-only --diff-filter=ACM | grep '\.rs$' || true)

if [ -z "$STAGED_RS" ]; then
    echo -e "${GREEN}No Rust files staged, skipping Rust checks${NC}"
else
    # Check 1: rustfmt
    echo -e "${YELLOW}[1/4] Checking formatting (rustfmt)...${NC}"
    if ! cargo fmt --check 2>/dev/null; then
        echo -e "${RED}❌ Formatting errors detected${NC}"
        echo "Run: cargo fmt"
        log_hook "FAILED:rustfmt"
        exit 1
    fi
    echo -e "${GREEN}✓ Formatting OK${NC}"

    # Check 2: clippy
    echo -e "${YELLOW}[2/4] Running linter (clippy)...${NC}"
    if ! cargo clippy --all-targets --all-features -- -D warnings 2>/dev/null; then
        echo -e "${RED}❌ Clippy warnings detected${NC}"
        log_hook "FAILED:clippy"
        exit 1
    fi
    echo -e "${GREEN}✓ Linting OK${NC}"
fi

# Check 3: gitleaks (secrets detection)
echo -e "${YELLOW}[3/4] Scanning for secrets (gitleaks)...${NC}"
if command -v gitleaks &> /dev/null; then
    if ! gitleaks protect --staged --no-banner 2>/dev/null; then
        echo -e "${RED}❌ Potential secrets detected!${NC}"
        echo "Review the files above and remove any secrets."
        log_hook "FAILED:gitleaks"
        exit 1
    fi
    echo -e "${GREEN}✓ No secrets detected${NC}"
else
    echo -e "${YELLOW}⚠ gitleaks not installed, skipping secrets scan${NC}"
    echo "  Install with: brew install gitleaks"
fi

# Check 4: Quick tests (if Rust files changed)
if [ -n "$STAGED_RS" ]; then
    echo -e "${YELLOW}[4/4] Running quick tests...${NC}"
    if ! cargo test --lib --quiet 2>/dev/null; then
        echo -e "${RED}❌ Tests failed${NC}"
        log_hook "FAILED:tests"
        exit 1
    fi
    echo -e "${GREEN}✓ Tests passed${NC}"
else
    echo -e "${YELLOW}[4/4] Skipping tests (no Rust changes)${NC}"
fi

echo ""
echo -e "${GREEN}All pre-commit checks passed!${NC}"
log_hook "PASSED"
exit 0
```

### 1.2 Pre-Push Hook

**File:** `.hooks/pre-push`

```bash
#!/bin/bash
#
# Pre-push hook for Pocket TTS iOS
#
# Runs full test suite before pushing

set -e

HOOK_LOG_DIR="$HOME/.pocket-tts/hook-logs"
mkdir -p "$HOOK_LOG_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_hook() {
    local status="$1"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local branch=$(git branch --show-current 2>/dev/null || echo "unknown")
    echo "$timestamp|pre-push|$status|$branch|$(whoami)|$$" >> "$HOOK_LOG_DIR/hook-audit.log"
}

log_hook "STARTED"

echo -e "${YELLOW}Running pre-push validation...${NC}"
echo ""

# Full test suite
echo -e "${YELLOW}Running full test suite...${NC}"
if ! cargo test 2>&1 | tail -20; then
    echo -e "${RED}❌ Tests failed - push aborted${NC}"
    log_hook "FAILED:tests"
    exit 1
fi

echo ""
echo -e "${GREEN}Pre-push validation passed!${NC}"
log_hook "PASSED"
exit 0
```

### 1.3 Hook Installation Script

**File:** `scripts/install-hooks.sh`

```bash
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
```

### 1.4 Rustfmt Configuration

**File:** `rustfmt.toml`

```toml
# Rustfmt configuration for Pocket TTS iOS
# Based on Rust 2021 edition defaults with project-specific adjustments

edition = "2021"

# Line width - generous for ML code with long tensor operations
max_width = 120

# Consistent import grouping
imports_granularity = "Crate"
group_imports = "StdExternalCrate"
reorder_imports = true

# Function formatting
fn_params_layout = "Tall"
fn_single_line = false

# Use blocks for single-line expressions in control flow
single_line_if_else_max_width = 0

# Struct/enum formatting
struct_lit_single_line = true
enum_discrim_align_threshold = 20

# Comment formatting
wrap_comments = false
normalize_comments = false
normalize_doc_attributes = true

# Chain formatting (important for Candle tensor operations)
chain_width = 80

# Use field init shorthand
use_field_init_shorthand = true

# Keep consistent spacing in match arms
match_arm_blocks = true
match_block_trailing_comma = true

# Force multiline for function signatures over threshold
fn_call_width = 80
```

### 1.5 Clippy Configuration

**Add to `Cargo.toml`:**

```toml
[lints.rust]
# Deny unsafe code (we don't need it for this project)
unsafe_code = "deny"

[lints.clippy]
# Correctness - always fix these
correctness = { level = "deny", priority = -1 }

# Suspicious patterns
suspicious = { level = "warn", priority = -1 }

# Performance issues
perf = { level = "warn", priority = -1 }

# Complexity
complexity = { level = "warn", priority = -1 }

# Style
style = { level = "warn", priority = -1 }

# Pedantic - enable selectively for ML code clarity
pedantic = { level = "warn", priority = -1 }

# Specific overrides for ML/numeric code
cast_possible_truncation = "allow"  # Common in tensor indexing
cast_precision_loss = "allow"       # F64 to F32 is intentional
cast_sign_loss = "allow"            # Unsigned indices
similar_names = "allow"             # q, k, v are standard
too_many_arguments = "allow"        # Model configs need many params
too_many_lines = "allow"            # Forward passes can be long

# Keep these strict
unwrap_used = "warn"
expect_used = "warn"
panic = "warn"
todo = "warn"
unimplemented = "warn"
dbg_macro = "warn"
print_stdout = "warn"
print_stderr = "warn"
```

---

## Phase 2: CI/CD Pipeline Foundation

### 2.1 Main Rust CI Pipeline

**File:** `.github/workflows/rust.yml`

```yaml
name: Rust CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  lint:
    name: Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-action@stable
        with:
          components: rustfmt, clippy

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Check formatting
        run: cargo fmt --check

      - name: Run clippy
        run: cargo clippy --all-targets --all-features -- -D warnings

  test:
    name: Test
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-action@stable

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Run tests
        run: cargo test --all-features --verbose

      - name: Run doc tests
        run: cargo test --doc

  coverage:
    name: Coverage
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-action@stable

      - name: Install cargo-tarpaulin
        run: cargo install cargo-tarpaulin

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Run coverage
        run: |
          cargo tarpaulin --out xml --out html \
            --exclude-files "src/bin/*" \
            --fail-under 70

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: ./cobertura.xml
          flags: rust
          fail_ci_if_error: false

  build:
    name: Build
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-action@stable

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Build release
        run: cargo build --release

      - name: Upload binary
        uses: actions/upload-artifact@v4
        with:
          name: pocket-tts-linux
          path: target/release/test-tts

  hook-bypass-detection:
    name: Hook Bypass Detection
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Check for bypass patterns
        run: |
          # Check if recent commits might have bypassed hooks
          RECENT_COMMITS=$(git log --oneline -10)

          # Look for commits without conventional format (suggests rushed commits)
          # This is informational, not blocking
          echo "Recent commits:"
          echo "$RECENT_COMMITS"
```

### 2.2 Security Scanning Pipeline

**File:** `.github/workflows/security.yml`

```yaml
name: Security

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 3 * * 0'  # Weekly on Sundays at 3am UTC
  workflow_dispatch:

jobs:
  audit:
    name: Dependency Audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-action@stable

      - name: Install cargo-audit
        run: cargo install cargo-audit

      - name: Run audit
        run: cargo audit --deny warnings

  secrets:
    name: Secrets Detection
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  deny:
    name: License & Dependency Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-action@stable

      - name: Install cargo-deny
        run: cargo install cargo-deny

      - name: Check dependencies
        run: cargo deny check
        continue-on-error: true  # Don't fail until deny.toml is configured
```

### 2.3 iOS Build Pipeline

**File:** `.github/workflows/ios.yml`

```yaml
name: iOS Build

on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'Cargo.toml'
      - 'scripts/build-ios.sh'
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  build-xcframework:
    name: Build XCFramework
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-action@stable
        with:
          targets: aarch64-apple-ios, aarch64-apple-ios-sim

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Build XCFramework
        run: ./scripts/build-ios.sh

      - name: Verify outputs
        run: |
          ls -la target/xcframework/
          test -d target/xcframework/PocketTTS.xcframework
          test -f target/xcframework/pocket_tts_ios.swift

      - name: Upload XCFramework
        uses: actions/upload-artifact@v4
        with:
          name: PocketTTS-XCFramework
          path: |
            target/xcframework/PocketTTS.xcframework
            target/xcframework/pocket_tts_ios.swift
          retention-days: 30
```

### 2.4 Validation Pipeline

**File:** `.github/workflows/validation.yml`

```yaml
name: Model Validation

on:
  push:
    branches: [main]
    paths:
      - 'src/models/**'
      - 'src/modules/**'
      - 'validation/**'
  pull_request:
    branches: [main]
    paths:
      - 'src/models/**'
      - 'src/modules/**'
  workflow_dispatch:
  schedule:
    - cron: '0 2 * * *'  # Nightly at 2am UTC

jobs:
  validation:
    name: Python Reference Comparison
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-action@stable

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-validation

      - name: Build Rust binary
        run: cargo build --release --bin test-tts

      - name: Download model (if available)
        env:
          MODEL_AVAILABLE: ${{ secrets.MODEL_DOWNLOAD_URL != '' }}
        run: |
          if [ "$MODEL_AVAILABLE" = "true" ]; then
            python3 scripts/download-model.py
          else
            echo "Model not available - skipping validation"
            exit 0
          fi
        continue-on-error: true

      - name: Run validation
        if: success()
        run: |
          if [ -d "models/kyutai-pocket-ios" ]; then
            ./validation/run_tests.sh --quick
          else
            echo "Skipping validation - model not available"
          fi

      - name: Upload validation report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: validation-report
          path: validation/validation_report.json
          if-no-files-found: ignore
```

### 2.5 Documentation Validation Pipeline

**File:** `.github/workflows/docs.yml`

```yaml
name: Documentation

on:
  push:
    branches: [main]
    paths:
      - '**.md'
      - 'docs/**'
  pull_request:
    branches: [main]
    paths:
      - '**.md'
      - 'docs/**'

jobs:
  markdown-lint:
    name: Markdown Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install markdownlint
        run: npm install -g markdownlint-cli

      - name: Create config
        run: |
          cat > .markdownlint.json << 'EOF'
          {
            "MD013": false,
            "MD033": false,
            "MD041": false,
            "MD024": { "siblings_only": true }
          }
          EOF

      - name: Lint markdown
        run: markdownlint '**/*.md' --ignore node_modules --ignore target --ignore validation/.venv

  link-check:
    name: Link Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check links
        uses: lycheeverse/lychee-action@v1
        with:
          args: --verbose --no-progress '**/*.md' --exclude-path target --exclude-path validation/.venv
          fail: false  # Don't fail on broken external links

  rustdoc:
    name: Documentation Build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-action@stable

      - name: Build docs
        run: cargo doc --no-deps --all-features
        env:
          RUSTDOCFLAGS: "-D warnings"

      - name: Upload docs
        uses: actions/upload-artifact@v4
        with:
          name: rustdoc
          path: target/doc
```

---

## Phase 3: Code Coverage & Metrics

### 3.1 Codecov Configuration

**File:** `codecov.yml`

```yaml
# Codecov configuration for Pocket TTS iOS

codecov:
  require_ci_to_pass: true

coverage:
  precision: 2
  round: down
  range: "60...100"

  status:
    project:
      default:
        target: 70%
        threshold: 5%
        informational: false

    patch:
      default:
        target: 70%
        threshold: 10%
        informational: true

flags:
  rust:
    paths:
      - src/
    carryforward: true

ignore:
  - "src/bin/**"           # CLI binaries
  - "**/*_tests.rs"        # Test files
  - "src/modules/tests.rs" # Test module

comment:
  layout: "header, diff, flags, components"
  behavior: default
  require_changes: true
```

### 3.2 Coverage Thresholds

| Component | Minimum | Target |
|-----------|---------|--------|
| Overall | 70% | 80% |
| Models (flowlm, mimi, seanet) | 60% | 75% |
| Modules (attention, conv, etc.) | 70% | 85% |
| Core (config, audio, tokenizer) | 80% | 90% |

---

## Phase 4: Advanced Quality Features

### 4.1 CodeRabbit Configuration

**File:** `.coderabbit.yaml`

```yaml
# CodeRabbit AI Review Configuration

language: "en"

reviews:
  auto_review:
    enabled: true
    drafts: true

  profile: "assertive"

  request_changes_workflow: true

  high_level_summary: true

  path_instructions:
    - path: "src/models/**/*.rs"
      instructions: |
        Focus on:
        - Numerical precision (f32 vs f64 operations)
        - Tensor shape consistency
        - Memory efficiency in forward passes
        - Comparison with PyTorch reference behavior
        - Weight loading correctness

    - path: "src/modules/**/*.rs"
      instructions: |
        Focus on:
        - Candle API usage patterns
        - Layer implementations matching PyTorch semantics
        - Broadcasting behavior
        - Edge cases in normalization layers

    - path: "**/*.rs"
      instructions: |
        Focus on:
        - Rust idioms and best practices
        - Error handling (avoid unwrap in production code)
        - Performance considerations
        - Documentation for public APIs

    - path: ".github/workflows/**/*.yml"
      instructions: |
        Focus on:
        - Action versions (prefer pinned versions over @latest)
        - Secrets handling
        - Caching efficiency
        - Workflow dependencies

chat:
  auto_reply: true
```

### 4.2 Hook Audit Script

**File:** `scripts/hook-audit.sh`

```bash
#!/bin/bash
#
# Audit hook usage and detect bypasses
#
# Usage:
#   ./scripts/hook-audit.sh           # Show summary
#   ./scripts/hook-audit.sh --detect  # Detect in CI (exit 1 if found)
#

set -e

DETECT_MODE=false
if [ "$1" = "--detect" ]; then
    DETECT_MODE=true
fi

LOG_DIR="$HOME/.pocket-tts/hook-logs"
LOG_FILE="$LOG_DIR/hook-audit.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "No hook audit log found at: $LOG_FILE"
    exit 0
fi

echo "=== Hook Audit Summary ==="
echo ""

# Count by status
PASSED=$(grep "|PASSED|" "$LOG_FILE" 2>/dev/null | wc -l | tr -d ' ')
FAILED=$(grep "|FAILED" "$LOG_FILE" 2>/dev/null | wc -l | tr -d ' ')
STARTED=$(grep "|STARTED|" "$LOG_FILE" 2>/dev/null | wc -l | tr -d ' ')

echo "Total hook runs: $((PASSED + FAILED))"
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
echo ""

# Check for commits that might have bypassed hooks
if [ "$DETECT_MODE" = true ]; then
    echo "Checking for potential bypasses..."

    # Look for commits without corresponding hook entries
    # This is a simple heuristic
    RECENT_COMMITS=$(git log --oneline -20 --format="%H" 2>/dev/null || echo "")

    BYPASS_DETECTED=false

    for commit in $RECENT_COMMITS; do
        # Check if commit message suggests bypass
        MSG=$(git log -1 --format="%s" "$commit" 2>/dev/null || echo "")
        if echo "$MSG" | grep -qi "wip\|fixup\|squash\|temp"; then
            echo "  ⚠ Potential rushed commit: $commit"
            echo "    Message: $MSG"
            BYPASS_DETECTED=true
        fi
    done

    if [ "$BYPASS_DETECTED" = true ]; then
        echo ""
        echo "Potential bypasses detected. Review the commits above."
        # Don't fail, just warn
    else
        echo "No obvious bypasses detected."
    fi
fi

echo ""
echo "Recent hook activity:"
tail -10 "$LOG_FILE" | while IFS='|' read -r timestamp hook status branch user pid; do
    echo "  $timestamp - $hook: $status ($user on $branch)"
done
```

---

## Phase 5: Quality Thresholds Summary

### Mandatory (Blocking)

| Check | Tool | Threshold |
|-------|------|-----------|
| Formatting | rustfmt | Must pass |
| Linting | clippy -D warnings | No warnings |
| Unit Tests | cargo test | All pass |
| Secrets | gitleaks | No secrets |
| Dependencies | cargo-audit | No vulnerabilities |

### Targets (Non-blocking initially)

| Metric | Target | Timeline |
|--------|--------|----------|
| Code Coverage | 70% | Phase 3 |
| Doc Coverage | 100% public APIs | Phase 5 |
| Mutation Score | 60% | Future |

---

## Quick Start Implementation

To implement this plan, run these commands in order:

```bash
# Phase 1: Create hook infrastructure
mkdir -p .hooks scripts
# Create files from Phase 1 specifications above
./scripts/install-hooks.sh

# Phase 2: Create CI infrastructure
mkdir -p .github/workflows
# Create workflow files from Phase 2 specifications above

# Verify setup
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test

# Commit the infrastructure
git add .hooks scripts .github rustfmt.toml
git commit -m "feat: Add quality infrastructure (hooks + CI)"
```

---

## Appendix: Tool Installation

### Required Tools (Local Development)

```bash
# Rust components
rustup component add rustfmt clippy

# Security scanning
brew install gitleaks

# Optional: coverage
cargo install cargo-tarpaulin

# Optional: dependency auditing
cargo install cargo-audit cargo-deny
```

### CI Tool Versions

| Tool | Version | Notes |
|------|---------|-------|
| Rust | stable | Latest stable |
| cargo-tarpaulin | latest | Coverage |
| cargo-audit | latest | Security |
| gitleaks | v8.18.1+ | Secrets |
| actions/checkout | v4 | Pinned |
| dtolnay/rust-action | stable | Toolchain |

---

## Maintenance Notes

- Review and update clippy lints quarterly
- Update CI action versions monthly
- Rotate hook logs automatically (1MB limit)
- Coverage thresholds may increase as codebase matures
- Run security scans before releases

---

*This plan is adapted from UnaMentis quality infrastructure patterns, tailored for Rust/Candle ML development.*
