# Contributing to Pocket TTS iOS

Thank you for your interest in contributing! This document provides guidelines for contributing to the Pocket TTS iOS project.

## Getting Started

### Prerequisites

1. **Rust toolchain** (1.75+) with iOS targets:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup target add aarch64-apple-ios
   rustup target add aarch64-apple-ios-sim
   ```

2. **Xcode** with iOS SDK (iOS 17+)

3. **Optional tools**:
   ```bash
   brew install gitleaks  # Secrets detection
   ```

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/UnaMentis/pocket-tts-ios.git
   cd pocket-tts-ios
   ```

2. Install git hooks:
   ```bash
   ./scripts/install-hooks.sh
   ```

3. Verify setup:
   ```bash
   cargo check
   cargo test
   ```

## Development Workflow

### Common Commands

```bash
# Check compilation
cargo check

# Run tests
cargo test

# Run lints
cargo clippy -- -D warnings

# Format code
cargo fmt

# Build iOS XCFramework
./scripts/build-ios.sh
```

### Pre-commit Hooks

The project uses pre-commit hooks that run automatically:
- **rustfmt** - Code formatting
- **clippy** - Linting with strict warnings
- **gitleaks** - Secrets detection
- **Quick tests** - Fast unit test suite

If you need to bypass hooks (not recommended):
```bash
git commit --no-verify -m "message"
```

Bypasses are logged for audit purposes.

## Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write tests for new functionality
   - Update documentation if needed
   - Follow the code style guidelines below

3. **Ensure quality checks pass**:
   ```bash
   cargo fmt --check
   cargo clippy -- -D warnings
   cargo test
   ```

4. **Submit your PR**:
   - Write a clear description of the changes
   - Reference any related issues
   - Ensure CI passes

5. **Address review feedback**:
   - Respond to comments
   - Push additional commits as needed

## Code Style

### Rust Guidelines

- Follow the project's `rustfmt.toml` settings
- No clippy warnings (enforced by CI)
- Document public APIs with doc comments
- Use meaningful variable names (ML code may use short names like `q`, `k`, `v` for standard concepts)

### Commit Messages

Use clear, descriptive commit messages:
```
feat: Add new voice support
fix: Correct RoPE embedding calculation
docs: Update integration guide
test: Add FlowLM unit tests
```

### What to Avoid

- Don't commit debug statements (`eprintln!`, `dbg!`)
- Don't commit secrets or API keys
- Don't commit large binary files
- Don't modify generated files directly

## Reporting Issues

### Bug Reports

Include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Rust version, iOS version, device)
- Relevant logs or error messages

### Feature Requests

Include:
- Description of the feature
- Use case / motivation
- Potential implementation approach (optional)

## Project Structure

```
pocket-tts/
├── src/                 # Rust source code
│   ├── models/          # ML model implementations
│   ├── modules/         # Neural network modules
│   └── lib.rs           # Library entry point
├── swift/               # Swift wrapper code
├── scripts/             # Build and utility scripts
├── tests/               # Test harness and demo app
├── docs/                # Documentation
│   ├── quality/         # Quality infrastructure
│   ├── python-reference/# Python implementation docs
│   └── prompts/         # Agent prompts
└── validation/          # Validation scripts
```

## Questions?

- Check existing issues and discussions
- Open a new issue for questions
- See [README.md](README.md) for project overview
- See [docs/quality/QUALITY_PLAN.md](docs/quality/QUALITY_PLAN.md) for quality standards

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
