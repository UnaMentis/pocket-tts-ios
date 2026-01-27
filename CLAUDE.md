# CLAUDE.md - Pocket TTS iOS (Rust/Candle)

This directory contains the Rust/Candle implementation of Kyutai Pocket TTS for native iOS inference.

## Quick Commands

```bash
# Check compilation
cargo check

# Build debug
cargo build

# Build release
cargo build --release

# Run tests
cargo test

# Lint
cargo clippy -- -D warnings

# Format
cargo fmt

# Build for iOS (creates XCFramework)
./scripts/build-ios.sh

# Run latency benchmark (streaming mode with TTFA)
./scripts/run-latency-bench.sh --streaming

# Quick latency test (3 iterations)
./scripts/run-latency-bench.sh --quick
```

## Architecture

The implementation follows the Pocket TTS architecture:

```
Text → Tokenizer → FlowLM (Transformer) → MLP Sampler → Mimi Decoder → Audio
```

### Components

| Component | File | Description |
|-----------|------|-------------|
| FlowLM | `src/models/flowlm.rs` | 6-layer transformer (~70M params) |
| MLP Sampler | `src/modules/mlp.rs` | Consistency sampling (~10M params) |
| Mimi Decoder | `src/models/mimi.rs` | VAE decoder (~20M params) |
| SEANet | `src/models/seanet.rs` | Upsampling convolutions |
| Tokenizer | `src/tokenizer.rs` | SentencePiece wrapper |

### Modules

| Module | File | Description |
|--------|------|-------------|
| Attention | `src/modules/attention.rs` | Multi-head attention with KV cache |
| RoPE | `src/modules/rotary.rs` | Rotary position embeddings |
| MLP | `src/modules/mlp.rs` | Feed-forward and gated MLP |
| Conv | `src/modules/conv.rs` | Causal convolutions, SEANet blocks |
| Embeddings | `src/modules/embeddings.rs` | Text and voice embeddings |
| LayerNorm | `src/modules/layer_norm.rs` | RMS and standard layer norm |

## UniFFI Interface

The Swift interface is defined in `src/pocket_tts.udl`:

- `PocketTTSEngine` - Main engine class
- `TTSConfig` - Configuration (voice, temperature, speed, etc.)
- `SynthesisResult` - Audio output
- `AudioChunk` - Streaming chunk
- `TTSEventHandler` - Streaming callback

## iOS Build Process

1. Build for device: `cargo build --release --target aarch64-apple-ios`
2. Build for simulator: `cargo build --release --target aarch64-apple-ios-sim`
3. Generate Swift bindings: `cargo run --bin uniffi-bindgen generate ...`
4. Create XCFramework: `xcodebuild -create-xcframework ...`

The `scripts/build-ios.sh` script automates this process.

## Model Files

The Rust implementation loads model files from:

```
kyutai-pocket-ios/
├── model.safetensors     # Main model weights
├── tokenizer.model       # SentencePiece tokenizer
└── voices/               # Voice embeddings
    ├── alba.safetensors
    └── ...
```

## Performance Notes

- CPU-only on iOS (Candle doesn't support Metal on iOS)
- Pocket TTS is optimized for CPU, targeting ~3-4x realtime on iPhone
- Memory-mapped safetensors for efficient loading
- KV caching for efficient streaming

## Latency Testing

Reference baselines (see `docs/LATENCY_TESTING.md` for full details):

| Metric | Target | Acceptable |
|--------|--------|------------|
| TTFA (Time To First Audio) | ~200ms | ≤300ms |
| RTF (Real-Time Factor) | 3-4x | ≥2.5x |

Run latency benchmarks:
```bash
# Streaming mode (measures TTFA)
./scripts/run-latency-bench.sh --streaming

# Both sync and streaming
./scripts/run-latency-bench.sh --all

# Quick 3-iteration test
./scripts/run-latency-bench.sh --quick
```

The iOS demo app also displays TTFA when using streaming mode (toggle Sync/Stream).

## Dependencies

Key crates:
- `candle-core`, `candle-nn` - ML framework
- `uniffi` - Swift FFI bindings
- `safetensors` - Model loading
- `tokenizers` - SentencePiece
- `rubato` - Audio resampling
- `hound` - WAV encoding

## Integration with iOS App

After building the XCFramework:

1. Add `PocketTTS.xcframework` to Xcode project
2. Add `pocket_tts_ios.swift` (generated bindings)
3. Optionally add `swift/PocketTTSSwift.swift` for async/await API
4. Bundle model files in app resources

See `rust/pocket-tts-ios/README.md` for detailed integration instructions.

## Git Policy

**IMPORTANT: Claude MUST NOT run `git commit` or `git push` without explicit user permission.**

- `git add` (staging files) is allowed and helpful
- `git commit` is FORBIDDEN - user will commit manually
- `git push` is FORBIDDEN - user will push manually
- `git status` and `git diff` are allowed for inspection

This ensures the user maintains control over what gets committed to the repository.

## Audit Reports

Periodic audit reports are stored in `docs/audit/`. Review these for cleanup tasks and recommendations.

## Multi-Agent Collaboration

This project uses a multi-agent collaboration pattern for development. See [docs/prompts/AGENT_ORCHESTRATION.md](docs/prompts/AGENT_ORCHESTRATION.md) for full details.

### Agent Quick Reference

| When | Run This Agent |
|------|----------------|
| After code changes to `src/models/` or `src/modules/` | **Verification Agent** - checks numerical accuracy |
| When stuck for >1 hour | **Research Advisor** - external research and fresh hypotheses |
| Before committing changes | **Cleanup Auditor** - finds debug code and technical debt |
| Weekly or for planning | **Progress Tracker** - dashboard with metrics and timeline |

### Running an Agent

1. Start a **fresh** Claude Code session
2. Paste the prompt from `docs/prompts/[agent-name].md`
3. Let it complete and save its report to `docs/audit/`
4. Review the report in your main session

### Available Prompts

- `docs/prompts/verification-agent.md` - Numerical accuracy testing
- `docs/prompts/research-advisor.md` - External research for blockers
- `docs/prompts/cleanup-audit.md` - Technical debt inventory
- `docs/prompts/progress-tracker.md` - Progress dashboard
