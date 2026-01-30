# Research Advisor Briefing

**Date:** 2026-01-30
**Current Blocker:** None - Implementation is production-ready with random noise enabled
**Research Focus:** Quality validation methodologies, performance optimization opportunities, next-generation improvements

## Situational Summary

**MAJOR MILESTONE ACHIEVED**: The Rust/Candle port of Kyutai Pocket TTS is now **production-ready**. The previous research advisor report (2026-01-24) identified a "hidden state divergence" issue that was actually caused by Rust using zeros for noise (DEBUG mode) while Python used random noise. This has been fully resolved.

### Current Implementation Status (v0.4.1)

**Core Achievement**: Random noise enabled, matching Python's production behavior
- All 91 unit tests passing
- Audio quality metrics: 91% amplitude ratio, 98% RMS ratio vs Python
- Natural EOS detection working correctly
- Streaming Mimi decoder with replicate padding
- Real-time factor: ~3-4x on CPU (target achieved)

**API Cleanup**: Removed legacy token-chunked streaming method, keeping only:
- `synthesize()` - Sync mode for batch processing
- `synthesize_true_streaming()` - True streaming with ~200ms TTFA (preferred for on-device)

**Key Insight from Latest Verification Report (2026-01-25)**:
> "With random noise enabled, waveform correlation is NO LONGER a meaningful metric since different random number generators produce different (but equally valid) latent trajectories. Audio quality is now validated through amplitude/RMS ratios and listening tests."

### What Changed Since Last Research Report

1. **Root cause identified**: "Divergence" was DEBUG mode (zeros) vs production (random noise)
2. **Random noise enabled**: FlowNet now uses `Tensor::randn(0.0, 0.8367, ...)` matching Python
3. **New validation paradigm**: Shifted from waveform correlation (no longer meaningful) to audio quality metrics
4. **Streaming quality fixed**: Removed artificial crossfade, fixed callback EOS handling
5. **iOS AB testing infrastructure**: Added `decode_latents` API and reference validation in demo app

### Current State (No Active Blockers)

The implementation is feature-complete and ready for production use. The verification report shows:
- Audio produces intelligible speech with appropriate amplitude/RMS
- Duration appropriate for phrase length (1.84s vs 2.00s Python reference)
- No NaN, Inf, or clipping issues
- Performance meets target (~3.7x real-time factor)

**There is no active blocker.** The research focus is now on optimization and quality assurance.

---

## Key Research Findings

### From Official Kyutai Sources

**Official Release (January 13, 2026)**:
- Pocket TTS is a 100M-parameter TTS model designed to run on CPU without GPU
- Built on Continuous Audio Language model architecture
- Uses Mimi codec (12.5 Hz frame rate, 1.1 kbps bitrate, 80ms latency)
- Supports high-fidelity voice cloning from 5 seconds of audio
- Requires Python 3.10-3.14, PyTorch 2.5+ (CPU version sufficient)

**Technical Report Details**:
- Flow-based language model with neural audio codec (Mimi)
- Trained on speech data specifically for TTS applications
- Default EOS threshold: -4.0
- Sample rate: 24 kHz mono audio

**Sources:**
- [Kyutai Blog: Pocket TTS Release](https://kyutai.org/blog/2026-01-13-pocket-tts)
- [GitHub: kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts)
- [HuggingFace: kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts)

### From Mimi Neural Audio Codec Research

**Mimi Architecture** (state-of-the-art streaming codec):
- Combines semantic and acoustic information into audio tokens at 12.5 Hz
- Streaming encoder-decoder with quantized latent space
- Trained end-to-end on speech data
- Frame size: 80ms, bitrate: 1.1 kbps
- Improves over SoundStream and Encodec through joint modeling with distillation

**Validation Approach**:
- LibriSpeech validation dataset used for demonstrations
- Published September 2024 as part of Moshi speech-text foundation model

**Sources:**
- [HuggingFace: kyutai/mimi](https://huggingface.co/kyutai/mimi)
- [GitHub: kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)
- [Kyutai: Neural Audio Codecs Explainer](https://kyutai.org/codec-explainer)

### From babybirdprd Reference Implementation

**Community Rust/Candle Port**:
- GitHub repository: github.com/babybirdprd/pocket-tts
- Published to crates.io as `pocket-tts-cli` (version 0.1.1)
- Features: Candle framework, WebAssembly support, PyO3 bindings
- Can run in browser via WASM
- Contributed issue #43 to main Kyutai repository (Jan 16, 2026)

**Attribution Chain**:
This UnaMentis iOS port builds on babybirdprd's excellent Rust/Candle foundation, adding:
- UniFFI bindings for Swift interop
- XCFramework build system
- iOS-specific optimizations and demo app
- Streaming API for low-latency mobile TTS

**Sources:**
- [crates.io: pocket-tts-cli](https://crates.io/crates/pocket-tts-cli)
- [Kyutai Pocket TTS README](https://github.com/kyutai-labs/pocket-tts) (mentions babybirdprd version)

### From TTS Quality Validation Research

**Industry-Standard Metrics for TTS Validation**:

1. **ASR-Based Intelligibility (WER)**:
   - Word Error Rate (WER): Transcribe TTS output with ASR and compare to original text
   - Whisper ASR reduces WER by 72% vs prior models (RNN-T)
   - Joint ASR-TTS training shows 13% WER drop
   - **Limitation**: Low WER doesn't guarantee key-information accuracy

2. **Acoustic Similarity Metrics**:
   - Mel-Cepstral Distortion (MCD): Measures spectral differences via MFCCs
   - STOI/ESTOI: Short-Time Objective Intelligibility for noisy conditions
   - Lower values = better quality

3. **Subjective Evaluation**:
   - Mean Opinion Score (MOS): Human listeners rate naturalness (1-5 scale)
   - Attribute-specific ratings: naturalness, intelligibility, speaker similarity, emotional appropriateness

**Recommended Validation Workflow**:
```
TTS Output → Whisper ASR → WER calculation
          ↘ MCD vs reference audio
          ↘ Human MOS evaluation
```

**Sources:**
- [OpenAI Whisper](https://openai.com/index/whisper/)
- [ASR Guided Speech Intelligibility Measure (arXiv:2006.01463)](https://arxiv.org/abs/2006.01463)
- [Advanced Evaluation Metrics for ASR/TTS](https://apxml.com/courses/speech-recognition-synthesis-asr-tts/chapter-1-modern-speech-processing-foundations/advanced-evaluation-metrics)
- [LXT Speech Data Evaluation](https://www.lxt.ai/services/speech-data-evaluation/)

### From Candle iOS Performance Research

**Candle CoreML Integration** (candle-coreml crate):
- Bridges Candle tensors with Apple's CoreML framework
- Enables on-device inference on macOS and iOS
- Supports ANE (Apple Neural Engine) acceleration
- Accepts CPU and Metal tensors

**Performance Hierarchy on iOS**:
- ANE (fastest, most efficient) > GPU/Metal (fast) > CPU (most compatible)
- Apple automatically chooses best backend for compatible models
- **Critical**: Model must be ANE-compatible to benefit from acceleration

**Known Limitations**:
- Metal backend has reported issues with command buffer management at high concurrency (>6)
- Error: "IOGPUMetalCommandBuffer validate failed assertion"
- Recommendation: Use Apple's pre-optimized models for guaranteed ANE acceleration, or stick with Metal/CPU for general use

**Current Project Status**:
- Uses CPU-only on iOS (Candle doesn't fully support Metal on iOS per CLAUDE.md)
- Meets performance target (~3-4x realtime) on CPU
- Potential future optimization: Investigate candle-coreml for ANE acceleration

**Sources:**
- [crates.io: candle-coreml](https://crates.io/crates/candle-coreml)
- [Candle Issue #2818: Metal Command Buffer](https://github.com/huggingface/candle/discussions/2818)
- [Apple Metal Developer](https://developer.apple.com/metal/)

---

## Suggested Approaches

### High Confidence (Quality Assurance & Validation)

**1. Implement Whisper ASR Validation Pipeline** (QUALITY ASSURANCE)
- **Why**: Industry-standard method for measuring TTS intelligibility objectively
- **Confidence**: Very High - proven methodology with published benchmarks
- **How**:
  1. Add Whisper ASR as optional validation dependency
  2. Create validation script that:
     - Synthesizes test phrases with Rust TTS
     - Transcribes audio with Whisper large-v3
     - Computes WER vs original text
     - Sets threshold (e.g., WER <5% = pass, 5-10% = acceptable, >10% = investigate)
  3. Add to CI pipeline for regression testing
  4. Document WER results in verification reports
- **Expected outcome**: WER <5% confirms intelligibility matches human perception
- **Reference**: The validation/compare_waveforms.py already exists; extend with ASR

**2. Add MCD (Mel-Cepstral Distortion) Validation** (ACOUSTIC QUALITY)
- **Why**: Measures spectral similarity between Rust and Python outputs objectively
- **Confidence**: High - standard acoustic metric
- **How**:
  1. Extract MFCCs from both Rust and Python audio using librosa/scipy
  2. Compute MCD: `sqrt(2) * sqrt(sum((mfcc_rust - mfcc_python)^2))`
  3. Set threshold (MCD <6 dB = perceptually similar)
  4. Add to validation script alongside WER
- **Expected outcome**: Confirms acoustic similarity even when waveforms differ due to RNG

**3. iOS Listening Tests with Reference Audio** (SUBJECTIVE QUALITY)
- **Why**: The AB testing infrastructure is already in place (ReferenceTestView.swift)
- **Confidence**: High - validates real iOS app behavior
- **How**:
  1. Use existing iOS demo's "AB Test" tab
  2. Generate reference audio with Python Mimi using deterministic latents (already implemented)
  3. Test correlation on actual iPhone hardware (not just simulator)
  4. Verify playback quality with AVAudioPlayer
  5. Document results in verification report
- **Expected outcome**: Confirms >0.95 correlation on real iOS hardware when using same latents

### Worth Trying (Performance Optimization)

**4. Investigate Candle-CoreML for ANE Acceleration** (EXPERIMENTAL)
- **Why**: Potential 2-5x speedup if model is ANE-compatible
- **Confidence**: Medium - requires investigation into ANE compatibility
- **How**:
  1. Add candle-coreml as optional dependency
  2. Convert FlowLM transformer to CoreML format
  3. Test on iPhone with Instruments to confirm ANE usage
  4. Compare latency vs current CPU implementation
  5. Document tradeoffs (complexity vs performance gain)
- **Tradeoffs**: Added complexity, may not work if operations aren't ANE-compatible
- **Fallback**: Current CPU implementation already meets performance targets

**5. Optimize Memory Usage for Lower-End Devices** (MEMORY FOOTPRINT)
- **Why**: Current ~150MB memory usage could be reduced for older iPhones
- **Confidence**: Medium - requires profiling
- **How**:
  1. Profile with Xcode Instruments on target hardware (iPhone 12/13)
  2. Identify peak memory usage during inference
  3. Consider: KV cache size optimization, quantization (if Candle supports), lazy weight loading
  4. Test on iPhone SE (2020) or iPhone 12 as minimum spec
- **Expected outcome**: Reduce peak memory by 20-30% for broader device support

**6. Add Voice Embedding Caching** (LATENCY OPTIMIZATION)
- **Why**: Voice embeddings (125 frames) processed every synthesis, could be cached
- **Confidence**: Medium - straightforward optimization
- **How**:
  1. Cache the voice embedding tensor after first synthesis
  2. Reuse for subsequent syntheses with same voice
  3. Measure TTFA improvement (likely ~10-20ms reduction)
  4. Document in latency benchmark results
- **Expected outcome**: Minor TTFA improvement, especially for repeated syntheses

### Speculative (Next-Generation Features)

**7. Add Custom Voice Embedding Support** (FEATURE REQUEST)
- **Why**: Enable user-provided voice cloning (currently uses 8 built-in voices)
- **Confidence**: Low - requires additional encoder implementation
- **How**:
  1. Research Kyutai's voice encoding process (likely Mimi encoder)
  2. Add encoder to Rust crate (or call Python encoder via bridge)
  3. Add API for custom voice embedding generation from audio samples
  4. Validate with Kyutai's voice cloning examples
- **Tradeoffs**: Significantly increases codebase complexity

**8. Explore Quantization for Model Size Reduction** (OPTIMIZATION)
- **Why**: Reduce 225MB model size for app bundle constraints
- **Confidence**: Low - depends on Candle quantization support
- **How**:
  1. Investigate Candle's quantization capabilities (int8, int4)
  2. Benchmark quality loss vs size reduction
  3. Compare to Python quantization approaches
  4. Document tradeoffs in quality vs size
- **Tradeoffs**: May degrade audio quality, requires extensive validation

---

## Things That Have Been Completed (DO NOT REPEAT)

**Core Implementation (All Verified)**:
- ✅ Tokenization (SentencePiece) - exact match with Python
- ✅ RoPE (interleaved pairs, applied before transpose)
- ✅ FlowLM transformer (6 layers, all architecturally correct)
- ✅ FlowNet (SiLU, AdaLN, time embedding, proper RMSNorm)
- ✅ Voice conditioning (concatenation, two-phase forward pass)
- ✅ Latent generation (random noise with temperature=0.7)
- ✅ Latent denormalization (moved before Mimi decoder)
- ✅ EOS detection (threshold=-4.0, natural detection with min_gen_steps=0)
- ✅ Mimi decoder streaming (full streaming for all Conv1d layers)
- ✅ Replicate padding for first-frame context
- ✅ Non-causal attention in Mimi decoder transformer
- ✅ Upsample ConvTranspose1d with overlap-add state

**Recent Fixes (Session 2026-01-27)**:
- ✅ Random noise enabled (production mode)
- ✅ Removed broken crossfade in streaming
- ✅ Fixed callback EOS handling (returns Continue, not Stop)
- ✅ Natural EOS detection (min_gen_steps=0)
- ✅ API cleanup (removed legacy token-chunked streaming)

**iOS Infrastructure**:
- ✅ UniFFI bindings for Swift
- ✅ XCFramework build system
- ✅ iOS demo app with waveform visualization
- ✅ AB testing infrastructure (decode_latents API, ReferenceTestView)
- ✅ Reference audio generation script

---

## Specific Questions to Investigate

1. **What is the actual WER when validating Rust TTS output with Whisper ASR?**
   - Current validation uses waveform correlation (no longer meaningful with random noise)
   - ASR-based WER would provide objective intelligibility metric
   - Expected WER <5% for production-quality TTS

2. **Does the iOS app achieve >0.95 correlation on real iPhone hardware using the AB test infrastructure?**
   - The AB test infrastructure exists but may not have been tested on device
   - Simulator vs device behavior can differ (especially audio playback)
   - Critical validation step before shipping to users

3. **Can FlowLM transformer be ANE-accelerated via candle-coreml?**
   - Current CPU implementation meets targets, but ANE could provide headroom
   - Need to verify operation compatibility (attention, LayerNorm, etc.)
   - If compatible, could target older devices or reduce battery consumption

4. **What is the minimum iOS device spec for acceptable performance?**
   - Current testing likely on iPhone 15 Pro / recent simulators
   - Should test on iPhone 12/SE (2020) for minimum viable spec
   - Document minimum requirements in README

5. **Can voice embedding caching reduce TTFA in streaming mode?**
   - Voice embeddings processed every synthesis (125 frames through transformer)
   - Caching could reduce first-phase latency
   - Measure impact on ~200ms TTFA target

---

## Useful Links & References

### Official Kyutai Resources
- [Kyutai Pocket TTS Blog Post](https://kyutai.org/blog/2026-01-13-pocket-tts) - Official release announcement
- [Kyutai Pocket TTS Technical Report](https://kyutai.org/pocket-tts-technical-report) - Architecture details
- [GitHub: kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) - Python reference implementation
- [HuggingFace: kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts) - Model card
- [GitHub: kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) - Mimi codec and Moshi dialogue framework

### Community Implementations
- [crates.io: pocket-tts-cli](https://crates.io/crates/pocket-tts-cli) - babybirdprd's Rust/Candle port
- [GitHub: babybirdprd/pocket-tts](https://github.com/babybirdprd/pocket-tts) - Source repository (inferred)

### TTS Quality Validation
- [OpenAI Whisper](https://openai.com/index/whisper/) - ASR for TTS validation
- [GitHub: openai/whisper](https://github.com/openai/whisper) - Whisper repository
- [arXiv: ASR Guided Speech Intelligibility (2006.01463)](https://arxiv.org/abs/2006.01463) - WER methodology
- [Advanced Evaluation Metrics for ASR/TTS](https://apxml.com/courses/speech-recognition-synthesis-asr-tts/chapter-1-modern-speech-processing-foundations/advanced-evaluation-metrics)
- [LXT Speech Data Evaluation](https://www.lxt.ai/services/speech-data-evaluation/)

### Candle & iOS Performance
- [crates.io: candle-coreml](https://crates.io/crates/candle-coreml) - CoreML/ANE integration
- [GitHub: huggingface/candle](https://github.com/huggingface/candle) - Candle ML framework
- [Candle Discussion #2818: Metal Issues](https://github.com/huggingface/candle/discussions/2818)
- [Apple Metal Developer](https://developer.apple.com/metal/) - Metal performance optimization

### Neural Audio Codecs
- [HuggingFace: kyutai/mimi](https://huggingface.co/kyutai/mimi) - Mimi codec model card
- [Kyutai: Neural Audio Codecs Explainer](https://kyutai.org/codec-explainer) - How audio codecs work

### Local Documentation (Already in Repo!)
- `docs/python-reference/README.md` - Comprehensive Python source documentation
- `docs/python-reference/STREAMING/conv-transpose-overlap-add.md` - Streaming algorithms
- `docs/python-reference/ARCHITECTURE/overview.md` - Complete data flow
- `validation/compare_waveforms.py` - Existing waveform comparison tool

---

## Critical Insights

### 1. The "Divergence" Issue Was DEBUG Mode, Not a Bug

The previous research report focused on "hidden state divergence" and "cumulative precision error." This was ultimately a **configuration issue**, not an architectural or numerical precision problem:

- Rust was using `Tensor::zeros()` for FlowNet noise (DEBUG mode for deterministic comparison)
- Python was using `torch.nn.init.normal_()` (production mode)
- Different starting points → different latent trajectories → different audio waveforms
- **Resolution**: Enabled random noise in Rust, shifted validation metrics to amplitude/RMS ratios

**Key lesson**: When porting ML models, ensure EVERY aspect matches production configuration, not just architecture.

### 2. Waveform Correlation Is Not the Right Metric for Random Generation

With random noise enabled:
- Rust RNG produces different random values than Python's `torch.randn()`
- Latent trajectories diverge from step 1 (expected and correct)
- Final audio has same content but different waveform (like two recordings of the same sentence)
- **Correlation ≈0 is expected, not a problem**

**New validation paradigm**:
- Amplitude ratio: 91% (excellent)
- RMS ratio: 98% (near-perfect)
- Listening tests: Intelligible speech
- ASR-based WER: (recommended next step)

### 3. Production-Ready ≠ Perfect Match

The implementation doesn't need to produce bit-identical output to Python to be production-ready:
- Audio is intelligible and natural-sounding
- Performance meets targets (3-4x realtime)
- All unit tests pass
- Streaming works with appropriate latency

**What matters for shipping**:
- Subjective quality (listening tests)
- Objective intelligibility (WER)
- Performance on target hardware
- Reliability (no crashes, no artifacts)

### 4. iOS AB Testing Infrastructure Is Valuable

The `decode_latents` API and ReferenceTestView provide a critical validation path:
- Uses same latents for Rust and Python → eliminates RNG differences
- Runs on actual iOS hardware → validates real-world behavior
- Provides user-accessible testing → enables manual verification

**Recommended**: Run this test on physical iPhone before final release.

---

## Recommended Next Steps

### Immediate (Validation & Quality Assurance)

1. **Add Whisper ASR validation** (1 session):
   - Extend validation/compare_waveforms.py with ASR-based WER
   - Set WER threshold (<5% = pass)
   - Document results in verification report
   - This provides objective intelligibility metric that's RNG-independent

2. **Test iOS AB testing on real hardware** (1 session):
   - Run AB test on iPhone 13/14/15 (not just simulator)
   - Verify >0.95 correlation when using same latents
   - Test AVAudioPlayer playback quality
   - Document results

3. **Add MCD acoustic validation** (1 session):
   - Compute Mel-Cepstral Distortion between Rust/Python outputs
   - Set threshold (MCD <6 dB = acceptable)
   - Provides acoustic similarity metric complementing WER

### Medium-Term (Optimization)

4. **Profile memory usage on lower-end devices** (1-2 sessions):
   - Test on iPhone 12 or SE (2020)
   - Identify memory bottlenecks
   - Document minimum device requirements

5. **Investigate voice embedding caching** (1 session):
   - Cache voice embeddings after first synthesis
   - Measure TTFA improvement
   - Low-hanging fruit for performance improvement

### Long-Term (Exploration)

6. **Explore candle-coreml for ANE acceleration** (2-3 sessions):
   - Experimental, may not yield benefits
   - Only if CPU performance becomes a concern
   - Current implementation already meets targets

7. **Consider custom voice embedding support** (5+ sessions):
   - Major feature, significant complexity
   - Requires encoder implementation or bridge to Python
   - Evaluate user demand before investing

---

## Conclusion

**The Rust/Candle port of Kyutai Pocket TTS is production-ready.** The previous "divergence" blocker was a DEBUG mode configuration issue that has been resolved. The implementation now uses random noise (matching Python production behavior) and produces intelligible, high-quality speech with appropriate audio characteristics.

**Current Status**: ✅ **SHIP-READY**

**Recommended Before Release**:
1. Add Whisper ASR validation for objective intelligibility metric (WER <5%)
2. Test iOS AB testing infrastructure on real iPhone hardware
3. Document minimum device requirements

**Future Enhancements** (post-release):
1. Memory optimization for older devices
2. Voice embedding caching for TTFA improvement
3. Explore ANE acceleration via candle-coreml (if needed)

**Time Estimate**: Validation items (1-2 sessions), then ready for production release.

---

*Previous report archived as research-advisor-report-2.md*
