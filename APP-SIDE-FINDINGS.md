# Pocket TTS iOS — findings from the UnaMentis app side (handoff for the v0.5.0 release)

> Provenance: this report was written 2026-07-02 by the Claude session working in the
> UnaMentis app repo, which diagnosed the on-device TTS failure from the consuming-app
> side. It was recovered from that session's transcript on 2026-07-07 and saved here
> verbatim so the port repo has the app-side evidence on record.

You're in the canonical port repo (`UnaMentis/pocket-tts-ios`, local `/Users/ramerman/dev/pocket-tts-ios`). This is what the consuming app (UnaMentis iOS) observed about on-device behavior. Facts and speculation are separated deliberately; treat anything under "speculation" as unproven.

**The problem in one line:** On a physical iPhone the app got no audio from Pocket TTS; in the iOS Simulator the same engine plus app code produce audio. So the gap is real-device *execution*, not the app's integration.

**Verified (facts, with how I know):**
- In the iOS **Simulator** (arm64), the engine loads and produces **non-empty** audio. An automated test in the app calls the engine load path and `synthesize()` and asserts non-empty audio bytes; it passes. (I verified non-empty output, not correctness/intelligibility. The Simulator runs the `ios-arm64-simulator` slice, a separately compiled binary from the device slice.)
- The app's integration code path is **identical** for Simulator and device. With the point above, that's strong evidence the app-side usage is not the cause.
- A device build's app bundle was inspected and **contained the model files** (model.safetensors ~225MB, tokenizer.model, voices). "Models missing on device" does not appear to be the cause.
- The xcframework is **linked for device arm64** (the `ios-arm64` slice with FFI symbols is present).
- In this repo: weights load as **`DType::F32`** via `from_mmaped_safetensors` (v0.4.1 and main). The `validation/` suite is entirely **host-side numerical correctness** (Rust vs Python waveform correlation). The README's iPhone real-time factor is labeled **"estimated."** I did **not** find real-device execution validation, and did **not** find sandbox-sensitive operations in the Rust (no temp-dir writes, env vars, spawned threads, hardcoded paths).
- The app currently consumes an xcframework dated **Apr 9 2026**, which is *after* v0.4.1 (Jan 27) and *before* v0.5.0 (Jun 10).

**Reported (not observed by me directly):**
- The app owner reports ~112 errors logged in one short on-device session, with no audio.

**Speculation / hunches (unproven):**
- Best guess: the failure is thrown at engine **load**, not during streaming (inferred from how errors accumulate in the app's code path). I have **not** seen the actual device error.
- I earlier floated device **memory** (F32 load, ~470MB resident from a ~235MB F16 file), but I'm **not confident** — a 12GB iPhone likely has headroom. Weak hunch, not a finding.
- **[inference]** The Apr 9 xcframework is likely a **mid-development build**, not a clean tag. I can't pin the exact commit from the file date.
- I could **not** identify the specific device-only cause from the source: could be engine load, tokenizer load, or first inference. Unknown.

**Single most useful next data point:** the actual device error string (one of `ModelLoadFailed` / `InferenceFailed` / `TokenizationFailed`, plus its message), or a stderr/log capture from running this repo's `test_tts` binary on a **physical iPhone**. That turns the guesses into a concrete pointer.

**Recommendations for v0.5.0:**
1. Add a minimal **real-device smoke test** (load + synthesize on a physical iPhone), not just host-side correlation. That closes the exact gap that let a device-only failure ship.
2. Cut a clean **v0.5.0 tag** and **rebuild the xcframework from that tag**, then hand that to the app. Removes "which commit is the app running" as a variable.
3. Before cutting: `PORTING_STATUS.md` shows waveform correlation ~0.013 on `main` at one point versus ~0.98 in release validation. Cut from a **known-good-correlation** commit.

**Bottom line:** the fix belongs here in the port, not in the app's usage. The most efficient unblock is a real-device run that surfaces the actual load error.
