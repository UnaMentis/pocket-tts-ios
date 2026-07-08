# Release Checklist — Pocket TTS iOS

A hard-gate checklist for cutting a release. Every gate below is mandatory; a
failure at any gate **stops the release** until resolved. Check items off in
order.

> This checklist exists because **v0.4.x shipped simulator-validated and
> produced zero audio on real hardware.** The consuming app logged ~112 errors
> and no audio on a physical iPhone while the same code produced audio in the
> Simulator. See [../APP-SIDE-FINDINGS.md](../APP-SIDE-FINDINGS.md). The
> physical-device gate below is the direct countermeasure.

## 1. Pre-tag gates (on the release branch)

- [ ] `cargo fmt --check` is clean.
- [ ] `cargo clippy -- -D warnings` is clean.
- [ ] `cargo test` is green.
- [ ] Correlation gate artifact **regenerated** and archived in `docs/audit/`
      (e.g. `docs/audit/correlation-vX.Y.Z-<date>.txt`); confirm the numbers
      still meet the noise-matched bar (1.000 on the canonical phrases).
- [ ] Swift bindings regenerated (`./scripts/build-ios.sh`), and **every code
      example in the docs verified against the freshly generated
      `pocket_tts_ios.swift`** — type names (`PocketTtsEngine`, `TtsConfig`,
      `SynthesisResult`, `AudioChunk`, `PocketTtsError`), initializer field
      lists, and method names (`startTrueStreaming`, `synthesize`, ...). Docs to
      check: `README.md`, `docs/INTEGRATION.md`, `tests/ios-harness/README.md`.
- [ ] `CHANGELOG.md` has a dated entry for this version.
- [ ] Version bumped consistently (Cargo, docs, tag name agree).

## 2. MANDATORY PHYSICAL-DEVICE GATE (real iPhone, not the Simulator)

This gate is non-negotiable and cannot be satisfied by the Simulator.

- [ ] Build the demo app (`tests/ios-harness/PocketTTSDemo`) for a **physical
      iPhone** and install it on the device.
- [ ] Run **engine load** on device — confirm the model loads without error.
- [ ] Run **synthesize** on device — confirm non-empty audio actually plays.
- [ ] Run the **Compare tab noise-matched gate** on device and confirm it reads
      the expected correlation (≈1.0000 on the canonical phrases).
- [ ] Record, in the release notes / audit log: **device model, iOS version,
      and result** for each check above.
- [ ] **On ANY failure:** capture the **exact `PocketTtsError` message string**
      (case + payload) before doing anything else. Do not proceed, do not tag,
      until the failure is understood. The exact message is the single most
      useful diagnostic — a prior device-only failure was undiagnosable for
      weeks precisely because it was never captured.

## 3. Tag & publish

- [ ] Merge the release branch to `main`; confirm `main` CI is **green**.
- [ ] Tag **only a green `main`** at the release commit.
- [ ] Let **CI build the release artifact from the tag** (never publish a local
      build).
- [ ] Verify the published zip:
  - [ ] `unzip` it and inspect the contents.
  - [ ] Check `BUILD_INFO.txt` — its provenance stamp (commit SHA) **matches the
        tag's SHA**.
  - [ ] Spot-check that the bundled Swift wrapper compiles against the bundled
        bindings.
  - [ ] Confirm CC-BY-4.0 attribution for the bundled model weights is present
        (see [../ATTRIBUTION.md](../ATTRIBUTION.md)).

## 4. Post-publish

- [ ] Hand the **CI-built zip** (never a local build) to consuming apps.
- [ ] Instruct the consuming app to log `build_info()` at engine init, so the
      exact artifact provenance is recorded on the app side from day one.

---

> **Note:** `BUILD_INFO.txt` and the `build_info()` binding are being added
> concurrently by other engineers; this checklist assumes they exist. If a gate
> references something not yet present, treat its absence as a blocker, not a
> reason to skip the gate.
