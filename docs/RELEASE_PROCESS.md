# Release Process

This document describes how to create a new release of Pocket TTS iOS.

## Overview

Releases are automated via GitHub Actions. When you push a version tag, the workflow:
1. Validates the tag matches Cargo.toml version
2. Builds the XCFramework for iOS device and simulator
3. Packages artifacts into a release zip
4. Creates a GitHub Release with the artifacts attached

## Prerequisites

Before creating a release:
- [ ] All tests passing on main branch
- [ ] Correlation gate re-run and raw artifact saved (see Step 3)
- [ ] CHANGELOG.md updated with release notes
- [ ] Version bumped in Cargo.toml
- [ ] All changes committed and pushed to main

## Creating a Release

### Step 1: Update Version

Edit `Cargo.toml` and update the version:

```toml
[package]
name = "pocket-tts-ios"
version = "X.Y.Z"  # Update this
```

### Step 2: Update CHANGELOG

Move items from `[Unreleased]` to a new version section in `CHANGELOG.md`:

```markdown
## [Unreleased]

## [X.Y.Z] - YYYY-MM-DD

### Added
- New feature description

### Fixed
- Bug fix description

### Changed
- Change description
```

### Step 3: Run the Correlation Gate

Before tagging, re-run the noise-matched correlation gate and archive the raw
output:

```bash
# Host parity gate — must read 1.000000 on all 4 phrases for BOTH v1 and v2
.claude/skills/verify/run_baseline.sh

# Save the raw artifact alongside the release (committed)
cp <gate output> docs/audit/correlation-vX.Y.Z-YYYY-MM-DD.txt
```

A release does not ship unless the gate reads 1.000000 × 4 phrases × both
models. The saved artifact (e.g.
`docs/audit/correlation-v0.5.0-2026-06-11.txt`) is the auditable record for
that release. Also verify the iOS demo's Compare tab reads 1.0000 on all 4
phrases on-device.

### Step 4: Commit Version Bump

```bash
git add Cargo.toml Cargo.lock CHANGELOG.md docs/audit/correlation-v*.txt
git commit -m "chore: prepare release vX.Y.Z"
git push origin main
```

### Step 5: Create and Push Tag

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Or push tag with the commit:

```bash
git push origin main --tags
```

### Step 6: Verify Release

1. Go to GitHub Actions and watch the release workflow
2. Once complete, check the Releases page
3. Verify the zip file is attached and contains expected files (including v1
   weights in `Models/` — and **no v2 weights**, see Model Weights Policy)
4. Test downloading and integrating in a sample iOS project

## Manual Release (workflow_dispatch)

You can also trigger a release manually from the GitHub Actions UI:

1. Go to Actions → Release workflow
2. Click "Run workflow"
3. Enter the version number (without 'v' prefix)
4. Click "Run workflow"

This is useful for testing the release process without creating a tag.

## Versioning Policy

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0): Breaking API changes
  - Removed or renamed public types/functions
  - Changed function signatures
  - Incompatible UDL interface changes

- **MINOR** (0.X.0): New features, backward compatible
  - New voices added
  - New configuration options
  - Performance improvements
  - New Swift wrapper features

- **PATCH** (0.0.X): Bug fixes, backward compatible
  - Audio quality fixes
  - Crash fixes
  - Documentation updates

### Pre-release Versions

For beta/RC releases, use suffixes:
- `v0.4.0-beta.1`
- `v0.4.0-rc.1`

These will be marked as pre-releases on GitHub.

## Release Artifacts

Each release includes:

```
PocketTTS-vX.Y.Z.zip
├── PocketTTS.xcframework/     # iOS framework (device + simulator)
├── Sources/
│   ├── pocket_tts_ios.swift   # UniFFI-generated bindings
│   └── PocketTTSSwift.swift   # High-level Swift wrapper
├── Models/                     # v1 model weights ONLY (see policy below)
│   ├── model.safetensors
│   ├── tokenizer.model
│   └── voices/
├── LICENSE
├── README.md                   # Integration guide
└── CHANGELOG.md
```

Plus a `.sha256` checksum file.

## Model Weights Policy

**v2 weights are NEVER bundled in release artifacts.** The v2 models
(`english_2026-04` and later) are distributed from the **gated** Hugging Face
repo `kyutai/pocket-tts`. Redistributing them in our release zips would
bypass Kyutai's gate, so the policy is:

- The release zip bundles **v1 weights only** (`english_2026-01`, from the
  public `kyutai/pocket-tts-without-voice-cloning` repo) — as it always has.
- Every v2 user downloads the weights directly from Hugging Face after
  accepting the gate.
- Before publishing, double-check the zip contains no v2 `model.safetensors`
  or v2 voice files.

**Release notes must include the v2 download instructions**, since they are
not in the zip:

```bash
# v2 (english_2026-04) — accept the gate at
# https://huggingface.co/kyutai/pocket-tts, then:
huggingface-cli login   # or HF_TOKEN=hf_...
python scripts/download-model.py --model v2
```

## Troubleshooting

### Tag version doesn't match Cargo.toml

The release workflow validates that the tag version matches Cargo.toml. If you see this error:
1. Update Cargo.toml to match the tag
2. Delete the tag: `git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z`
3. Commit the fix and create the tag again

### Build fails on macOS

Check the GitHub Actions logs. Common issues:
- Rust target not installed (should be automatic)
- XCFramework creation fails (check xcodebuild output)

### Manual local build

To test the build locally:

```bash
./scripts/build-ios.sh
./scripts/package-release.sh X.Y.Z
ls -la release/
```

## Post-Release

After a successful release:
1. Notify users/dependents of the new version
2. Update any documentation referencing specific versions
3. Create a new `[Unreleased]` section in CHANGELOG.md for future changes
