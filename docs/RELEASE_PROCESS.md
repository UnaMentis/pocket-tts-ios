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

### Step 3: Commit Version Bump

```bash
git add Cargo.toml Cargo.lock CHANGELOG.md
git commit -m "chore: prepare release vX.Y.Z"
git push origin main
```

### Step 4: Create and Push Tag

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Or push tag with the commit:

```bash
git push origin main --tags
```

### Step 5: Verify Release

1. Go to GitHub Actions and watch the release workflow
2. Once complete, check the Releases page
3. Verify the zip file is attached and contains expected files
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
├── LICENSE
├── README.md                   # Integration guide
└── CHANGELOG.md
```

Plus a `.sha256` checksum file.

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
