# Release Pocket TTS iOS

Create a new release of the Pocket TTS iOS XCFramework.

## Arguments

- `$ARGUMENTS` - Optional version number (e.g., "0.5.0"). If not provided, you will suggest the next version.

## Author Information

- **Git Name:** UnaMentis (Richard Amerman)
- **Git Email:** richard@unamentis.org

Always use these for any git operations.

## Instructions

You are creating a release for the Pocket TTS iOS XCFramework. Follow these steps:

### Step 1: Determine Version

1. Read the current version from `Cargo.toml`
2. If `$ARGUMENTS` contains a version number, use that
3. If no version provided, ask the user which version bump they want:
   - **Patch** (0.4.0 → 0.4.1): Bug fixes only
   - **Minor** (0.4.0 → 0.5.0): New features, backward compatible
   - **Major** (0.4.0 → 1.0.0): Breaking changes

   Suggest the most likely option based on recent commits.

### Step 2: Update Version Files

1. Update `version` in `Cargo.toml` to the new version
2. Run `cargo check` to update `Cargo.lock`
3. Add a new section to `CHANGELOG.md`:
   - Move any items from `[Unreleased]` to the new version section
   - If `[Unreleased]` is empty, check recent commits with `git log` and summarize changes
   - Use today's date in YYYY-MM-DD format

### Step 3: Format and Verify

1. Run `cargo fmt` to fix any formatting issues
2. Run `cargo check` to verify compilation
3. Run `cargo clippy` to check for warnings

### Step 4: Build Release Artifacts

1. Run `./scripts/build-ios.sh` to build the XCFramework
2. Run `./scripts/package-release.sh <version>` to create the release zip
3. Verify the release package was created in `release/`

### Step 5: Commit and Tag

1. Stage all release-related files:
   ```
   git add Cargo.toml Cargo.lock CHANGELOG.md
   ```
2. Commit with message:
   ```
   chore: release v<version>
   ```
3. Create an annotated tag:
   ```
   git tag -a v<version> -m "Release v<version>"
   ```

### Step 6: Push Release

1. Push the commit: `git push origin main`
2. Push the tag: `git push origin v<version>`

### Step 7: Verify

1. Confirm the GitHub Actions release workflow has started
2. Provide the user with:
   - Link to the Actions run
   - Link to where the release will appear once complete

## Output

After completing the release, summarize:
- Version released
- Files changed
- Tag created
- Link to GitHub release page: https://github.com/UnaMentis/pocket-tts-ios/releases

## Error Handling

- If any step fails, stop and report the error
- Do not push if build or tests fail
- If the tag already exists, ask the user how to proceed
