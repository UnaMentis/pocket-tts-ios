//! Build script for pocket-tts-ios
//!
//! Generates `UniFFI` scaffolding from the UDL file and stamps build
//! provenance (git SHA, dirty flag, build time) into the binary so any
//! built artifact can identify the exact commit it came from at runtime
//! via `build_info()`.

// Build scripts should panic with clear messages on failure
#![allow(clippy::expect_used)]

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn main() {
    uniffi::generate_scaffolding("src/pocket_tts.udl").expect("Failed to generate UniFFI scaffolding");

    // Provenance stamp. "unknown" (not a failure) outside a git checkout,
    // e.g. building from a source tarball.
    let sha = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = match git(&["status", "--porcelain"]) {
        Some(_) => "-dirty",            // non-empty porcelain output → uncommitted changes
        None if sha == "unknown" => "", // not a git checkout; no dirty concept
        None => "",                     // clean tree (empty output is mapped to None by git())
    };
    let describe = git(&["describe", "--tags", "--always", "--dirty"]).unwrap_or_else(|| "unknown".into());
    let build_time = Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());

    println!("cargo:rustc-env=POCKET_TTS_GIT_SHA={sha}{dirty}");
    println!("cargo:rustc-env=POCKET_TTS_GIT_DESCRIBE={describe}");
    println!("cargo:rustc-env=POCKET_TTS_BUILD_TIME={build_time}");

    // Re-stamp when the checked-out commit changes.
    if std::path::Path::new(".git/HEAD").exists() {
        println!("cargo:rerun-if-changed=.git/HEAD");
        println!("cargo:rerun-if-changed=.git/index");
    }
}
