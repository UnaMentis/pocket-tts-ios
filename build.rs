//! Build script for pocket-tts-ios
//!
//! Generates `UniFFI` scaffolding from the UDL file.

// Build scripts should panic with clear messages on failure
#![allow(clippy::expect_used)]

fn main() {
    uniffi::generate_scaffolding("src/pocket_tts.udl").expect("Failed to generate UniFFI scaffolding");
}
