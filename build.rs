//! Build script for pocket-tts-ios
//!
//! Generates UniFFI scaffolding from the UDL file.

fn main() {
    uniffi::generate_scaffolding("src/pocket_tts.udl").unwrap();
}
