//! Engine-level integration tests.
//!
//! These construct a real `PocketTTSEngine` and synthesize real speech, per
//! the project testing standard (no synthetic latents). They therefore need
//! model weights on disk; when no model directory is present (e.g. CI), each
//! test prints a notice and passes vacuously rather than failing.
//!
//! These exist to close the worst gaps found in the v0.5.0 coverage audit
//! (docs/TEST_COVERAGE.md): error-variant reachability, cancellation, and the
//! engine surface that no UI exercises.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use pocket_tts_ios::{AudioChunk, PocketTTSEngine, PocketTTSError, TTSEventHandler};

/// First model directory that exists, or None (test skips).
fn model_dir() -> Option<String> {
    for dir in ["kyutai-pocket-ios-en2026-04", "kyutai-pocket-ios"] {
        if std::path::Path::new(dir).join("model.safetensors").exists() {
            return Some(dir.to_string());
        }
    }
    eprintln!("SKIP: no model directory present (kyutai-pocket-ios*)");
    None
}

#[test]
fn engine_surface_reports_sane_metadata() {
    let Some(dir) = model_dir() else { return };
    let engine = PocketTTSEngine::new(dir).expect("engine should load");

    assert!(engine.is_ready());
    assert!(pocket_tts_ios::build_info().contains(&pocket_tts_ios::version()));

    let voices = engine.loaded_voices();
    assert!(!voices.is_empty(), "a loadable model must expose >=1 voice");
    for (i, v) in voices.iter().enumerate() {
        assert_eq!(v.index as usize, i, "voice indices must be bank positions");
        assert!(!v.name.is_empty());
    }
}

#[test]
fn unload_then_synthesize_returns_model_not_loaded() {
    let Some(dir) = model_dir() else { return };
    let engine = PocketTTSEngine::new(dir).expect("engine should load");

    engine.unload();
    assert!(!engine.is_ready(), "unload must clear readiness");

    match engine.synthesize("hello".to_string()) {
        Err(PocketTTSError::ModelNotLoaded) => {},
        Err(other) => panic!("expected ModelNotLoaded, got {other:?}"),
        Ok(_) => panic!("synthesize after unload must fail"),
    }
}

#[test]
fn invalid_voice_index_is_a_loud_error() {
    let Some(dir) = model_dir() else { return };
    let engine = PocketTTSEngine::new(dir).expect("engine should load");

    // Out-of-range index: rejected by config validation before synthesis.
    match engine.synthesize_with_voice("hello".to_string(), 999) {
        Err(PocketTTSError::InvalidConfig(_)) => {},
        Err(other) => panic!("expected InvalidConfig for index 999, got {other:?}"),
        Ok(_) => panic!("an out-of-range voice index must not silently synthesize"),
    }

    // Dangling index (in the canonical 0-7 range but not loaded): must be a
    // loud InvalidVoice, never silent unconditioned synthesis. Only testable
    // when the model dir ships fewer than the canonical 8 voices.
    let n = engine.loaded_voices().len() as u32;
    if n <= 7 {
        match engine.synthesize_with_voice("hello".to_string(), n) {
            Err(PocketTTSError::InvalidVoice(i)) if i == n => {},
            Err(other) => panic!("expected InvalidVoice({n}), got {other:?}"),
            Ok(_) => panic!("a dangling voice index must not silently synthesize"),
        }
    }
}

/// Handler that cancels the engine from the first audio chunk.
struct CancellingHandler {
    engine: Arc<PocketTTSEngine>,
    chunks: Arc<AtomicUsize>,
    completed: Arc<AtomicBool>,
}

impl TTSEventHandler for CancellingHandler {
    fn on_audio_chunk(&self, _chunk: AudioChunk) {
        if self.chunks.fetch_add(1, Ordering::SeqCst) == 0 {
            self.engine.cancel();
        }
    }
    fn on_progress(&self, _progress: f32) {}
    fn on_complete(&self) {
        self.completed.store(true, Ordering::SeqCst);
    }
    fn on_error(&self, _message: String) {}
}

#[test]
fn cancel_stops_streaming_and_engine_stays_usable() {
    let Some(dir) = model_dir() else { return };
    let engine = Arc::new(PocketTTSEngine::new(dir).expect("engine should load"));

    // Long enough that an un-cancelled stream would emit many chunks.
    let text = "The quick brown fox jumps over the lazy dog while reciting a \
                rather long passage about the history of speech synthesis, \
                which continues with many additional words so the stream \
                spans many chunks and cancellation has room to take effect."
        .to_string();

    let chunks = Arc::new(AtomicUsize::new(0));
    let completed = Arc::new(AtomicBool::new(false));
    let handler = CancellingHandler {
        engine: engine.clone(),
        chunks: chunks.clone(),
        completed: completed.clone(),
    };

    engine
        .start_true_streaming(text, Box::new(handler))
        .expect("a cancelled stream is not an error");

    // Cancelled on chunk 1 → the flag check stops the stream before chunk 2.
    assert_eq!(
        chunks.load(Ordering::SeqCst),
        1,
        "cancel() must stop the stream at the next chunk boundary"
    );
    assert!(
        !completed.load(Ordering::SeqCst),
        "a cancelled stream must not report completion"
    );

    // The engine must remain fully usable after a cancelled stream.
    let result = engine
        .synthesize("Still working after cancellation.".to_string())
        .expect("engine must survive cancellation");
    assert!(!result.audio_data.is_empty());
}
