//! Complete Pocket TTS Model
//!
//! Combines FlowLM transformer (with FlowNet) and Mimi decoder
//! into a complete text-to-speech pipeline.
//!
//! Portions of this file derived from:
//! https://github.com/babybirdprd/pocket-tts
//! Licensed under MIT

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::flowlm::{FlowLM, FlowLMConfig, LatentStreamControl};
use super::mimi::{MimiConfig, MimiDecoder};
use crate::config::TTSConfig;
use crate::error::PocketTTSError;
use crate::modules::embeddings::{VoiceBank, VoiceEmbedding};
use crate::tokenizer::PocketTokenizer;

/// Complete Pocket TTS Model
pub struct PocketTTSModel {
    flowlm: FlowLM,
    mimi: MimiDecoder,
    tokenizer: PocketTokenizer,
    voice_bank: VoiceBank,
    device: Device,
    config: TTSConfig,
    custom_voice: Option<VoiceEmbedding>,
    /// Pre-loaded noise tensors for correlation testing.
    /// When set, these are used instead of random noise during FlowNet generation.
    /// Each tensor corresponds to one autoregressive step.
    noise_tensors: Option<Vec<Tensor>>,
}

impl PocketTTSModel {
    /// Load model from directory containing all components
    pub fn load<P: AsRef<Path>>(model_dir: P, device: &Device) -> std::result::Result<Self, PocketTTSError> {
        let model_dir = model_dir.as_ref();

        // Load model weights using memory-mapped file
        let model_path = model_dir.join("model.safetensors");

        // Create VarBuilder from safetensors file
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F32, device)
                .map_err(|e| PocketTTSError::ModelLoadFailed(e.to_string()))?
        };

        // Load tokenizer (SentencePiece .model format)
        let tokenizer_path = model_dir.join("tokenizer.model");
        let tokenizer = PocketTokenizer::from_file(&tokenizer_path)?;

        // Load voice embeddings
        let voices_dir = model_dir.join("voices");
        let voice_bank = VoiceBank::load_from_dir(&voices_dir, device)
            .map_err(|e| PocketTTSError::ModelLoadFailed(format!("Failed to load voices: {}", e)))?;

        // Initialize model components
        let flowlm_config = FlowLMConfig::default();
        let flowlm = FlowLM::new(flowlm_config.clone(), vb.pp("flow_lm"), device)
            .map_err(|e| PocketTTSError::ModelLoadFailed(format!("FlowLM: {}", e)))?;

        let mimi_config = MimiConfig {
            latent_dim: flowlm_config.latent_dim,
            ..MimiConfig::default()
        };
        let mimi = MimiDecoder::new(mimi_config, vb.pp("mimi"))
            .map_err(|e| PocketTTSError::ModelLoadFailed(format!("Mimi: {}", e)))?;

        Ok(Self {
            flowlm,
            mimi,
            tokenizer,
            voice_bank,
            device: device.clone(),
            config: TTSConfig::default(),
            custom_voice: None,
            noise_tensors: None,
        })
    }

    /// Configure synthesis parameters
    pub fn configure(&mut self, config: TTSConfig) -> std::result::Result<(), PocketTTSError> {
        config.validate().map_err(PocketTTSError::InvalidConfig)?;
        self.config = config;
        Ok(())
    }

    /// Set custom voice from reference audio embedding
    pub fn set_custom_voice(&mut self, embedding: VoiceEmbedding) {
        self.custom_voice = Some(embedding);
    }

    /// Clear custom voice (use built-in)
    pub fn clear_custom_voice(&mut self) {
        self.custom_voice = None;
    }

    /// Load pre-captured noise tensors from a directory for correlation testing.
    ///
    /// The directory should contain .npy files named like:
    /// `phrase_XX_noise_step_000.npy`, `phrase_XX_noise_step_001.npy`, etc.
    ///
    /// These tensors are used instead of random noise during FlowNet generation,
    /// eliminating RNG differences between Python and Rust for correlation measurement.
    pub fn load_noise_tensors(&mut self, noise_dir: &Path, phrase_id: &str) -> std::result::Result<usize, PocketTTSError> {
        let mut tensors = Vec::new();
        let mut step = 0;

        loop {
            let npy_path = noise_dir.join(format!("{}_noise_step_{:03}.npy", phrase_id, step));
            if !npy_path.exists() {
                break;
            }

            // Load .npy file
            let data = std::fs::read(&npy_path)
                .map_err(|e| PocketTTSError::ModelLoadFailed(format!("Failed to read noise file {:?}: {}", npy_path, e)))?;

            // Parse .npy header to get shape and find data offset
            let tensor = Self::parse_npy_f32(&data, &self.device)
                .map_err(|e| PocketTTSError::ModelLoadFailed(format!("Failed to parse noise file {:?}: {}", npy_path, e)))?;

            tensors.push(tensor);
            step += 1;
        }

        let count = tensors.len();
        if count > 0 {
            eprintln!("[PocketTTS] Loaded {} noise tensors from {:?} for {}", count, noise_dir, phrase_id);
            self.noise_tensors = Some(tensors);
        } else {
            eprintln!("[PocketTTS] WARNING: No noise tensors found for {} in {:?}", phrase_id, noise_dir);
            self.noise_tensors = None;
        }
        Ok(count)
    }

    /// Clear pre-loaded noise tensors
    pub fn clear_noise_tensors(&mut self) {
        self.noise_tensors = None;
    }

    /// Parse a NumPy .npy file containing float32 data into a Tensor
    fn parse_npy_f32(data: &[u8], device: &Device) -> std::result::Result<Tensor, String> {
        // Validate magic number
        if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
            return Err("Invalid .npy magic number".to_string());
        }

        let major = data[6];
        let _minor = data[7];

        // Get header length
        let header_len = if major == 1 {
            u16::from_le_bytes([data[8], data[9]]) as usize
        } else if major == 2 {
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
        } else {
            return Err(format!("Unsupported .npy version {}", major));
        };

        let header_start = if major == 1 { 10 } else { 12 };
        let data_start = header_start + header_len;

        // Parse header to extract shape
        let header_str = std::str::from_utf8(&data[header_start..data_start])
            .map_err(|e| format!("Invalid header UTF-8: {}", e))?;

        // Extract shape from header dict, e.g., "'shape': (1, 1, 32)"
        let shape = Self::parse_npy_shape(header_str)?;

        // Read float32 data
        let float_data = &data[data_start..];
        let n_floats = float_data.len() / 4;
        let values: Vec<f32> = (0..n_floats)
            .map(|i| f32::from_le_bytes([
                float_data[i * 4],
                float_data[i * 4 + 1],
                float_data[i * 4 + 2],
                float_data[i * 4 + 3],
            ]))
            .collect();

        Tensor::from_vec(values, shape.as_slice(), device)
            .map_err(|e| format!("Tensor creation failed: {}", e))
    }

    /// Parse shape tuple from .npy header string
    fn parse_npy_shape(header: &str) -> std::result::Result<Vec<usize>, String> {
        // Find 'shape': (...) in the header dict
        let shape_start = header.find("'shape':")
            .or_else(|| header.find("\"shape\":"))
            .ok_or("No 'shape' field in .npy header")?;

        let paren_start = header[shape_start..].find('(')
            .ok_or("No '(' in shape")? + shape_start;
        let paren_end = header[paren_start..].find(')')
            .ok_or("No ')' in shape")? + paren_start;

        let shape_str = &header[paren_start + 1..paren_end];
        let dims: Vec<usize> = shape_str
            .split(',')
            .filter_map(|s| {
                let trimmed = s.trim();
                if trimmed.is_empty() { None } else { trimmed.parse().ok() }
            })
            .collect();

        if dims.is_empty() {
            return Err("Empty shape".to_string());
        }

        Ok(dims)
    }

    /// Synthesize text to audio (SYNC MODE)
    ///
    /// **NOTE: Not for current on-device use.** Latency is king for on-device TTS.
    /// Use [`synthesize_true_streaming`] instead, which provides audio chunks
    /// via callback with ~200ms Time To First Audio (TTFA).
    ///
    /// This sync mode exists for:
    /// - Reference/debugging purposes
    /// - Batch processing scenarios where latency doesn't matter
    /// - Comparison testing against streaming mode
    ///
    /// The sync mode processes all tokens at once and returns complete audio,
    /// which has higher latency but uses NON-CAUSAL transformer attention for
    /// potentially different audio characteristics.
    pub fn synthesize(&mut self, text: &str) -> std::result::Result<Vec<f32>, PocketTTSError> {
        eprintln!("[PocketTTS] synthesize called with text len: {}", text.len());

        // Tokenize text
        let token_ids = self.tokenizer.encode(text)?;
        eprintln!("[PocketTTS] tokenized to {} tokens: {:?}", token_ids.len(), token_ids);

        // Create tensor
        let token_tensor = Tensor::from_vec(
            token_ids.iter().map(|&id| id as i64).collect::<Vec<_>>(),
            (1, token_ids.len()),
            &self.device,
        )
        .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
        eprintln!("[PocketTTS] token tensor shape: {:?}", token_tensor.dims());

        // Get voice embedding
        let voice = if let Some(ref custom) = self.custom_voice {
            Some(custom)
        } else {
            self.voice_bank.get(self.config.voice_index as usize)
        };
        eprintln!("[PocketTTS] voice embedding loaded: {}", voice.is_some());

        // Reset caches for new sequence
        self.flowlm.reset_cache();

        // Generate latents with FlowLM + FlowNet
        // Reference implementation uses lsd_decode_steps = 1 (consistency model)
        // Single step is sufficient as the model is trained with consistency distillation
        let num_flow_steps = 1;
        eprintln!(
            "[PocketTTS] generating latents with {} flow step (consistency model)",
            num_flow_steps
        );
        let seed = if self.config.use_fixed_seed { Some(self.config.seed as u64) } else { None };
        let noise_ref = self.noise_tensors.as_deref();
        let latents = self
            .flowlm
            .generate_latents(&token_tensor, voice, num_flow_steps, self.config.temperature, seed, noise_ref)
            .map_err(|e| PocketTTSError::InferenceFailed(format!("FlowLM: {}", e)))?;
        eprintln!("[PocketTTS] latents shape: {:?}", latents.dims());

        // DIAGNOSTIC: Log latent statistics to verify FlowLM output quality
        let latents_flat: Vec<f32> = latents
            .flatten_all()
            .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?
            .to_vec1()
            .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
        let lat_mean = latents_flat.iter().sum::<f32>() / latents_flat.len() as f32;
        let lat_max = latents_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lat_min = latents_flat.iter().cloned().fold(f32::INFINITY, f32::min);
        let lat_std =
            (latents_flat.iter().map(|x| (x - lat_mean).powi(2)).sum::<f32>() / latents_flat.len() as f32).sqrt();
        eprintln!(
            "[PocketTTS] latent stats: mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
            lat_mean, lat_std, lat_min, lat_max
        );

        // Denormalize latents before passing to Mimi
        // Python: mimi_decoding_input = latent * emb_std + emb_mean
        let latents = self
            .flowlm
            .denormalize_latents(&latents)
            .map_err(|e| PocketTTSError::InferenceFailed(format!("Denormalize: {}", e)))?;

        // Decode to audio using streaming mode with batch transformer
        // IMPORTANT: The decoder transformer uses NON-CAUSAL attention, so it must
        // process the full sequence at once (not frame-by-frame with KV cache)
        eprintln!("[PocketTTS] decoding with Mimi (batch transformer, streaming SEANet)...");
        let audio = self
            .mimi
            .forward_streaming(&latents)
            .map_err(|e| PocketTTSError::InferenceFailed(format!("Mimi: {}", e)))?;
        eprintln!("[PocketTTS] audio tensor shape: {:?}", audio.dims());

        // Note: Amplitude scaling is no longer needed after fixing SEANet to use batch mode
        // which produces correct amplitude (~0.4-0.5) matching Python's streaming output

        // Convert to Vec<f32>
        let audio = audio.squeeze(0).map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
        let audio_vec: Vec<f32> = audio.to_vec1().map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

        // Debug: check amplitude to verify audio has signal
        let audio_max = audio_vec.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let audio_mean = audio_vec.iter().map(|s| s.abs()).sum::<f32>() / audio_vec.len() as f32;
        eprintln!(
            "[PocketTTS] final audio samples: {} (expect ~78720 for test phrase)",
            audio_vec.len()
        );
        eprintln!("[PocketTTS] audio max amplitude: {:.4} (expect > 0.01)", audio_max);
        eprintln!("[PocketTTS] audio mean amplitude: {:.4}", audio_mean);

        Ok(audio_vec)
    }

    /// True streaming synthesis - yields audio as soon as possible (PREFERRED METHOD)
    ///
    /// **This is the PREFERRED method for on-device TTS.** Latency is king - this
    /// method achieves ~200ms Time To First Audio (TTFA), delivering audio to the
    /// user as quickly as possible.
    ///
    /// Unlike `synthesize_streaming` which chunks by tokens, this method:
    /// 1. Tokenizes ALL text upfront
    /// 2. Generates latents one at a time (streaming FlowLM)
    /// 3. Decodes and yields audio in small batches DURING generation
    ///
    /// This achieves minimum TTFA by starting audio output as soon as
    /// the first batch of latents is generated.
    ///
    /// The callback receives:
    /// - `samples`: Audio samples for this chunk
    /// - `is_final`: Whether this is the last chunk
    ///
    /// Returns `false` from the callback to stop early.
    pub fn synthesize_true_streaming<F>(
        &mut self,
        text: &str,
        mut chunk_callback: F,
    ) -> std::result::Result<(), PocketTTSError>
    where
        F: FnMut(&[f32], bool) -> bool,
    {
        use std::cell::RefCell;

        // Tokenize ALL text upfront (not chunked)
        let token_ids = self.tokenizer.encode(text)?;

        // Create tensor for all tokens
        let token_tensor = Tensor::from_vec(
            token_ids.iter().map(|&id| id as i64).collect::<Vec<_>>(),
            (1, token_ids.len()),
            &self.device,
        )
        .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

        // Get voice embedding
        let voice = if let Some(ref custom) = self.custom_voice {
            Some(custom)
        } else {
            self.voice_bank.get(self.config.voice_index as usize)
        };

        // Pre-extract normalization tensors (to avoid borrowing self.flowlm in callback)
        let emb_mean = self.flowlm.emb_mean().clone();
        let emb_std = self.flowlm.emb_std().clone();

        // Reset caches for new sequence
        self.flowlm.reset_cache();
        self.mimi.reset_decoder_cache();

        let num_flow_steps = self.config.consistency_steps.max(1) as usize;
        let seed = if self.config.use_fixed_seed { Some(self.config.seed as u64) } else { None };

        // Decode batch size - how many latents to accumulate before decoding
        // Smaller = lower TTFA, larger = more efficient
        // 4 latents ≈ 320ms of audio, good balance for TTFA
        let decode_batch_size = 4;

        // Initialize persistent Mimi streaming state ONCE before streaming starts.
        // This state MUST persist across ALL decode batches for proper overlap-add.
        // Per Python reference: streaming ConvTranspose1d maintains partial buffers
        // that accumulate overlapping output contributions between frames.
        let mimi_state = self
            .mimi
            .init_streaming_state(1, &self.device)
            .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
        let mimi_state = RefCell::new(mimi_state);

        // SAFETY: We use raw pointer to access mimi from within the callback.
        // This is safe because:
        // 1. flowlm and mimi are separate fields with no shared state
        // 2. flowlm is borrowed mutably for generate_latents_streaming
        // 3. mimi is accessed mutably in the callback
        // 4. These are non-overlapping borrows of different struct fields
        // Rust's borrow checker can't prove this is safe, but we know it is.
        let mimi_ptr = &mut self.mimi as *mut MimiDecoder;

        // State for the callback
        let current_batch: RefCell<Vec<Tensor>> = RefCell::new(Vec::new());
        let should_continue = RefCell::new(true);
        let callback_error: RefCell<Option<PocketTTSError>> = RefCell::new(None);

        // NOTE: No crossfade needed - Mimi's streaming ConvTranspose1d state
        // handles overlap-add internally via partial buffers that persist
        // across frames. Adding crossfade would blend DIFFERENT audio content
        // (end of chunk A with start of chunk B) causing artifacts.

        // Generate latents with streaming callback that decodes INLINE
        let result = self.flowlm.generate_latents_streaming(
            &token_tensor,
            voice,
            num_flow_steps,
            self.config.temperature,
            seed,
            None, // noise_tensors: not used in production streaming
            |latent: &Tensor, _step: usize, is_eos: bool| {
                if !*should_continue.borrow() {
                    return LatentStreamControl::Stop;
                }

                // Add latent to current batch
                current_batch.borrow_mut().push(latent.clone());

                // Decode when batch is full or at EOS
                let batch_ready = current_batch.borrow().len() >= decode_batch_size || is_eos;

                if batch_ready && !current_batch.borrow().is_empty() {
                    let batch: Vec<Tensor> = current_batch.borrow_mut().drain(..).collect();

                    // Concatenate batch latents: [1, n, 32]
                    let latents_batch = match Tensor::cat(&batch, 1) {
                        Ok(t) => t,
                        Err(e) => {
                            *callback_error.borrow_mut() = Some(PocketTTSError::InferenceFailed(e.to_string()));
                            *should_continue.borrow_mut() = false;
                            return LatentStreamControl::Stop;
                        },
                    };

                    // Denormalize latents before Mimi
                    let denormalized =
                        match latents_batch.broadcast_mul(&emb_std).and_then(|t| t.broadcast_add(&emb_mean)) {
                            Ok(d) => d,
                            Err(e) => {
                                *callback_error.borrow_mut() = Some(PocketTTSError::InferenceFailed(e.to_string()));
                                *should_continue.borrow_mut() = false;
                                return LatentStreamControl::Stop;
                            },
                        };

                    // SAFETY: Access mimi through raw pointer - see safety comment above
                    let mimi = unsafe { &mut *mimi_ptr };

                    // Decode using TRUE streaming with persistent state
                    // This maintains ConvTranspose1d partial buffers across batches
                    let audio = match mimi.forward_streaming_stateful(&denormalized, &mut mimi_state.borrow_mut()) {
                        Ok(a) => a,
                        Err(e) => {
                            *callback_error.borrow_mut() = Some(PocketTTSError::InferenceFailed(e.to_string()));
                            *should_continue.borrow_mut() = false;
                            return LatentStreamControl::Stop;
                        },
                    };

                    // Extract samples - no crossfade needed, Mimi handles boundaries
                    let audio_vec: Vec<f32> = match audio.squeeze(0).and_then(|a| a.to_vec1()) {
                        Ok(v) => v,
                        Err(e) => {
                            *callback_error.borrow_mut() = Some(PocketTTSError::InferenceFailed(e.to_string()));
                            *should_continue.borrow_mut() = false;
                            return LatentStreamControl::Stop;
                        },
                    };

                    // Callback with audio chunk directly
                    if !audio_vec.is_empty() && !chunk_callback(&audio_vec, is_eos) {
                        *should_continue.borrow_mut() = false;
                        return LatentStreamControl::Stop;
                    }
                }

                // Always continue - let generate_latents_streaming handle EOS termination
                // with proper frames_after_eos logic (generates additional latents after EOS)
                // Returning Stop here would skip those additional latents!
                LatentStreamControl::Continue
            },
        );

        // Check for callback errors
        if let Some(err) = callback_error.into_inner() {
            return Err(err);
        }

        // Handle FlowLM errors
        result.map_err(|e| PocketTTSError::InferenceFailed(format!("FlowLM streaming: {}", e)))?;

        Ok(())
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.mimi.sample_rate() as u32
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> u64 {
        117_856_642 // From model manifest
    }

    /// Get model version
    pub fn version(&self) -> &str {
        "1.0.2"
    }

    /// Decode raw latents to audio (for reference testing)
    ///
    /// Takes raw f32 latent data (already denormalized) and decodes it through Mimi.
    /// The latents should be [num_frames, 32] f32 values.
    ///
    /// Note: The latents are NOT denormalized here - they should already be
    /// in the denormalized form (matching Python's mimi_decoding_input).
    pub fn decode_latents(
        &mut self,
        latents_f32: &[f32],
        num_frames: usize,
    ) -> std::result::Result<Vec<f32>, PocketTTSError> {
        let latent_dim = 32;
        let expected_len = num_frames * latent_dim;

        if latents_f32.len() != expected_len {
            return Err(PocketTTSError::InferenceFailed(format!(
                "Expected {} floats for {} frames, got {}",
                expected_len,
                num_frames,
                latents_f32.len()
            )));
        }

        // Create tensor [1, num_frames, 32]
        let latents = Tensor::from_vec(latents_f32.to_vec(), (1, num_frames, latent_dim), &self.device)
            .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

        eprintln!("[PocketTTS] decode_latents: input shape {:?}", latents.dims());

        // Decode with Mimi (batch transformer, streaming SEANet)
        let audio = self
            .mimi
            .forward_streaming(&latents)
            .map_err(|e| PocketTTSError::InferenceFailed(format!("Mimi decode: {}", e)))?;

        // Convert to Vec<f32>
        let audio = audio.squeeze(0).map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
        let audio_vec: Vec<f32> = audio.to_vec1().map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

        eprintln!("[PocketTTS] decode_latents: output {} samples", audio_vec.len());

        Ok(audio_vec)
    }

    /// Synthesize text to audio, also returning raw latents for debugging
    pub fn synthesize_with_latents(
        &mut self,
        text: &str,
    ) -> std::result::Result<(Vec<f32>, Vec<f32>, [usize; 3]), PocketTTSError> {
        eprintln!("[PocketTTS] synthesize_with_latents called with text len: {}", text.len());

        // Tokenize text
        let token_ids = self.tokenizer.encode(text)?;
        eprintln!("[PocketTTS] tokenized to {} tokens: {:?}", token_ids.len(), token_ids);

        // Create tensor
        let token_tensor = Tensor::from_vec(
            token_ids.iter().map(|&id| id as i64).collect::<Vec<_>>(),
            (1, token_ids.len()),
            &self.device,
        )
        .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

        // Get voice embedding
        let voice = if let Some(ref custom) = self.custom_voice {
            Some(custom)
        } else {
            self.voice_bank.get(self.config.voice_index as usize)
        };

        // Reset caches for new sequence
        self.flowlm.reset_cache();

        // Generate latents with FlowLM + FlowNet
        let num_flow_steps = 1;
        let seed = if self.config.use_fixed_seed { Some(self.config.seed as u64) } else { None };
        let latents = self
            .flowlm
            .generate_latents(&token_tensor, voice, num_flow_steps, self.config.temperature, seed, None)
            .map_err(|e| PocketTTSError::InferenceFailed(format!("FlowLM: {}", e)))?;

        // Get latent shape and data
        let latent_dims = latents.dims();
        let latent_shape = [latent_dims[0], latent_dims[1], latent_dims[2]];
        let latents_flat: Vec<f32> = latents
            .flatten_all()
            .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?
            .to_vec1()
            .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

        eprintln!("[PocketTTS] latents shape: {:?}", latent_shape);

        // Denormalize latents before passing to Mimi
        let latents = self
            .flowlm
            .denormalize_latents(&latents)
            .map_err(|e| PocketTTSError::InferenceFailed(format!("Denormalize: {}", e)))?;

        // Decode to audio using TRUE streaming mode for correct amplitude
        let audio = self
            .mimi
            .forward_true_streaming(&latents)
            .map_err(|e| PocketTTSError::InferenceFailed(format!("Mimi: {}", e)))?;

        // Convert to Vec<f32>
        let audio = audio.squeeze(0).map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
        let audio_vec: Vec<f32> = audio.to_vec1().map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

        Ok((audio_vec, latents_flat, latent_shape))
    }
}

impl std::fmt::Debug for PocketTTSModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PocketTTSModel")
            .field("version", &self.version())
            .field("parameter_count", &self.parameter_count())
            .field("sample_rate", &self.sample_rate())
            .field("voice_count", &self.voice_bank.len())
            .finish()
    }
}
