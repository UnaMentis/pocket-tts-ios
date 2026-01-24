//! Tokenizer for Pocket TTS
//!
//! Uses SentencePiece for proper subword tokenization matching the Python reference.
//!
//! Portions of this file derived from:
//! https://github.com/babybirdprd/pocket-tts
//! Licensed under MIT

use std::path::Path;

use sentencepiece::SentencePieceProcessor;

use crate::error::PocketTTSError;

/// Wrapper for the SentencePiece tokenizer
pub struct PocketTokenizer {
    processor: SentencePieceProcessor,
}

impl PocketTokenizer {
    /// Load tokenizer from SentencePiece .model file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, PocketTTSError> {
        let processor = SentencePieceProcessor::open(path.as_ref())
            .map_err(|e| PocketTTSError::TokenizationFailed(format!("Failed to load SentencePiece model: {}", e)))?;

        Ok(Self { processor })
    }

    /// Load tokenizer from bytes (for bundled models)
    pub fn from_bytes(data: &[u8]) -> Result<Self, PocketTTSError> {
        let processor = SentencePieceProcessor::from_serialized_proto(data)
            .map_err(|e| PocketTTSError::TokenizationFailed(format!("Failed to load SentencePiece from bytes: {}", e)))?;

        Ok(Self { processor })
    }

    /// Encode text to token IDs
    /// Uses SentencePiece BPE/Unigram tokenization for proper subword tokens
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, PocketTTSError> {
        // SentencePiece encode returns PieceWithId, we need to extract the IDs
        let pieces = self.processor.encode(text)
            .map_err(|e| PocketTTSError::TokenizationFailed(format!("Encoding failed: {}", e)))?;

        // Convert to u32 IDs
        let tokens: Vec<u32> = pieces.iter().map(|p| p.id as u32).collect();

        Ok(tokens)
    }

    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[u32]) -> Result<String, PocketTTSError> {
        let text = self.processor.decode_piece_ids(ids)
            .map_err(|e| PocketTTSError::TokenizationFailed(format!("Decoding failed: {}", e)))?;

        Ok(text)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.processor.len()
    }

    /// Get BOS token ID
    pub fn bos_token_id(&self) -> Option<u32> {
        self.processor.bos_id().map(|id| id as u32)
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> Option<u32> {
        self.processor.eos_id().map(|id| id as u32)
    }

    /// Get PAD token ID
    pub fn pad_token_id(&self) -> Option<u32> {
        self.processor.pad_id().map(|id| id as u32)
    }

    /// Get UNK token ID
    pub fn unk_token_id(&self) -> u32 {
        self.processor.unk_id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_creation() {
        // This test would require a .model file
        // Just verify the type compiles
        let _: fn() -> Option<PocketTokenizer> = || None;
    }
}
