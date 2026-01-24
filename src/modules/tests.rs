//! Tests for neural network modules

#[cfg(test)]
mod layer_norm_tests {
    use candle_core::{Device, Tensor};
    use crate::modules::layer_norm::RMSNorm;

    #[test]
    fn test_rms_norm_output_shape() {
        let device = Device::Cpu;
        let hidden_size = 64;
        let weight = Tensor::ones((hidden_size,), candle_core::DType::F32, &device).unwrap();
        let norm = RMSNorm::load(hidden_size, 1e-6, weight);

        let input = Tensor::randn(0.0f32, 1.0, (2, 10, hidden_size), &device).unwrap();
        let output = candle_nn::Module::forward(&norm, &input).unwrap();

        assert_eq!(output.dims(), input.dims());
    }

    #[test]
    fn test_rms_norm_preserves_dtype() {
        let device = Device::Cpu;
        let hidden_size = 32;
        let weight = Tensor::ones((hidden_size,), candle_core::DType::F32, &device).unwrap();
        let norm = RMSNorm::load(hidden_size, 1e-6, weight);

        let input = Tensor::randn(0.0f32, 1.0, (1, 5, hidden_size), &device).unwrap();
        let output = candle_nn::Module::forward(&norm, &input).unwrap();

        assert_eq!(output.dtype(), input.dtype());
    }
}

#[cfg(test)]
mod rotary_tests {
    use candle_core::{Device, Tensor};
    use crate::modules::rotary::RotaryEmbedding;

    #[test]
    fn test_rotary_embedding_creation() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 2048, 10000.0, &device);
        assert!(rope.is_ok());
    }

    #[test]
    fn test_rotary_embedding_forward() {
        let device = Device::Cpu;
        let dim = 64;
        let rope = RotaryEmbedding::new(dim, 2048, 10000.0, &device).unwrap();

        // Create Q and K tensors: [batch, seq, num_heads, head_dim]
        // The implementation expects seq at dim 1 (q.dim(1)? in forward)
        let q = Tensor::randn(0.0f32, 1.0, (1, 10, 4, dim), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 10, 4, dim), &device).unwrap();

        let (q_rot, k_rot) = rope.forward(&q, &k, 0).unwrap();

        assert_eq!(q_rot.dims(), q.dims());
        assert_eq!(k_rot.dims(), k.dims());
    }

    #[test]
    fn test_rotary_embedding_with_offset() {
        let device = Device::Cpu;
        let dim = 64;
        let rope = RotaryEmbedding::new(dim, 2048, 10000.0, &device).unwrap();

        // Shape: [batch, seq, num_heads, head_dim]
        let q = Tensor::randn(0.0f32, 1.0, (1, 5, 4, dim), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 5, 4, dim), &device).unwrap();

        // Apply with offset
        let result = rope.forward(&q, &k, 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rotary_embedding_exceeds_max_len() {
        let device = Device::Cpu;
        let dim = 64;
        let max_len = 100;
        let rope = RotaryEmbedding::new(dim, max_len, 10000.0, &device).unwrap();

        // Shape: [batch, seq, num_heads, head_dim]
        let q = Tensor::randn(0.0f32, 1.0, (1, 10, 4, dim), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 10, 4, dim), &device).unwrap();

        // Offset + seq_len > max_len should fail
        let result = rope.forward(&q, &k, 95);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod attention_tests {
    use crate::modules::attention::KVCache;

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::new();
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_kv_cache_default() {
        let cache = KVCache::default();
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = KVCache::new();
        cache.clear();
        assert_eq!(cache.seq_len(), 0);
    }
}

#[cfg(test)]
mod mlp_tests {
    #[test]
    fn test_mlp_sampler_step_clamping() {
        // Test that consistency steps are clamped to 1-4
        use crate::modules::mlp::MLPSampler;

        // We can't easily test the full sampler without weights,
        // but we can verify the struct exists and compiles
        let _sampler_type: fn() -> Option<MLPSampler> = || None;
    }
}

#[cfg(test)]
mod embedding_tests {
    use candle_core::{Device, Tensor};
    use crate::modules::embeddings::{VoiceEmbedding, VoiceBank};

    #[test]
    fn test_voice_embedding_from_tensor() {
        let device = Device::Cpu;
        let embedding = Tensor::randn(0.0f32, 1.0, (512,), &device).unwrap();
        let voice = VoiceEmbedding::from_tensor(embedding);
        assert!(voice.is_ok());
        assert_eq!(voice.unwrap().voice_dim(), 512);
    }

    #[test]
    fn test_voice_embedding_expand() {
        let device = Device::Cpu;
        // Voice embeddings are [seq_len, dim] - e.g., [125, 1024] for Kyutai
        let embedding = Tensor::randn(0.0f32, 1.0, (10, 256), &device).unwrap();
        let voice = VoiceEmbedding::from_tensor(embedding).unwrap();

        let expanded = voice.expand_to_seq(2, 10);
        assert!(expanded.is_ok());

        let expanded = expanded.unwrap();
        // Output should be [batch, prompt_seq, dim]
        assert_eq!(expanded.dims(), &[2, 10, 256]);
    }

    #[test]
    fn test_voice_bank_new() {
        let bank = VoiceBank::new(512);
        assert_eq!(bank.len(), 0);
        assert!(bank.is_empty());
        assert_eq!(bank.voice_dim(), 512);
    }

    #[test]
    fn test_voice_bank_get_empty() {
        let bank = VoiceBank::new(512);
        assert!(bank.get(0).is_none());
        assert!(bank.get(7).is_none());
    }
}

#[cfg(test)]
mod conv_tests {
    // Conv tests would require VarBuilder with actual weights
    // For now, just verify the types compile

    #[test]
    fn test_conv_types_exist() {
        use crate::modules::conv::{Conv1d, CausalConv1d, ConvTranspose1d};
        use crate::modules::conv::{SEANetEncoderBlock, SEANetDecoderBlock};

        // Type checking - these should compile
        let _: fn() -> Option<Conv1d> = || None;
        let _: fn() -> Option<CausalConv1d> = || None;
        let _: fn() -> Option<ConvTranspose1d> = || None;
        let _: fn() -> Option<SEANetEncoderBlock> = || None;
        let _: fn() -> Option<SEANetDecoderBlock> = || None;
    }
}
