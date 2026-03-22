#!/usr/bin/env python3
"""
Dump per-layer intermediate tensors from Python FlowLM at autoregressive step 0.

Saves tensors to /tmp/python_step0/ for comparison with Rust equivalents.
Each file is a .npy with shape (1024,) — the hidden dimension at the last token position.

Captured points per layer (6 layers, i=0..5):
  layer{i}_input.npy       — input to the layer (before norm1)
  layer{i}_norm1.npy       — after LayerNorm1
  layer{i}_attn.npy        — raw attention output (before residual add)
  layer{i}_post_attn.npy   — after attention residual add
  layer{i}_norm2.npy       — after LayerNorm2
  layer{i}_mlp.npy         — raw MLP output (before residual add)
  layer{i}_output.npy      — final layer output (after MLP residual add)

Top-level:
  input_linear.npy         — output of input_linear (input to layer 0)
  out_norm.npy             — output of out_norm (final output)

Usage:
    cd /tmp && /path/to/validation/.venv/bin/python3 /path/to/validation/dump_intermediates.py
"""

import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DUMP_DIR = Path("/tmp/python_step0")
PHRASE = "Hello, this is a test of the Pocket TTS system."
VOICE = "alba"
SEED = 42
CONSISTENCY_STEPS = 1


def save(name, tensor):
    """Save tensor as .npy — extract last position, squeeze to 1D."""
    arr = tensor.detach().cpu().float().numpy()
    # Squeeze batch dims
    while arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    # Take last position if 2D (seq_len, dim)
    if arr.ndim == 2:
        arr = arr[-1]
    np.save(str(DUMP_DIR / f"{name}.npy"), arr)
    print(f"  {name}: shape={arr.shape} mean={arr.mean():.6f} std={arr.std():.6f} "
          f"first4=[{arr[0]:.6f},{arr[1]:.6f},{arr[2]:.6f},{arr[3]:.6f}]")


def patch_transformer_layers(transformer):
    """Monkey-patch each layer's forward to dump intermediates at step 0."""
    # Track whether we've seen step 0 (first autoregressive call with seq_len=1)
    state = {"dumped": False}

    for layer_idx, layer in enumerate(transformer.layers):
        def make_patched(li, lyr):
            def patched_forward(x, model_state=None):
                # Detect step 0: seq_len=1 and haven't dumped yet
                is_step0 = (x.shape[1] == 1 and not state["dumped"])

                if is_step0:
                    save(f"layer{li}_input", x[:, -1:, :])

                # -- Self-attention block (inlined from _sa_block) --
                x_orig = x
                normed = lyr.norm1(x)
                if is_step0:
                    save(f"layer{li}_norm1", normed[:, -1:, :])

                attn_out = lyr.self_attn(normed, model_state)
                if is_step0:
                    save(f"layer{li}_attn", attn_out[:, -1:, :])

                x = x_orig.to(attn_out) + lyr.layer_scale_1(attn_out)
                if is_step0:
                    save(f"layer{li}_post_attn", x[:, -1:, :])

                # -- Feed-forward block (inlined from _ff_block) --
                x_orig2 = x
                normed2 = lyr.norm2(x)
                if is_step0:
                    save(f"layer{li}_norm2", normed2[:, -1:, :])

                mlp_out = lyr.linear2(F.gelu(lyr.linear1(normed2)))
                if is_step0:
                    save(f"layer{li}_mlp", mlp_out[:, -1:, :])

                x = x_orig2.to(mlp_out) + lyr.layer_scale_2(mlp_out)
                if is_step0:
                    save(f"layer{li}_output", x[:, -1:, :])
                    # Mark step 0 done after last layer
                    if li == len(transformer.layers) - 1:
                        state["dumped"] = True

                return x
            return patched_forward

        layer.forward = make_patched(layer_idx, layer)


def patch_backbone(flow_lm):
    """Patch FlowLM.backbone to capture input_linear and out_norm at steps 0-2."""
    state = {"step": 0}

    def patched_backbone(input_, text_embeddings, sequence, model_state):
        step = state["step"]
        is_ar = (sequence.shape[1] == 1)

        input_cat = torch.cat([text_embeddings, input_], dim=1)

        if is_ar and step <= 2:
            if step > 0:
                print(f"\n--- Step {step} intermediates ---")
            save(f"step{step}_input_linear", input_cat[:, -1:, :])

        transformer_out = flow_lm.transformer(input_cat, model_state)

        if flow_lm.out_norm:
            transformer_out = flow_lm.out_norm(transformer_out)

        transformer_out = transformer_out[:, -sequence.shape[1]:]

        if is_ar and step <= 2:
            save(f"step{step}_out_norm", transformer_out[:, -1:, :])

        if is_ar:
            state["step"] += 1

        return transformer_out

    flow_lm.backbone = patched_backbone


def patch_forward_for_latent(flow_lm):
    """Patch FlowLM.forward to capture FlowNet output (latent) at steps 0-2."""
    orig_forward = flow_lm.forward
    state = {"step": 0}

    def patched_forward(**kwargs):
        result = orig_forward(**kwargs)
        step = state["step"]
        sequence = kwargs.get("sequence")
        is_ar = (sequence is not None and sequence.shape[1] == 1)

        if is_ar and step <= 2:
            latent, eos = result
            save(f"step{step}_latent", latent)

        if is_ar:
            state["step"] += 1

        return result

    flow_lm.forward = patched_forward


def main():
    if DUMP_DIR.exists():
        shutil.rmtree(DUMP_DIR)
    DUMP_DIR.mkdir(parents=True)

    print("Loading model...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model()

    flow_lm = model.flow_lm
    n_layers = len(flow_lm.transformer.layers)
    print(f"Patching {n_layers} transformer layers...")

    patch_transformer_layers(flow_lm.transformer)
    patch_backbone(flow_lm)
    patch_forward_for_latent(flow_lm)

    print(f"Getting voice state for '{VOICE}'...")
    voice_state = model.get_state_for_audio_prompt(VOICE)

    torch.manual_seed(SEED)
    print(f"Generating: \"{PHRASE}\" (seed={SEED}, steps={CONSISTENCY_STEPS})")
    print()
    print("--- Step 0 intermediate tensors ---")

    # lsd_decode_steps defaults to 1 (matches Rust consistency_steps=1)
    print(f"lsd_decode_steps={model.lsd_decode_steps}")
    audio = model.generate_audio(voice_state, PHRASE)

    print()
    files = sorted(os.listdir(DUMP_DIR))
    print(f"Done. {len(audio.numpy())} samples generated.")
    print(f"Saved {len(files)} tensor files to {DUMP_DIR}/")


if __name__ == "__main__":
    main()
