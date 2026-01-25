# StatefulModule: Streaming State Management Pattern

## Source File
`validation/.venv/lib/python3.11/site-packages/pocket_tts/modules/stateful_module.py`

---

## Overview

`StatefulModule` is an abstract base class that provides a pattern for managing streaming state across neural network modules. Each stateful module has its own state dictionary, and all states are collected into a single model-level state dictionary keyed by module name.

---

## Complete Python Implementation

```python
from abc import ABC, abstractmethod
import torch
from torch import nn


def init_states(
    model: nn.Module, batch_size: int, sequence_length: int
) -> dict[str, dict[str, torch.Tensor]]:
    """Initialize states for all StatefulModules in a model.

    Walks the module tree and calls init_state() on each StatefulModule.
    The result is a flat dictionary mapping module names to their states.

    Args:
        model: The root module (e.g., MimiModel)
        batch_size: Batch size for state tensors
        sequence_length: Sequence length hint (used by some modules)

    Returns:
        Dictionary: {module_name: {state_key: tensor, ...}, ...}
    """
    result = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, StatefulModule):
            continue
        # Store the module's absolute name for later lookup
        module._module_absolute_name = module_name
        # Initialize this module's state
        module_state = module.init_state(batch_size, sequence_length=sequence_length)
        result[module_name] = module_state
    return result


def increment_steps(
    module: nn.Module, model_state: dict[str, dict[str, torch.Tensor]], increment: int = 1
):
    """Increment step counters for all stateful modules.

    Some modules (like attention with KV cache) track position.
    """
    for module_name, module in module.named_modules():
        if not isinstance(module, StatefulModule):
            continue
        module.increment_step(model_state[module_name], increment)


class StatefulModule(ABC, nn.Module):
    """Base class for modules that maintain streaming state.

    Subclasses must implement:
        init_state(batch_size, sequence_length) -> dict[str, Tensor]

    The state dict typically contains tensors for:
        - Previous samples for causal convolution
        - Partial outputs for overlap-add
        - KV cache for attention
        - Position counters
    """

    def __init__(self, *args, **kwds):
        # Will be set by init_states() when walking the module tree
        self._module_absolute_name = None
        return super().__init__(*args, **kwds)

    @abstractmethod
    def init_state(self, batch_size: int, sequence_length: int):
        """Initialize the state for this module.

        Args:
            batch_size: Batch dimension for state tensors
            sequence_length: Total sequence length (hint for pre-allocation)

        Returns:
            dict[str, Tensor]: State dictionary for this module
        """
        raise NotImplementedError

    def increment_step(self, state: dict, increment: int = 1):
        """Increment position/step counter in state (optional).

        Default implementation does nothing. Override for modules
        that track position (e.g., attention with KV cache).
        """
        pass

    def get_state(self, model_state: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Get this module's state from the model state dictionary.

        Args:
            model_state: The full model state from init_states()

        Returns:
            This module's state dictionary
        """
        return model_state[self._module_absolute_name]
```

---

## State Dictionary Structure

The model state is a nested dictionary:

```python
model_state = {
    # StreamingConv1d layers
    "decoder.model.0": {
        "previous": Tensor([B, C, context_len]),
        "first": Tensor([B], dtype=bool)
    },

    # StreamingConvTranspose1d layers
    "decoder.model.2.convtr": {
        "partial": Tensor([B, C, overlap_len])
    },

    # Attention layers (if streaming)
    "transformer.layers.0.self_attn": {
        "cache": Tensor([2, B, max_len, heads, dim_per_head]),
        "current_end": Tensor([positions_so_far])
    },

    # ... one entry per StatefulModule
}
```

---

## Module Naming Convention

Module names come from `model.named_modules()`, which uses dot-separated paths:

```
decoder                          # SEANetDecoder
decoder.model                    # nn.ModuleList
decoder.model.0                  # First StreamingConv1d
decoder.model.1                  # ELU (not stateful)
decoder.model.2                  # SEANetResnetBlock
decoder.model.2.block            # nn.ModuleList
decoder.model.2.block.0          # ELU
decoder.model.2.block.1          # StreamingConv1d
decoder.model.2.block.1.conv     # nn.Conv1d (not stateful)
...
```

**Key:** The `_module_absolute_name` is set during `init_states()` and used by `get_state()` to look up the correct entry.

---

## Usage Pattern

### Initialization (Before Generation Loop)

```python
from pocket_tts.modules.stateful_module import init_states

# Create model
mimi = MimiModel.from_config(config)

# Initialize all streaming states
batch_size = 1
max_sequence_length = 1000  # hint for pre-allocation
mimi_state = init_states(mimi, batch_size, max_sequence_length)
```

### During Generation (Each Frame)

```python
# Generate one latent frame
latent = flow_lm.generate_next(...)

# Decode with streaming - state is updated in-place
audio_chunk = mimi.decode_from_latent(latent, mimi_state)

# For modules that track position:
# increment_steps(mimi, mimi_state, increment=1)
```

### State Persistence

The state dictionary contains regular tensors. It can be:
- Saved/loaded with `torch.save()`/`torch.load()`
- Cloned with `{k: {k2: v2.clone() for k2, v2 in v.items()} for k, v in state.items()}`
- Reset by calling `init_states()` again

---

## SEANet Decoder State Layout

For the complete SEANet decoder, here are all stateful modules:

```python
# Example state keys for SEANet decoder with 4 upsample stages
# Ratios: [8, 5, 4, 2], 3 residual blocks per stage

state_keys = [
    # Initial conv
    "decoder.model.0",  # StreamingConv1d, previous: [B, 512, 6]

    # Stage 0: upsample 8x
    "decoder.model.2",  # StreamingConvTranspose1d, partial: [B, 256, 8]
    "decoder.model.3.block.1",  # ResBlock conv1, previous: [B, 128, 2]
    "decoder.model.3.block.3",  # ResBlock conv2, previous: [B, 256, 0]
    "decoder.model.4.block.1",  # ResBlock conv1 (dilation=2)
    "decoder.model.4.block.3",
    "decoder.model.5.block.1",  # ResBlock conv1 (dilation=4)
    "decoder.model.5.block.3",

    # Stage 1: upsample 5x
    "decoder.model.7",  # StreamingConvTranspose1d, partial: [B, 128, 5]
    # ... residual blocks ...

    # Stage 2: upsample 4x
    # Stage 3: upsample 2x

    # Final conv
    "decoder.model.N",  # StreamingConv1d, previous: [B, 32, 6]
]
```

---

## Rust Implementation

### State Structure

```rust
/// State for all streaming modules in the model
pub struct MimiState {
    /// Map from module name to module state
    states: HashMap<String, ModuleState>,
}

pub enum ModuleState {
    Conv1d {
        previous: Tensor,
        first: Tensor,  // bool tensor
    },
    ConvTranspose1d {
        partial: Tensor,
    },
    Attention {
        cache: Tensor,
        current_end: usize,
    },
}
```

### Initialization

```rust
impl MimiState {
    pub fn new(model: &MimiModel, batch_size: usize, max_seq_len: usize) -> Self {
        let mut states = HashMap::new();

        // Walk model and initialize each stateful layer
        for (name, layer) in model.named_layers() {
            match layer {
                Layer::StreamingConv1d(conv) => {
                    let context = conv.effective_kernel_size() - conv.stride();
                    states.insert(name, ModuleState::Conv1d {
                        previous: Tensor::zeros(&[batch_size, conv.in_channels(), context]),
                        first: Tensor::ones(&[batch_size]).to_dtype(DType::Bool),
                    });
                }
                Layer::StreamingConvTranspose1d(conv) => {
                    let overlap = conv.kernel_size() - conv.stride();
                    states.insert(name, ModuleState::ConvTranspose1d {
                        partial: Tensor::zeros(&[batch_size, conv.out_channels(), overlap]),
                    });
                }
                // ... other layer types
            }
        }

        MimiState { states }
    }

    pub fn get(&self, name: &str) -> Option<&ModuleState> {
        self.states.get(name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut ModuleState> {
        self.states.get_mut(name)
    }
}
```

---

## Key Implementation Notes

### 1. In-Place State Updates

Python updates state tensors in-place:
```python
state["previous"][:] = x[..., -TP:]  # In-place update
layer_state[:] = for_partial         # In-place update
```

In Rust, you'll need mutable references or interior mutability.

### 2. State Lifetime

State must persist across generation calls. Don't reinitialize per-call:
```python
# WRONG: Reinitializing each call
def generate_frame():
    state = init_states(model, batch_size, seq_len)  # Bug!
    return model(x, state)

# RIGHT: Persistent state
state = init_states(model, batch_size, seq_len)
for _ in range(num_frames):
    output = model(x, state)  # State updated in-place
```

### 3. Batch Independence

States are independent per batch item. The `first` tensor tracks per-batch whether it's the first frame:
```python
first = torch.ones(batch_size, dtype=torch.bool)  # [True, True, ...]
```

---

## Related Files

- `stateful_module.py` - Base class and init_states function
- `conv.py` - StreamingConv1d and StreamingConvTranspose1d
- `transformer.py` - StreamingMultiheadAttention
- `mimi.py:81-86` - Uses state in decode_from_latent
