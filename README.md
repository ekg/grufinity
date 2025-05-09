# GRUfinity: Efficient Long-Context RNN Language Model

**Development Status**: This project is being improved following best practices from Burn examples. See [TODO.md](TODO.md) for the current improvement plan.

GRUfinity is a Rust implementation of the simplified Gated Recurrent Unit (GRU) architecture using the Burn machine learning framework. This implementation features two key innovations:

1. **Parallel Associative Scan** - Enables efficient parallel computation of recurrent networks
2. **Hidden State Passing** - Provides unlimited context length with fixed memory requirements

This combination creates a highly efficient RNN model capable of processing extremely long sequences without the quadratic complexity of attention-based transformers.

## Installation

### Prerequisites

- Rust (latest stable recommended)
- Cargo
- For CUDA support: CUDA toolkit (11.x or later recommended)

### Building

Clone the repository and build with the appropriate features:

```bash
# Clone the repository
git clone https://github.com/yourusername/grufinity.git
cd grufinity

# Build with WGPU backend (for laptops/desktops without CUDA)
cargo build --release --features wgpu,fusion,autodiff,optimizer-adam

# Or build with CUDA support
cargo build --release --features cuda,fusion,autodiff,optimizer-adam
```

## Feature Flags

GRUfinity supports various backends through feature flags:

### Core Features
- `wgpu` - WebGPU backend (works on most machines)
- `vulkan` - Vulkan backend for better GPU performance on compatible systems
- `fusion` - Enable fusion optimization for better performance
- `autodiff` - Enable automatic differentiation (required for training)
- `optimizer-adam` - Use Adam optimizer
- `optimizer-sgd` - Use SGD optimizer (alternative to Adam)
- `g-func` - Use original activation function (g(x) = x + 0.5 for x >= 0, sigmoid(x) for x < 0)
- `swish` - Use Swish/SiLU activation function (f(x) = x * sigmoid(x)) - default
- `gelu` - Use GELU activation function (x * Φ(x) where Φ is the CDF of the standard normal)
- `swiglu` - Use SwiGLU activation (instead of SiLU)
- `tanh` - Apply tanh nonlinearity between chunks (default: disabled)

### Accelerated Backends
- `cuda` - CUDA JIT backend for NVIDIA GPUs
- `candle-cuda` - Candle CUDA backend (alternative CUDA implementation)
- `candle-metal` - Metal backend for Apple Silicon

### Combinations
- `wgpu-fusion` = `wgpu` + `fusion` + `autodiff` + `autotune`
- `vulkan-fusion` = `vulkan` + `fusion` + `autodiff` + `autotune`
- `wgpu-spirv-fusion` = Alias for `vulkan-fusion` (backward compatibility)
- `cuda-full` = `cuda` + `fusion` + `autodiff`
- `candle-cuda-full` = `candle` + `candle-cuda` + `fusion` + `autodiff`

### Feature Selection Guidelines

1. **For NVIDIA GPUs**: Use `cuda-full` feature for best performance
   ```
   cargo run --release --features cuda-full,optimizer-adam
   ```

2. **For Apple Silicon**: Use `candle-metal-full`  
   ```
   cargo run --release --features candle-metal-full,optimizer-adam
   ```

3. **For AMD/Intel GPUs**: Use `vulkan-fusion`
   ```
   cargo run --release --features vulkan-fusion,optimizer-adam
   ```

4. **For compatibility with any GPU**: Use `wgpu-fusion`
   ```
   cargo run --release --features wgpu-fusion,optimizer-adam
   ```

5. **For CPU-only usage**: Use `candle` (for newer machines) or `ndarray` (for maximum compatibility)
   ```
   cargo run --release --features candle,fusion,autodiff,optimizer-adam
   ```

## Backend Selection

GRUfinity supports multiple compute backends with different performance characteristics:

### Backend Priority Order

When multiple backends are enabled, they are prioritized in this order:
1. CUDA (NVIDIA GPUs) - Fastest for CUDA-capable hardware
2. Candle CUDA - Alternative CUDA implementation
3. Vulkan - High-performance cross-platform GPU acceleration
4. WebGPU - Portable GPU acceleration (fallback)
5. Metal (Apple Silicon) - Optimized for Apple hardware
6. Candle CPU - CPU-only fallback
7. Others (NdArray, LibTorch) - Further fallbacks

### Vulkan Backend

The Vulkan backend provides efficient GPU acceleration on compatible hardware without requiring CUDA. It's particularly useful for:
- Linux systems with AMD or Intel GPUs
- Windows machines with non-NVIDIA GPUs
- Systems where CUDA is unavailable

To use the Vulkan backend:
```bash
cargo run --release --bin train --features vulkan,fusion,autodiff,optimizer-adam -- [OPTIONS]
```

For backward compatibility, the legacy `wgpu-spirv-fusion` feature name is maintained as an alias for `vulkan-fusion`.

## Getting Training Data

Download some sample data to get started:

```bash
# Download tiny Shakespeare dataset
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/tinyshakespeare.txt

# Ensure data directory exists
mkdir -p data
```

## Training

### Basic Training (WGPU - for laptops/desktops)

```bash
cargo run --release --bin train --features wgpu,fusion,autodiff,optimizer-adam -- \
  --data data/tinyshakespeare.txt \
  --output out \
  --update-tokens 128 \
  --backprop-tokens 1024 \
  --chunk-size 128 \
  --context-length 16384 \
  --learning-rate 0.001 \
  --num-epochs 100 \
  --grad-clip 0.5 \
  --batch-size 32 \
  --lr-scheduler cosine \
  --model-dim 128
```

### Training with Vulkan (for systems with Vulkan support)

```bash
cargo run --release --bin train --features vulkan,fusion,autodiff,optimizer-adam -- \
  --data data/tinyshakespeare.txt \
  --output out \
  --update-tokens 192 \
  --backprop-tokens 1536 \
  --chunk-size 192 \
  --context-length 16384 \
  --learning-rate 0.001 \
  --num-epochs 100 \
  --grad-clip 0.5 \
  --batch-size 32 \
  --lr-scheduler cosine \
  --model-dim 128
```

### Training with CUDA (for CUDA-capable systems)

```bash
cargo run --release --bin train --features cuda,fusion,autodiff,optimizer-adam -- \
  --data data/tinyshakespeare.txt \
  --output out \
  --update-tokens 256 \
  --backprop-tokens 2048 \
  --chunk-size 256 \
  --context-length 16384 \
  --learning-rate 0.01 \
  --min-lr-factor 0.001 \
  --num-epochs 20 \
  --grad-clip 0.5 \
  --batch-size 32 \
  --lr-scheduler cosine \
  --device-id 0 \
  --model-dim 128
```

### Training Parameters

- `--data` - Path to training text file
- `--output` - Directory for output artifacts
- `--update-tokens` - Update parameters every ~N tokens
- `--backprop-tokens` - Backpropagate through ~N tokens
- `--chunk-size` - Characters per chunk for processing
- `--context-length` - Total context length in characters
- `--learning-rate` - Learning rate for optimization
- `--num-epochs` - Number of training epochs
- `--grad-clip` - Gradient clipping value
- `--batch-size` - Number of random start positions
- `--lr-scheduler` - Learning rate scheduler (constant, cosine, linear)
- `--model-dim` - Model hidden dimension size
- `--model-depth` - Number of MinGRU layers (default: 3)
- `--device-id` - CUDA device ID (if using CUDA)

## Text Generation

Generate text with a trained model:

```bash
cargo run --release --bin generate --features wgpu,fusion,autodiff,optimizer-adam -- \
  --model out/model_best.bin \
  --vocab out/vocab.txt \
  --prompt "Once upon a time" \
  --length 500 \
  --chunk-size 64 \
  --temperature 0.8 \
  --top-k 40
```

### Text Generation Parameters

- `--model` - Path to trained model file
- `--vocab` - Path to vocabulary file
- `--prompt` - Initial text to seed generation
- `--length` - Number of characters to generate
- `--temperature` - Sampling temperature (higher = more random)
- `--top-k` - Top-k sampling value (0 = disabled)
- `--chunk-size` - Characters per chunk for processing

## Model Architecture

GRUfinity uses a MinGRU (Minimal Gated Recurrent Unit) architecture with several optimizations:

1. **GPU-Optimized Dimensions**: All model dimensions are multiples of 32 (preferably powers of 2 like 128, 256, 512, 1024) for optimal GPU memory alignment and computation
2. **MinGRU Layers**: Simplified GRU cells with a single gate and efficient activation function:
   ```
   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ g(h̃_t)
   
   where g(x) = x + 0.5 for x ≥ 0, sigmoid(x) for x < 0
   ```

2. **Feed-Forward Layers**: Between GRU layers for additional capacity
3. **Residual Connections**: For better gradient flow and training stability
4. **Layer Normalization**: RMSNorm is used for stable training

The model processes text in chunks using Truncated Backpropagation Through Time (TBPTT) with hidden state passing between chunks for unlimited context length.

### Key Innovations

- **Parallel Associative Scan**: Allows computing recurrent neural networks in parallel using log(n) steps
- **Log-Space Computation**: Improves numerical stability for processing long sequences
- **Hidden State Passing**: Enables processing arbitrarily long texts with fixed memory usage

## Performance Considerations

- Memory usage is constant regardless of context length
- The hidden state size doesn't grow with sequence length (unlike KV caches in Transformers)
- Computation scales linearly with sequence length
- Efficient on both CPUs and GPUs with appropriate backend selection

## References

- Burn ML Framework: [https://github.com/tracel-ai/burn](https://github.com/tracel-ai/burn)
