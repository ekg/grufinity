# GRUfinity: Efficient Long-Context RNN Language Model

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
cargo build --release --features cuda-jit,fusion,autodiff,optimizer-adam
```

## Feature Flags

GRUfinity supports various backends through feature flags:

### Core Features
- `wgpu` - WebGPU backend (works on most machines)
- `fusion` - Enable fusion optimization for better performance
- `autodiff` - Enable automatic differentiation (required for training)
- `optimizer-adam` - Use Adam optimizer
- `optimizer-sgd` - Use SGD optimizer (alternative to Adam)

### Accelerated Backends
- `cuda-jit` - CUDA JIT backend for NVIDIA GPUs
- `candle-cuda` - Candle CUDA backend (alternative CUDA implementation)
- `candle-metal` - Metal backend for Apple Silicon

### Combinations
- `wgpu-fusion` = `wgpu` + `fusion` + `autodiff` + `autotune`
- `cuda-full` = `cuda-jit` + `fusion` + `autodiff`
- `candle-cuda-full` = `candle` + `candle-cuda` + `fusion` + `autodiff`

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

### Training with CUDA (for CUDA-capable systems)

```bash
cargo run --release --bin train --features cuda-jit,fusion,autodiff,optimizer-adam -- \
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

1. **MinGRU Layers**: Simplified GRU cells with a single gate and efficient activation function:
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
