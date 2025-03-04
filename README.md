# MinGRU: Efficient Long-Context RNN

## Overview

MinGRU is a Rust implementation of the simplified Gated Recurrent Unit (GRU) architecture using the Burn machine learning framework. This implementation features two key innovations:

1. **Parallel Associative Scan** - Enables efficient parallel computation of recurrent networks
2. **Hidden State Passing** - Provides unlimited context length with fixed memory requirements

This combination creates a highly efficient RNN model capable of processing extremely long sequences without the quadratic complexity of attention-based transformers.

## Table of Contents

- [Theoretical Background](#theoretical-background)
  - [MinGRU Architecture](#mingru-architecture)
  - [Parallel Associative Scan](#parallel-associative-scan)
  - [Hidden State Passing](#hidden-state-passing)
- [Code Architecture](#code-architecture)
  - [Project Structure](#project-structure)
  - [Key Components](#key-components)
- [Implementation Details](#implementation-details)
  - [Parallel Scan Algorithm](#parallel-scan-algorithm)
  - [Log-Space Computation](#log-space-computation)
  - [Sequential vs. Parallel Mode](#sequential-vs-parallel-mode)
  - [Chunked Processing](#chunked-processing)
- [Usage Guide](#usage-guide)
  - [Installation](#installation)
  - [Training](#training)
  - [Text Generation](#text-generation)
  - [Model Configuration](#model-configuration)
- [Performance Considerations](#performance-considerations)
- [Development and Testing](#development-and-testing)
- [Future Improvements](#future-improvements)

## Theoretical Background

### MinGRU Architecture

MinGRU is a simplified variant of the GRU (Gated Recurrent Unit) architecture. The standard GRU equation is:

```
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

where:
- `h_t` is the hidden state at time step t
- `z_t` is the update gate
- `h̃_t` is the candidate activation

MinGRU simplifies this by:
1. Using a single gate (`z_t`) instead of two (update and reset)
2. Applying a special `g` function to the candidate activation:
   ```
   g(x) = x + 0.5 for x ≥ 0
   g(x) = sigmoid(x) for x < 0
   ```

This simplification reduces computation while maintaining most of the modeling capacity.

### Parallel Associative Scan

Traditional RNN processing is sequential by nature, making it difficult to parallelize. The associative scan algorithm transforms the recursive dependency into a parallelizable operation by exploiting the associative property.

For the recurrence relation `h_t = a_t ⊙ h_{t-1} + b_t`, we can compute all hidden states in parallel by:

1. Computing prefix products of coefficients: `a_star[t] = a_1 * a_2 * ... * a_t`
2. Computing weighted sums of values: `b_star[t] = b_t + a_t * b_{t-1} + a_t * a_{t-1} * b_{t-2} + ...`
3. Combining: `h_t = a_star[t] * h_0 + b_star[t]`

This allows us to compute RNN outputs in `O(log n)` parallel steps instead of `O(n)` sequential steps.

### Hidden State Passing

Inspired by Transformer-XL and insights from LongMamba, our implementation can process arbitrarily long sequences by:

1. Processing fixed-size chunks with parallel scan (for efficiency)
2. Passing the final hidden state of each chunk to the next chunk
3. Maintaining fixed memory usage regardless of sequence length

This approach offers unique advantages compared to attention-based models:
- Memory usage is constant regardless of context length
- The hidden state size doesn't grow with sequence length (unlike KV caches)
- Computation scales linearly with sequence length

## Code Architecture

### Project Structure

```
mingru/
├── Cargo.toml                 # Project configuration
├── src/
│   ├── lib.rs                 # Library exports
│   ├── parallel_scan.rs       # Parallel associative scan implementation
│   ├── mingru.rs              # MinGRU module implementation
│   ├── model.rs               # Language model implementation 
│   ├── dataset.rs             # Dataset and vocabulary handling
│   └── bin/
│       ├── train.rs           # Training executable
│       └── generate.rs        # Text generation executable
```

### Key Components

1. **ParallelScan** - Implementation of the parallel associative scan algorithm:
   - Regular and log-space versions
   - Supports initial hidden state (h0)

2. **MinGRU** - Efficient RNN module:
   - Integrates with Burn module system
   - Supports both sequential and parallel computation
   - Log-space computation for numerical stability

3. **MinGRULM** - Language model using MinGRU layers:
   - Multiple MinGRU layers with residual connections
   - Feed-forward layers between MinGRU layers
   - Automatic chunking of long sequences
   - Hidden state management between chunks

4. **Dataset Handling** - Efficient data management:
   - Character-level vocabulary
   - Text dataset with chunk support
   - Document-aware chunked dataset
   - Batching with device management

## Implementation Details

### Parallel Scan Algorithm

The core of our implementation is the parallel associative scan algorithm, which enables efficient computation of recurrences. The implementation supports both regular computation and a more numerically stable log-space version.

```rust
pub fn parallel_scan<B: Backend>(
    coeffs: Tensor<B, 3>,  // a_t coefficients (batch_size, seq_len, hidden_dim)
    values: Tensor<B, 3>,  // b_t values (batch_size, seq_len, hidden_dim)
    h0: Option<Tensor<B, 2>>, // Initial hidden state (batch_size, hidden_dim)
) -> Tensor<B, 3> { ... }
```

The algorithm works in three main steps:
1. Compute cumulative products of coefficients (a_star)
2. Compute prefix sums of scaled values (b_star_prefix_sum)
3. Combine them to produce the final hidden states

### Log-Space Computation

For numerical stability, we implement a log-space version of the parallel scan:

```rust
pub fn parallel_scan_log<B: Backend>(
    log_coeffs: Tensor<B, 3>,  // log(a_t) coefficients
    log_values: Tensor<B, 3>,  // log(b_t) values
    h0: Option<Tensor<B, 2>>,  // Initial hidden state
) -> Tensor<B, 3> { ... }
```

This uses the log-sum-exp trick for stable computation of sums in log-space:

```rust
fn log_cumsum_exp<B: Backend>(log_x: Tensor<B, 3>) -> Tensor<B, 3> { ... }
```

This is particularly important for long sequences where numerical underflow/overflow can occur.

### Sequential vs. Parallel Mode

The MinGRU module operates in two modes:

1. **Sequential Mode** - Used when processing one token at a time:
   ```rust
   fn forward_sequential(&self, hidden: Tensor<B, 3>, gate: Tensor<B, 3>, prev_hidden: Tensor<B, 2>) -> Tensor<B, 3> { ... }
   ```
   This is efficient for inference and when hidden states need to be explicitly passed.

2. **Parallel Mode** - Used when processing multiple tokens in parallel:
   ```rust
   fn forward_parallel(&self, hidden: Tensor<B, 3>, gate: Tensor<B, 3>, prev_hidden: Option<Tensor<B, 2>>) -> Tensor<B, 3> { ... }
   ```
   This leverages the parallel scan algorithm for efficient training.

The mode is automatically selected based on the input sequence length and whether a previous hidden state is provided.

### Chunked Processing

For handling extremely long sequences, we implement chunked processing:

```rust
fn forward_chunked(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Vec<Tensor<B, 2>>) { ... }
```

This automatically breaks a long sequence into chunks, processes each chunk with the appropriate hidden state, and concatenates the results. This allows processing sequences of any length with fixed memory usage.

The hidden state passing happens in the `forward` method:

```rust
pub fn forward(
    &self, 
    x: Tensor<B, 2, Int>, 
    hidden_states: Option<Vec<Tensor<B, 2>>>
) -> (Tensor<B, 3>, Vec<Tensor<B, 2>>) { ... }
```

Both training and generation support hidden state passing for unlimited context length.

## Usage Guide

### Installation

Add the MinGRU crate to your Cargo.toml:

```toml
[dependencies]
mingru = { path = "path/to/mingru" }
burn = { version = "0.16.0", features = ["ndarray", "wgpu", "train"] }
```

Or clone and build the repository:

```bash
git clone https://github.com/yourusername/mingru.git
cd mingru
cargo build --release
```

### Training

Train a model with:

```bash
cargo run --release --bin train -- --data path/to/training/text.txt --output artifacts_dir
```

Options:
- `--data` - Path to training text file
- `--output` - Directory to save model artifacts

The training script will:
1. Process the text and build a character vocabulary
2. Save the vocabulary and configuration
3. Create training and validation splits
4. Train the model with the Burn Learner API
5. Save the trained model
6. Generate a sample text using the trained model

### Text Generation

Generate text with a trained model:

```bash
cargo run --release --bin generate -- \
    --model artifacts_dir/model_final.bin \
    --vocab artifacts_dir/vocab.txt \
    --seed "Your seed text" \
    --length 200 \
    --temperature 0.8
```

Options:
- `--model` - Path to the trained model file
- `--vocab` - Path to the vocabulary file
- `--seed` - Initial text to seed generation
- `--length` - Number of characters to generate
- `--temperature` - Sampling temperature (higher = more random)
- `--config` - Path to model configuration (optional)

The generation script also demonstrates long-context generation by processing a long text in chunks and continuing generation with the accumulated hidden state.

### Model Configuration

Customize the model with the following configuration:

```rust
let model_config = MinGRULMConfig::new(
    vocab.size(),  // num_tokens - vocabulary size
    256,           // dim - embedding and hidden dimension
    3,             // depth - number of MinGRU layers
    4.0,           // ff_mult - feed-forward dimension multiplier
    1.5,           // expansion_factor - MinGRU expansion factor
    256,           // chunk_size - size of chunks for processing
);
```

Training configuration:

```rust
let config = TrainingConfig::new(
    model_config,
    optimizer_config,
)
.with_sequence_length(128)  // context window size
.with_step_size(3)          // stride between training examples
.with_batch_size(32)        // batch size
.with_num_epochs(10)        // number of training epochs
.with_learning_rate(1e-3);  // learning rate
```

## Performance Considerations

1. **Memory Usage**
   - The parallel scan algorithm requires O(seq_len) memory
   - Chunked processing maintains constant memory regardless of context length
   - Hidden states are compact (hidden_dim * num_layers)

2. **Computation Efficiency**
   - Parallel mode: O(log n) steps for n tokens
   - Sequential mode: O(n) steps for n tokens
   - Chunked processing: O(n * log(chunk_size)) for n tokens

3. **Numerical Stability**
   - Log-space computation prevents underflow/overflow
   - Special g-function helps with gradient stability

4. **GPU Utilization**
   - Parallel scan is GPU-friendly
   - Tensor operations use Burn's optimized backend

## Development and Testing

### Testing Strategy

For comprehensive testing of the MinGRU implementation, consider:

1. **Unit Tests**
   - Parallel scan algorithm correctness
   - Log-space computation accuracy
   - MinGRU forward and backward pass
   - Hidden state passing correctness

2. **Integration Tests**
   - Model training convergence
   - Hidden state continuity across chunks
   - Generation quality with different contexts

3. **Performance Tests**
   - Memory usage with increasing context length
   - Computation time for different sequence lengths
   - Comparison with baseline GRU and Transformer models

### Testing Approach

```rust
// Example test for parallel scan correctness
#[test]
fn test_parallel_scan_simple_case() {
    // Initialize a simple sequence with known result
    let coeffs = tensor![[0.5, 0.5, 0.5]]; // a_t = 0.5 for all t
    let values = tensor![[1.0, 1.0, 1.0]]; // b_t = 1.0 for all t
    
    // Expected: h_t = 0.5 * h_{t-1} + 1.0
    // h_1 = 0.5 * 0 + 1.0 = 1.0
    // h_2 = 0.5 * 1.0 + 1.0 = 1.5
    // h_3 = 0.5 * 1.5 + 1.0 = 1.75
    let expected = tensor![[1.0, 1.5, 1.75]];
    
    let result = parallel_scan(coeffs, values, None);
    assert_tensor_close(result, expected, 1e-5);
}
```

## Future Improvements

1. **CUDA Kernels**
   - Implement custom CUDA kernels for the parallel scan
   - Optimize chunked processing

2. **Memory Optimization**
   - Implement gradient checkpointing for larger batch sizes
   - Optimize tensor operations for reduced memory footprint

3. **Architecture Enhancements**
   - Add attention capabilities for hybrid model support
   - Implement conditional computation for adaptive processing

4. **Training Improvements**
   - Support for distributed training
   - Implement quantization for faster inference

5. **Evaluation and Benchmarking**
   - Create benchmarks for different sequence lengths
   - Compare with Transformers and other RNN architectures

## References

1. MinGRU Paper: [Pre-print Link]
2. Parallel Scan Algorithm: [Paper/Reference Link]
3. LongMamba: [Repository Link]
4. Burn ML Framework: [https://github.com/tracel-ai/burn](https://github.com/tracel-ai/burn)