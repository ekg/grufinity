use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor, Int, activation},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    config::Config,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

// Conditionally import SwiGlu when feature is enabled
#[cfg(feature = "swiglu")]
use burn::nn::{SwiGlu, SwiGluConfig};

use crate::mingru::{MinGRU, MinGRUConfig};

/// Feed Forward module config
#[derive(Config)]
pub struct FeedForwardConfig {
    dim: usize,
    #[config(default = "4.0")]
    mult: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[cfg(feature = "ndarray")]
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    
    #[cfg(feature = "ndarray")]
    type TestBackend = NdArray<f32>;
    
    #[test]
    fn test_model_init() {
        let device = NdArrayDevice::default();
        
        // Create a small model config for testing
        let config = MinGRULMConfig::new(10, 32) // vocab_size=10, dim=32
            .with_depth(2)
            .with_chunk_size(16);
            
        // Initialize the model
        let model = config.init::<TestBackend>(&device);
        
        // Check model structure
        assert_eq!(model.token_emb.weight.dims(), [10, 32]); // shape is [vocab_size, dim]
        assert_eq!(model.mingru_layers.len(), 2); // depth
        assert_eq!(model.norm1_layers.len(), 2);
        assert_eq!(model.norm2_layers.len(), 2);
        assert_eq!(model.ff_layers.len(), 2);
        assert_eq!(model.to_logits.weight.dims(), [32, 10]); // shape is [dim, vocab_size]
        assert_eq!(model.chunk_size, 16);
    }
    
    #[test]
    fn test_model_forward() {
        let device = NdArrayDevice::default();
        
        // Create a small model config for testing
        let config = MinGRULMConfig::new(10, 32)
            .with_depth(1)  // Single layer for simpler testing
            .with_chunk_size(16);
            
        // Initialize the model
        let model = config.init::<TestBackend>(&device);
        
        // Create input tensor - batch of 2, sequence length of 4
        let input_data: Vec<i32> = vec![
            1, 2, 3, 4,  // Sample 1
            5, 6, 7, 8,  // Sample 2
        ];
        
        let input = Tensor::<TestBackend, 1, Int>::from_data(&input_data[..], &device)
            .reshape([2, 4]);
        
        // Run forward pass
        let (logits, hidden_states) = model.forward(input, None);
        
        // Check output shapes
        assert_eq!(logits.dims(), [2, 4, 10]); // [batch_size, seq_len, vocab_size]
        assert_eq!(hidden_states.len(), 1); // One per layer
        assert_eq!(hidden_states[0].dims(), [2, 38]); // [batch_size, hidden_dim] is 38 due to expansion factor
    }
    
    #[test]
    fn test_parameter_count() {
        // Test that parameter count calculation is consistent
        let config = MinGRULMConfig::new(256, 512)
            .with_depth(3)
            .with_ff_mult(4.0)
            .with_expansion_factor(1.5);
            
        let param_count = config.calculate_parameters();
        
        // Not checking exact count, but verifying it's in a reasonable range
        assert!(param_count > 1_000_000); // Should have over 1M parameters
        assert!(param_count < 50_000_000); // But under 50M
        
        // Test dimension calculation
        let target_params = 10_000_000; // 10M
        let dim = config.compute_dim_for_param_count(target_params);
        
        // Verify dimension is a multiple of 32
        assert_eq!(dim % 32, 0);
        
        // Create new config with that dimension and verify parameter count is close
        let new_config = MinGRULMConfig::new(256, dim)
            .with_depth(3)
            .with_ff_mult(4.0)
            .with_expansion_factor(1.5);
            
        let new_param_count = new_config.calculate_parameters();
        
        // Should be within 10% of target
        let ratio = new_param_count as f64 / target_params as f64;
        assert!(ratio > 0.9 && ratio < 1.1);
    }
}

#[cfg(feature = "swiglu")]
impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        let dim_inner = (self.dim as f64 * self.mult) as usize;
        
        FeedForward {
            swiglu: SwiGluConfig::new(self.dim, dim_inner).init(device),
            proj: LinearConfig::new(dim_inner, self.dim).init(device),
        }
    }
}

#[cfg(not(feature = "swiglu"))]
impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        let dim_inner = (self.dim as f64 * self.mult) as usize;
        
        FeedForward {
            w1: LinearConfig::new(self.dim, dim_inner).init(device),
            w2: LinearConfig::new(self.dim, dim_inner).init(device),
            proj: LinearConfig::new(dim_inner, self.dim).init(device),
        }
    }
}

/// Feed Forward module with SwiGLU activation (when feature is enabled)
#[cfg(feature = "swiglu")]
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    swiglu: SwiGlu<B>,
    proj: Linear<B>,
}

#[cfg(feature = "swiglu")]
impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.swiglu.forward(x);
        self.proj.forward(x)
    }
}

/// Feed Forward module with SiLU activation (default)
#[cfg(not(feature = "swiglu"))]
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    w1: Linear<B>,
    w2: Linear<B>,
    proj: Linear<B>,
}

#[cfg(not(feature = "swiglu"))]
impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x1 = self.w1.forward(x.clone());
        let x2 = self.w2.forward(x);
        let x = x1 * activation::silu(x2); // SiLU implementation matching SwiGLU behavior
        self.proj.forward(x)
    }
}

/// MinGRU Language Model configuration.
///
/// This configuration defines a GRU-based language model with the following architecture:
/// 1. Token embedding layer
/// 2. Multiple MinGRU layers with feed-forward networks
/// 3. Normalization layers (RMSNorm)
/// 4. Final projection to vocabulary logits
///
/// The model processes text in chunks and can handle arbitrarily long sequences with
/// constant memory usage by preserving hidden states between chunks.
#[derive(Config, Debug)]
pub struct MinGRULMConfig {
    /// Number of tokens in vocabulary
    num_tokens: usize,
    
    /// Hidden dimension size throughout the model
    dim: usize,
    
    /// Number of MinGRU layers in the model
    #[config(default = "2")]
    depth: usize,
    
    /// Multiplier for feed-forward layer dimension
    /// Controls the size of feed-forward networks as: dim * ff_mult
    #[config(default = "2.0")]
    ff_mult: f64,
    
    /// Expansion factor for MinGRU internal representations
    /// Higher values increase model capacity with minimal computational overhead
    #[config(default = "1.2")]
    expansion_factor: f64,
    
    /// Size of chunks (in tokens) for processing long sequences
    /// Affects both memory usage and context window size
    #[config(default = "256")]
    chunk_size: usize,
}


impl MinGRULMConfig {
    /// Set the dimension and return a new config
    pub fn with_dim(&self, dim: usize) -> Self {
        Self {
            num_tokens: self.num_tokens,
            dim,
            depth: self.depth,
            ff_mult: self.ff_mult,
            expansion_factor: self.expansion_factor,
            chunk_size: self.chunk_size,
        }
    }

    /// Calculate the total parameter count for this model configuration
    pub fn calculate_parameters(&self) -> usize {
        let vocab_size = self.num_tokens;
        let dim = self.dim;
        let depth = self.depth;
        let expansion_factor = self.expansion_factor;
        let ff_mult = self.ff_mult;
        
        // 1. Token embedding
        let token_emb_params = vocab_size * dim;
        
        // 2. MinGRU layers
        let mut mingru_params = 0;
        for _i in 0..depth {
            // Input size is 'dim' for all layers
            let input_size = dim;
            let dim_inner = (dim as f64 * expansion_factor) as usize;
            
            // to_hidden_and_gate: input_size × (dim_inner * 2)
            let to_hidden_and_gate_params = input_size * (dim_inner * 2);
            
            // to_out (if expansion_factor != 1.0): dim_inner × dim
            let to_out_params = if expansion_factor != 1.0 {
                dim_inner * dim
            } else {
                0
            };
            
            mingru_params += to_hidden_and_gate_params + to_out_params;
        }
        
        // 3. Feed-forward layers
        let mut ff_params = 0;
        for _ in 0..depth {
            let dim_inner = (dim as f64 * ff_mult) as usize;
            
            #[cfg(feature = "swiglu")]
            {
                // SwiGLU variant: dim × dim_inner (gate) + dim_inner × dim (proj)
                ff_params += dim * dim_inner + dim_inner * dim;
            }
            
            #[cfg(not(feature = "swiglu"))]
            {
                // SiLU variant: dim × dim_inner (w1) + dim × dim_inner (w2) + dim_inner × dim (proj)
                ff_params += dim * dim_inner + dim * dim_inner + dim_inner * dim;
            }
        }
        
        // 4. Normalization layers (approximate as dim per layer)
        let norm_params = depth * 2 * dim + dim; // 2 norms per layer + final norm
        
        // 5. Output projection
        let output_proj_params = dim * vocab_size;
        
        // Total parameters
        token_emb_params + mingru_params + ff_params + norm_params + output_proj_params
    }
    
    /// Round a dimension to the nearest multiple of 32
    fn round_to_multiple_of_32(dim: usize) -> usize {
        let remainder = dim % 32;
        if remainder == 0 {
            return dim; // Already a multiple of 32
        }
        
        // Round to nearest multiple of 32
        if remainder < 16 {
            dim - remainder // Round down
        } else {
            dim + (32 - remainder) // Round up
        }
    }
    
    /// Compute the dimension needed to achieve a target parameter count
    pub fn compute_dim_for_param_count(&self, target_params: usize) -> usize {
        // We'll use a binary search to find the optimal dimension
        let mut min_dim = 32; // Minimum reasonable dimension (already a multiple of 32)
        let mut max_dim = 8192; // Maximum reasonable dimension (already a multiple of 32)
        
        while min_dim <= max_dim {
            let mid_dim = (min_dim + max_dim) / 2;
            let config = self.clone().with_dim(mid_dim);
            let params = config.calculate_parameters();
            
            if params < target_params {
                min_dim = mid_dim + 1;
            } else if params > target_params {
                max_dim = mid_dim - 1;
            } else {
                // Exact match - round to nearest multiple of 32
                return Self::round_to_multiple_of_32(mid_dim);
            }
        }
        
        // Return the dimension that gives us just enough parameters
        let under_config = self.clone().with_dim(max_dim);
        let over_config = self.clone().with_dim(min_dim);
        
        let under_params = under_config.calculate_parameters();
        let over_params = over_config.calculate_parameters();
        
        // Find which is closer to the target before rounding
        let dim = if (target_params as i64 - under_params as i64).abs() < 
                    (target_params as i64 - over_params as i64).abs() {
            max_dim
        } else {
            min_dim
        };
        
        // Round to nearest multiple of 32
        Self::round_to_multiple_of_32(dim)
    }

    /// Get number of tokens (vocabulary size)
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }
    
    /// Get model dimension
    pub fn dim(&self) -> usize {
        self.dim
    }
    
    /// Get number of layers
    pub fn depth(&self) -> usize {
        self.depth
    }
    
    /// Get feed-forward multiplier
    pub fn ff_mult(&self) -> f64 {
        self.ff_mult
    }
    
    /// Get expansion factor
    pub fn expansion_factor(&self) -> f64 {
        self.expansion_factor
    }
    
    /// Get chunk size
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }
    
    pub fn init<B: Backend>(&self, device: &B::Device) -> MinGRULM<B> {
        // Ensure dimension is a multiple of 32 for optimal CUDA memory alignment
        let dim = Self::round_to_multiple_of_32(self.dim);
        
        // Token embedding
        let token_emb = EmbeddingConfig::new(self.num_tokens, dim).init(device);
        
        // Create layers
        let mut mingru_layers = Vec::with_capacity(self.depth);
        let mut norm1_layers = Vec::with_capacity(self.depth);
        let mut norm2_layers = Vec::with_capacity(self.depth);
        let mut ff_layers = Vec::with_capacity(self.depth);
        
        for i in 0..self.depth {
            // Layer norms
            norm1_layers.push(RmsNormConfig::new(self.dim).init(device));
            norm2_layers.push(RmsNormConfig::new(self.dim).init(device));
            
            // MinGRU layers
            let input_size = if i == 0 { self.dim } else { self.dim };
            let mingru_config = MinGRUConfig::new(input_size, self.dim)
                .with_expansion_factor(self.expansion_factor);
            mingru_layers.push(mingru_config.init(device));
            
            // Feed forward layers
            let ff_config = FeedForwardConfig::new(self.dim).with_mult(self.ff_mult);
            ff_layers.push(ff_config.init(device));
        }
        
        // Output normalization and projection
        let norm = RmsNormConfig::new(self.dim).init(device);
        let to_logits = LinearConfig::new(self.dim, self.num_tokens)
            .with_bias(false)
            .init(device);
        
        MinGRULM {
            token_emb,
            mingru_layers,
            norm1_layers,
            norm2_layers,
            ff_layers,
            norm,
            to_logits,
            chunk_size: self.chunk_size,
        }
    }
}

/// MinGRU Language Model
#[derive(Module, Debug)]
pub struct MinGRULM<B: Backend> {
    token_emb: Embedding<B>,
    mingru_layers: Vec<MinGRU<B>>,
    norm1_layers: Vec<RmsNorm<B>>,
    norm2_layers: Vec<RmsNorm<B>>,
    ff_layers: Vec<FeedForward<B>>,
    norm: RmsNorm<B>,
    to_logits: Linear<B>,
    #[module(skip)]
    chunk_size: usize,
}

impl<B: Backend> MinGRULM<B> {
    /// Forward pass of the model
    ///
    /// # Arguments
    ///
    /// * `x` - Input token IDs with shape [batch_size, seq_len]
    /// * `hidden_states` - Optional previous hidden states for each layer
    ///
    /// # Returns
    ///
    /// * `logits` - Output logits with shape [batch_size, seq_len, vocab_size]
    /// * `next_hidden_states` - Next hidden states for each layer
    ///
    /// # Tensor Shapes
    ///
    /// Input:
    /// - x: [batch_size, seq_len] of type Int - Token IDs
    /// - hidden_states (optional): Vec of [batch_size, hidden_size] tensors - One per layer
    ///
    /// Output:
    /// - logits: [batch_size, seq_len, vocab_size] - Prediction scores for each token
    /// - next_hidden_states: Vec of [batch_size, hidden_size] tensors - Updated states
    pub fn forward(
        &self, 
        x: Tensor<B, 2, Int>, 
        hidden_states: Option<Vec<Tensor<B, 2>>>
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 2>>) {
        let [_batch_size, seq_len] = x.dims();
    
        // Embed tokens
        let mut x = self.token_emb.forward(x);
    
        // Initialize hidden states
        let mut next_hidden_states = Vec::with_capacity(self.mingru_layers.len());
    
        // Check if we need to chunk the sequence - only based on sequence length
        if seq_len > self.chunk_size {
            // Process in chunks and carry hidden states between them
            return self.forward_chunked(x, hidden_states);
        }
    
        // Process through layers
        for (idx, (((mingru, norm1), norm2), ff)) in self.mingru_layers.iter()
            .zip(self.norm1_layers.iter())
            .zip(self.norm2_layers.iter())
            .zip(self.ff_layers.iter())
            .enumerate()
        {
            // Get previous hidden state for this layer if available
            let prev_hidden = hidden_states.as_ref()
                .and_then(|states| states.get(idx).cloned());
        
            // MinGRU with residual connection
            let x_norm = norm1.forward(x.clone());
            let (mingru_out, hidden) = mingru.forward(x_norm, prev_hidden);
            x = x + mingru_out;
            next_hidden_states.push(hidden);
            
            // Feed Forward with residual connection
            let x_norm = norm2.forward(x.clone());
            let ff_out = ff.forward(x_norm);
            x = x + ff_out;
        }
        
        // Final normalization and projection to logits
        let x = self.norm.forward(x);
        let logits = self.to_logits.forward(x);
        
        (logits, next_hidden_states)
    }
    
    /// Process a sequence in chunks, carrying hidden states between chunks
    fn forward_chunked(&self, x: Tensor<B, 3>, initial_hidden_states: Option<Vec<Tensor<B, 2>>>) -> (Tensor<B, 3>, Vec<Tensor<B, 2>>) {
        let [batch_size, seq_len, _] = x.dims();
        let _device = x.device();
        
        // Calculate number of chunks
        let num_chunks = (seq_len + self.chunk_size - 1) / self.chunk_size;
        let mut hidden_states = initial_hidden_states; // Start with provided hidden states
        let mut outputs = Vec::with_capacity(num_chunks);
        
        // Process each chunk
        for chunk_idx in 0..num_chunks {
            let start_idx = chunk_idx * self.chunk_size;
            let end_idx = (start_idx + self.chunk_size).min(seq_len);
            
            // Extract current chunk
            let chunk = x.clone().slice([0..batch_size, start_idx..end_idx, 0..x.dims()[2]]);
            
            // Process chunk with current hidden states
            let (chunk_output, next_hidden_states) = self.forward_no_chunking(chunk, hidden_states);
            
            // Apply tanh nonlinearity to hidden states before passing to next chunk (if feature enabled)
            #[cfg(feature = "tanh")]
            let next_hidden_states_processed = next_hidden_states.iter()
                .map(|h| h.clone().tanh())
                .collect();
                
            #[cfg(not(feature = "tanh"))]
            let next_hidden_states_processed = next_hidden_states;
            
            // Store output and update hidden states
            outputs.push(chunk_output);
            hidden_states = Some(next_hidden_states_processed);
        }
        
        // Concatenate outputs from all chunks
        let output = Tensor::cat(outputs, 1);
        
        (output, hidden_states.unwrap_or_default())
    }
    
    /// Forward pass without chunking, used by forward_chunked
    /// Get model configuration
    pub fn config(&self) -> MinGRULMConfig {
        // Use the actual dimensions from the model
        let vocab_size = self.to_logits.weight.dims()[0]; // output dimension is first dimension of weight matrix
        let hidden_dim = self.dim();
        
        MinGRULMConfig::new(vocab_size, hidden_dim)
            .with_depth(self.mingru_layers.len())
            .with_chunk_size(self.chunk_size)
    }
    
    /// Get the hidden dimension of the model
    pub fn hidden_dim(&self) -> usize {
        self.dim()
    }
    
    /// Accessor for mingru layers
    pub fn mingru_layers(&self) -> &Vec<MinGRU<B>> {
        &self.mingru_layers
    }
    
    /// Get hidden dimension for other modules
    pub fn dim(&self) -> usize {
        self.token_emb.weight.dims()[0] // embedding dimension is first dimension of weight matrix
    }

    fn forward_no_chunking(
        &self, 
        x: Tensor<B, 3>, 
        hidden_states: Option<Vec<Tensor<B, 2>>>
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 2>>) {
        let mut x = x;
        
        // Initialize hidden states
        
        // Initialize hidden states
        let mut next_hidden_states = Vec::with_capacity(self.mingru_layers.len());
        
        // Process through layers
        for (idx, (((mingru, norm1), norm2), ff)) in self.mingru_layers.iter()
            .zip(self.norm1_layers.iter())
            .zip(self.norm2_layers.iter())
            .zip(self.ff_layers.iter())
            .enumerate()
        {
            // Get previous hidden state for this layer if available
            let prev_hidden = hidden_states.as_ref()
                .and_then(|states| states.get(idx).cloned());
            
            // MinGRU with residual connection
            let x_norm = norm1.forward(x.clone());
            let (mingru_out, hidden) = mingru.forward(x_norm, prev_hidden);
            x = x + mingru_out;
            next_hidden_states.push(hidden);
            
            // Feed Forward with residual connection
            let x_norm = norm2.forward(x.clone());
            let ff_out = ff.forward(x_norm);
            x = x + ff_out;
        }
        
        // Final normalization and projection to logits
        let x = self.norm.forward(x);
        let logits = self.to_logits.forward(x);
        
        (logits, next_hidden_states)
    }
    
    /// Generate text autoregressively with hidden state passing
    pub fn generate(
        &self,
        prompt: Tensor<B, 2, Int>,
        max_tokens: usize,
        temperature: f64,
        hidden_states: Option<Vec<Tensor<B, 2>>>,
    ) -> (Tensor<B, 2, Int>, Vec<Tensor<B, 2>>) {
        let [batch_size, prompt_len] = prompt.dims();
        let device = prompt.device();
        
        // Guard against empty prompts
        if prompt_len == 0 {
            let empty_tensor = Tensor::zeros([batch_size, 0], &device);
            return (empty_tensor, hidden_states.unwrap_or_default());
        }
        
        let mut tokens = prompt.clone();
        let mut current_hidden_states = hidden_states;
        
        // Generate tokens one by one
        for _ in 0..max_tokens {
            // Get the last token - add a check to avoid issues with empty tensors
            let token_len = tokens.dims()[1];
            if token_len == 0 {
                break;
            }
            
            // Make sure we have valid batch dimension
            let last_token = tokens.clone().slice([0..batch_size, token_len-1..token_len]);
            
            // Forward pass with hidden state
            let (logits, new_hidden_states) = self.forward(last_token, current_hidden_states);
            
            // Apply tanh nonlinearity to hidden states (if feature enabled)
            #[cfg(feature = "tanh")]
            let next_hidden_states_processed = new_hidden_states.iter()
                .map(|h| h.clone().tanh())
                .collect();
                
            #[cfg(not(feature = "tanh"))]
            let next_hidden_states_processed = new_hidden_states;
            
            // Update hidden states
            current_hidden_states = Some(next_hidden_states_processed);
            
            // Get next token by sampling
            let next_token = self.sample_token(logits, temperature);
            
            // Reshape to ensure proper dimensions for concatenation
            let next_token_reshaped = next_token.reshape([batch_size, 1]);
            
            // Append to generated tokens
            tokens = Tensor::cat(vec![tokens, next_token_reshaped], 1);
        }
        
        (tokens, current_hidden_states.unwrap_or_default())
    }
    
    /// Sample next token from logits with temperature
    fn sample_token(&self, logits: Tensor<B, 3>, temperature: f64) -> Tensor<B, 1, Int> {
        let device = logits.device();
        let [batch_size, seq_len, vocab_size] = logits.dims();
        
        // Safety check for empty tensors
        if seq_len == 0 {
            return Tensor::zeros([batch_size], &device);
        }
        
        // Get logits for last position
        let last_logits = logits.slice([0..batch_size, seq_len-1..seq_len, 0..vocab_size]).squeeze::<2>(1);
        
        // Apply temperature
        let scaled_logits = if temperature != 1.0 {
            last_logits / temperature as f32
        } else {
            last_logits
        };
        
        // Sample from the distribution
        let probs = activation::softmax(scaled_logits, 1);
        
        // For simplicity, just take argmax here
        // In a real implementation, you'd want to sample from the distribution
        let result = probs.argmax(1);
        
        // Ensure we get a 1D tensor (shape [batch_size]) 
        // Make sure we return the correct shape
        if result.dims()[0] != batch_size {
            return Tensor::zeros([batch_size], &device);
        }
        // Squeeze to ensure we return a 1D tensor as required by the function signature
        result.squeeze::<1>(0)
    }
}

/// Implementation of TrainStep for MinGRULM
impl<B: burn::tensor::backend::AutodiffBackend> TrainStep<TextBatch<B>, ClassificationOutput<B>> for MinGRULM<B> {
    fn step(&self, batch: TextBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        // Forward pass
        let (logits, _) = self.forward(batch.input, None);
        
        // Reshape logits and targets for loss calculation
        let [batch_size, seq_len, vocab_size] = logits.dims();
        
        // Shift targets (predict next token)
        let targets = batch.target;
        
        // Reshape for classification output
        let logits_reshaped = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_reshaped = targets.reshape([batch_size * seq_len]);
        
        // Compute cross-entropy loss
        let loss_fn = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits_reshaped.device());
        let loss = loss_fn.forward(logits_reshaped.clone(), targets_reshaped.clone());
        
        // Create output and compute gradients
        let output = ClassificationOutput::new(loss.clone(), logits_reshaped, targets_reshaped);
        let grads = loss.backward();
        
        TrainOutput::new(self, grads, output)
    }
}

/// Implementation of ValidStep for MinGRULM
impl<B: Backend> ValidStep<TextBatch<B>, ClassificationOutput<B>> for MinGRULM<B> {
    fn step(&self, batch: TextBatch<B>) -> ClassificationOutput<B> {
        // Forward pass (no gradients needed)
        let (logits, _) = self.forward(batch.input, None);
        
        // Reshape logits and targets for loss calculation
        let [batch_size, seq_len, vocab_size] = logits.dims();
        
        // Shift targets (predict next token)
        let targets = batch.target;
        
        // Reshape for classification output
        let logits_reshaped = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_reshaped = targets.reshape([batch_size * seq_len]);
        
        // Compute cross-entropy loss
        let loss_fn = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits_reshaped.device());
        let loss = loss_fn.forward(logits_reshaped.clone(), targets_reshaped.clone());
        
        ClassificationOutput::new(loss, logits_reshaped, targets_reshaped)
    }
}

/// Batch structure for text data
#[derive(Debug, Clone)]
pub struct TextBatch<B: Backend> {
    pub input: Tensor<B, 2, Int>,
    pub target: Tensor<B, 2, Int>,
}
