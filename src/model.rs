use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor, Int, activation},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, SwiGlu, SwiGluConfig},
    config::Config,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::mingru::{MinGRU, MinGRUConfig};

/// Feed Forward module config
#[derive(Config)]
pub struct FeedForwardConfig {
    dim: usize,
    #[config(default = "4.0")]
    mult: f64,
}

impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        let dim_inner = (self.dim as f64 * self.mult) as usize;
        
        FeedForward {
            swiglu: SwiGluConfig::new(self.dim, dim_inner).init(device),
            proj: LinearConfig::new(dim_inner, self.dim).init(device),
        }
    }
}

/// Feed Forward module
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    swiglu: SwiGlu<B>,
    proj: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.swiglu.forward(x);
        self.proj.forward(x)
    }
}

/// MinGRU Language Model config
#[derive(Config, Debug)]
pub struct MinGRULMConfig {
    num_tokens: usize,
    dim: usize,
    #[config(default = "2")]
    depth: usize,
    #[config(default = "2.0")]
    ff_mult: f64,
    #[config(default = "1.2")]
    expansion_factor: f64,
    #[config(default = "256")]
    chunk_size: usize,
}

impl MinGRULMConfig {
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
        // Token embedding
        let token_emb = EmbeddingConfig::new(self.num_tokens, self.dim).init(device);
        
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
    /// * `x` - Input token IDs [batch_size, seq_len]
    /// * `hidden_states` - Optional previous hidden states for each layer
    ///
    /// # Returns
    ///
    /// * `logits` - Output logits [batch_size, seq_len, vocab_size]
    /// * `next_hidden_states` - Next hidden states for each layer
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
    
        // Check if we need to chunk the sequence
        if seq_len > self.chunk_size && hidden_states.is_none() {
            // Process in chunks and carry hidden states between them
            return self.forward_chunked(x);
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
    fn forward_chunked(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Vec<Tensor<B, 2>>) {
        let [batch_size, seq_len, _] = x.dims();
        let _device = x.device();
        
        // Calculate number of chunks
        let num_chunks = (seq_len + self.chunk_size - 1) / self.chunk_size;
        let mut hidden_states = None;
        let mut outputs = Vec::with_capacity(num_chunks);
        
        // Process each chunk
        for chunk_idx in 0..num_chunks {
            let start_idx = chunk_idx * self.chunk_size;
            let end_idx = (start_idx + self.chunk_size).min(seq_len);
            
            // Extract current chunk
            let chunk = x.clone().slice([0..batch_size, start_idx..end_idx, 0..x.dims()[2]]);
            
            // Process chunk with current hidden states
            let (chunk_output, next_hidden_states) = self.forward_no_chunking(chunk, hidden_states);
            
            // Store output and update hidden states for next chunk
            outputs.push(chunk_output);
            hidden_states = Some(next_hidden_states);
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
        self.token_emb.weight.dims()[1] // embedding dimension is second dimension of weight matrix
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
            
            // Update hidden states
            current_hidden_states = Some(new_hidden_states);
            
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
