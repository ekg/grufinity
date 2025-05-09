use burn::data::dataloader::batcher::Batcher;
use burn::{
    config::Config,
    module::{AutodiffModule, Module},
    nn::loss::CrossEntropyLossConfig,
    optim::{GradientsAccumulator, GradientsParams, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{backend::AutodiffBackend, cast::ToElement, Tensor},
    train::metric::MetricEntry,
};

/// Result of processing a batch during training
#[derive(Debug)]
enum BatchResult {
    /// Batch was skipped due to invalid data
    Skip,
    /// Batch was processed successfully
    Success { loss: f32 },
}

/// Result of processing a validation batch
#[derive(Debug)]
enum ValidationResult {
    /// Batch was skipped due to invalid data
    Skip,
    /// Batch was processed successfully
    Success { loss: f32 },
}

#[cfg(feature = "optimizer-sgd")]
use burn::optim::SgdConfig;

#[cfg(feature = "optimizer-adam")]
use burn::{optim::decay::WeightDecayConfig, optim::AdamConfig};

// Enable Adam by default if no optimizer is specified
#[cfg(not(any(feature = "optimizer-sgd", feature = "optimizer-adam")))]
use burn::optim::AdamConfig;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::fmt::Debug;
use std::io;
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;

use crate::dataset::{
    CharVocab, ChunkedTextBatch, ChunkedTextBatcher, ContinuousChunkedTextDataset, TextBatcher,
};
use crate::model::{MinGRULM, MinGRULMConfig};
use burn::data::dataset::Dataset;
use burn::record::FileRecorder;

/// Learning rate scheduler type
#[derive(Config, Debug, Copy, PartialEq)]
pub enum LRSchedulerType {
    /// Constant learning rate
    Constant,
    /// Cosine annealing learning rate
    Cosine,
    /// Linear decay learning rate
    Linear,
}

impl Default for LRSchedulerType {
    fn default() -> Self {
        LRSchedulerType::Constant
    }
}

/// Convert string to LRSchedulerType
impl FromStr for LRSchedulerType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "constant" => Ok(LRSchedulerType::Constant),
            "cosine" => Ok(LRSchedulerType::Cosine),
            "linear" => Ok(LRSchedulerType::Linear),
            _ => Err(format!("Unknown scheduler type: {}", s)),
        }
    }
}

/// Learning rate scheduler that works with both Adam and SGD
pub struct LearningRateScheduler {
    base_lr: f64,
    current_lr: f64,
    min_lr_factor: f64,
    scheduler_type: LRSchedulerType,
    warmup_epochs: usize,
    total_epochs: usize,
    reduce_factor: f64,     // Factor to reduce LR when plateau is detected
    reduce_threshold: f64,  // % decrease threshold to consider improvement
    last_valid_loss: f32,   // Last validation loss for comparison
    plateau_count: usize,   // Count of plateaus detected
    has_reduced: bool,      // Whether we've reduced the learning rate before
    stall_counter: usize,   // Counter for stalling progress after reduction
    stall_epochs: usize,    // How many epochs of stall to trigger an increase
    stall_threshold: f64,   // Improvement threshold below which an epoch is considered stalled
    plateau_counter: usize, // Counter for consecutive plateau epochs
    plateau_epochs: usize,  // Number of plateau epochs to trigger reduction
}

impl LearningRateScheduler {
    pub fn new(
        base_lr: f64,
        min_lr_factor: f64,
        scheduler_type: LRSchedulerType,
        warmup_epochs: usize,
        total_epochs: usize,
        reduce_threshold: f64,
        reduce_factor: f64,
        stall_epochs: usize,
        stall_threshold: f64,
        plateau_epochs: usize,
    ) -> Self {
        let current_lr = if warmup_epochs > 0 {
            // Start with min_lr if warmup is enabled
            base_lr * min_lr_factor
        } else {
            base_lr
        };

        Self {
            base_lr,
            current_lr,
            min_lr_factor,
            scheduler_type,
            warmup_epochs,
            total_epochs,
            reduce_factor,
            reduce_threshold,
            last_valid_loss: f32::MAX,
            plateau_count: 0,
            has_reduced: false,
            stall_counter: 0,
            stall_epochs,
            stall_threshold,
            plateau_counter: 0,
            plateau_epochs,
        }
    }

    pub fn get_lr_for_epoch(&mut self, epoch: usize) -> f64 {
        let min_lr = self.base_lr * self.min_lr_factor;

        // Calculate the scheduled learning rate without plateau modifications
        let scheduled_lr = if epoch <= self.warmup_epochs {
            // Linear warmup from min_lr to base_lr
            min_lr + (self.base_lr - min_lr) * (epoch as f64 / self.warmup_epochs.max(1) as f64)
        } else if self.scheduler_type == LRSchedulerType::Cosine {
            // Cosine annealing
            let progress = (epoch - self.warmup_epochs) as f64
                / (self.total_epochs - self.warmup_epochs).max(1) as f64;
            let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            min_lr + (self.base_lr - min_lr) * cosine_decay
        } else if self.scheduler_type == LRSchedulerType::Linear {
            // Linear decay
            let progress = (epoch - self.warmup_epochs) as f64
                / (self.total_epochs - self.warmup_epochs).max(1) as f64;
            self.base_lr - (self.base_lr - min_lr) * progress.min(1.0)
        } else {
            // Constant learning rate
            self.base_lr
        };

        // Set the current learning rate to scheduled_lr
        self.current_lr = scheduled_lr;

        self.current_lr
    }

    pub fn get_current_lr(&self) -> f64 {
        self.current_lr
    }

    /// Check if we should reduce learning rate based on validation loss
    pub fn check_reduce_on_plateau(&mut self, valid_loss: f32) -> bool {
        // If threshold is 0, the feature is disabled
        if self.reduce_threshold <= 0.0 {
            return false;
        }

        // If this is the first check, just store the loss
        if self.last_valid_loss == f32::MAX {
            self.last_valid_loss = valid_loss;
            return false;
        }

        // Calculate percentage improvement
        let improvement = (self.last_valid_loss - valid_loss) / self.last_valid_loss;

        // Always print the improvement percentage
        println!(
            "Loss improvement: {:.2}% (previous: {:.6}, current: {:.6})",
            improvement * 100.0,
            self.last_valid_loss,
            valid_loss
        );

        // Update last loss for next comparison
        self.last_valid_loss = valid_loss;

        // Check if we're in a stall situation (after previous reduction)
        if self.stall_epochs > 0 && self.has_reduced && improvement < self.stall_threshold as f32 {
            // Only track and report stalls if stall_epochs > 0 (explicitly enabled)
            // Increment stall counter
            self.stall_counter += 1;
            println!("🔍 Potential learning rate stall detected: {} consecutive epochs with <{}% improvement", 
                     self.stall_counter, self.stall_threshold * 100.0);

            // If stall_epochs is enabled (>0) and we've stalled for that many epochs, increase the learning rate
            if self.stall_epochs > 0 && self.stall_counter >= self.stall_epochs {
                // Increase learning rate by the reciprocal of the reduce factor (e.g., if reduce=0.1, increase by 10x)
                let increased_lr = self.current_lr / self.reduce_factor;

                // Apply the increase to the base learning rate
                self.base_lr = self.base_lr / self.reduce_factor;

                // Update current learning rate
                self.current_lr = increased_lr;

                // Reset stall counter
                self.stall_counter = 0;

                println!(
                    "🚀 Learning rate increased to {:.6e} to escape plateau",
                    self.current_lr
                );
                return true;
            }
        } else if self.stall_epochs > 0 && improvement >= self.stall_threshold as f32 {
            // Only reset stall counter if feature is enabled
            // Good improvement, reset stall counter
            if self.stall_counter > 0 {
                println!(
                    "✅ Good improvement detected ({}%), resetting stall counter",
                    improvement * 100.0
                );
            }
            self.stall_counter = 0;
        }

        // If improvement is below threshold, increment plateau counter
        if improvement < self.reduce_threshold as f32 {
            self.plateau_count += 1;
            self.plateau_counter += 1;

            println!(
                "🔍 Potential plateau detected: {} consecutive epochs with <{}% improvement",
                self.plateau_counter,
                self.reduce_threshold * 100.0
            );

            // Only reduce learning rate if we've hit the plateau_epochs counter
            if self.plateau_counter >= self.plateau_epochs {
                // Calculate new learning rate with reduction
                let reduced_lr = self.current_lr * self.reduce_factor;

                // Only apply if it would actually reduce the LR
                if reduced_lr < self.current_lr {
                    // Apply the reduction to our actual base learning rate as well
                    // This is key to ensuring future epoch calculations use the reduced base
                    self.base_lr = self.base_lr * self.reduce_factor;

                    // Update current learning rate
                    self.current_lr = reduced_lr;

                    // Mark that we've reduced the learning rate
                    self.has_reduced = true;

                    // Reset stall counter when we reduce
                    self.stall_counter = 0;

                    // Reset plateau counter after taking action
                    self.plateau_counter = 0;

                    println!(
                        "🔥 Learning rate reduced to {:.6e} due to plateau",
                        self.current_lr
                    );
                    return true;
                }
            }
        } else {
            // Reset plateau counter on sufficient improvement
            self.plateau_counter = 0;
            self.plateau_count = 0;
        }

        false
    }
}

/// Configuration for TBPTT training
#[derive(Config)]
pub struct TBPTTConfig {
    /// Model configuration
    pub model: MinGRULMConfig,

    // Use conditional compilation to ensure we only have one optimizer field
    // based on which feature is enabled
    #[cfg(feature = "optimizer-sgd")]
    /// SGD Optimizer configuration
    #[config(default = "SgdConfig::new()")]
    pub optimizer: SgdConfig,

    #[cfg(all(feature = "optimizer-adam", not(feature = "optimizer-sgd")))]
    /// Adam Optimizer configuration
    #[config(default = "AdamConfig::new()")]
    pub optimizer: AdamConfig,

    #[cfg(all(not(feature = "optimizer-sgd"), not(feature = "optimizer-adam")))]
    /// Default optimizer (Adam) when no optimizer feature is specified
    #[config(default = "AdamConfig::new()")]
    pub optimizer: AdamConfig,

    // Store our own copies of Adam parameters (since they're private in AdamConfig)
    #[cfg(any(feature = "optimizer-adam", not(feature = "optimizer-sgd")))]
    #[config(default = 0.9)]
    pub adam_beta1: f32,

    #[cfg(any(feature = "optimizer-adam", not(feature = "optimizer-sgd")))]
    #[config(default = 0.999)]
    pub adam_beta2: f32,

    #[cfg(any(feature = "optimizer-adam", not(feature = "optimizer-sgd")))]
    #[config(default = 1e-8)]
    pub adam_epsilon: f32,

    /// Learning rate - used for both SGD and Adam
    /// For Adam, this is passed during optimizer.step() calls
    #[config(default = 1e-3)]
    pub learning_rate: f64,

    /// Frequency of parameter updates (k1 parameter)
    /// Updates parameters after processing this many chunks
    #[config(default = 4)]
    pub tbptt_k1: usize,

    /// Length of backpropagation window (k2 parameter)
    /// Gradients flow back through this many chunks (chunk_size * tbptt_k2 total tokens)
    #[config(default = 8)]
    pub tbptt_k2: usize,

    /// Size of each chunk in tokens
    #[config(default = 64)]
    pub chunk_size: usize,

    /// Maximum number of chunks to process per epoch
    /// This determines the effective sequence length: chunk_size * max_chunks_per_epoch
    #[config(default = 1000)]
    pub max_chunks_per_epoch: usize,

    /// Whether to preserve hidden states between batches
    #[config(default = true)]
    pub preserve_hidden_states: bool,
    
    /// Whether to use random sequence sampling instead of TBPTT
    #[config(default = false)]
    pub random_sampling: bool,

    /// Random seed for reproducibility
    #[config(default = 42)]
    pub seed: u64,

    /// Number of epochs to train
    #[config(default = 10)]
    pub num_epochs: usize,

    /// Maximum number of epochs if training to target loss
    #[config(default = 1000)]
    pub max_epochs: usize,

    /// Target validation loss to stop training (0.0 to ignore)
    #[config(default = 0.0)]
    pub target_valid_loss: f32,

    /// Target test loss to stop training (0.0 to ignore)
    #[config(default = 0.0)]
    pub target_test_loss: f32,

    /// Learning rate scheduler type
    #[config(default = "LRSchedulerType::Cosine")]
    pub lr_scheduler: LRSchedulerType,

    /// Minimum learning rate for scheduler (as fraction of base lr)
    #[config(default = 0.1)]
    pub min_lr_factor: f64,

    /// Number of warmup epochs
    #[config(default = 0)]
    pub warmup_epochs: usize,

    /// Plateau threshold - improvement % required to avoid reducing learning rate
    /// Set to 0.0 to disable (default: 0.001 = 0.1%)
    #[config(default = 0.001)]
    pub plateau_threshold: f64,

    /// Plateau factor - amount to reduce learning rate by when plateau is detected
    /// (default: 0.1 = reduce to 10% of current rate)
    #[config(default = 0.1)]
    pub plateau_factor: f64,

    /// Stall threshold - improvement percentage below which an epoch is considered stalled
    /// (default: 0.01 = 1% improvement)
    #[config(default = 0.01)]
    pub stall_threshold: f64,

    /// Stall epochs - number of epochs with low improvement to trigger LR increase
    /// (default: 0 means disabled, set to a positive value to enable)
    #[config(default = 0)]
    pub stall_epochs: usize,

    /// Plateau epochs - number of consecutive epochs with minimal improvement before reducing LR
    /// (default: 2 epochs)
    #[config(default = 2)]
    pub plateau_epochs: usize,

    /// Batch size
    #[config(default = 32)]
    pub batch_size: usize,

    /// Log interval (in batches)
    #[config(default = 10)]
    pub log_interval: usize,

    /// Checkpoint interval (in epochs)
    #[config(default = 1)]
    pub checkpoint_interval: usize,

    /// Gradient clipping value (0.0 to disable)
    #[config(default = 0.0)]
    pub grad_clip: f32,

    /// Weight decay penalty factor (0.0 to disable)
    #[config(default = "None")]
    pub weight_decay: Option<f32>,
}

/// TBPTT metrics for tracking training progress
#[derive(Clone, Default, Debug)]
pub struct TBPTTMetrics {
    /// Loss values for each batch
    batch_losses: Vec<f32>,

    /// Chunk losses within current batch
    chunk_losses: Vec<f32>,

    /// Current learning rate
    current_lr: f64,

    /// Progress within current epoch
    epoch_progress: f32,

    /// Metrics storage
    metrics: HashMap<String, Vec<MetricEntry>>,

    /// Tokens processed
    tokens_processed: usize,

    /// Tokens processed in current epoch
    epoch_tokens: usize,

    /// Total tokens processed across all epochs
    total_tokens: usize,

    /// Tokens processed since last timing update (for recent throughput)
    recent_tokens: usize,

    /// Training start time
    start_time: Option<Instant>,

    /// Epoch start time
    epoch_start_time: Option<Instant>,

    /// Last update time for throughput calculation
    last_update_time: Option<Instant>,
}

impl TBPTTMetrics {
    pub fn new() -> Self {
        let mut metrics = Self::default();
        metrics.start_time = Some(Instant::now());
        metrics.last_update_time = Some(Instant::now());
        metrics.recent_tokens = 0;
        metrics
    }

    pub fn update_batch_loss(&mut self, loss: f32) {
        self.batch_losses.push(loss);
        self.record_metric("batch_loss", loss);
    }

    pub fn batch_count(&self) -> usize {
        self.batch_losses.len()
    }

    pub fn update_chunk_loss(&mut self, loss: f32) {
        self.chunk_losses.push(loss);
        self.record_metric("chunk_loss", loss);
    }

    pub fn start_epoch(&mut self) {
        self.epoch_start_time = Some(Instant::now());
        self.epoch_tokens = 0;
    }

    pub fn add_tokens(&mut self, token_count: usize) {
        self.tokens_processed += token_count;
        self.epoch_tokens += token_count;
        self.total_tokens += token_count;
        self.recent_tokens += token_count;
    }

    pub fn tokens_per_second(&self) -> f64 {
        if let Some(start_time) = self.start_time {
            let elapsed = start_time.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                self.tokens_processed as f64 / elapsed
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    pub fn epoch_tokens_per_second(&self) -> f64 {
        if let Some(start_time) = self.epoch_start_time {
            let elapsed = start_time.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                self.epoch_tokens as f64 / elapsed
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    pub fn recent_tokens_per_second(&self) -> f64 {
        if let Some(last_time) = self.last_update_time {
            let elapsed = last_time.elapsed().as_secs_f64();
            if elapsed > 0.0 && self.recent_tokens > 0 {
                // Calculate throughput based on tokens added since last timing update
                self.recent_tokens as f64 / elapsed
            } else {
                // Fall back to overall rate if no recent data
                if self.tokens_processed > 0 {
                    let total_elapsed = self
                        .start_time
                        .map(|t| t.elapsed().as_secs_f64())
                        .unwrap_or(1.0);
                    self.tokens_processed as f64 / total_elapsed.max(0.001)
                } else {
                    0.0
                }
            }
        } else {
            0.0
        }
    }

    pub fn update_timing(&mut self) {
        // Record the tokens processed since last update, then reset counter
        self.last_update_time = Some(Instant::now());
        self.recent_tokens = 0;
    }

    pub fn epoch_tokens(&self) -> usize {
        self.epoch_tokens
    }

    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    pub fn record_metric(&mut self, name: &str, value: f32) {
        let entry = self
            .metrics
            .entry(name.to_string())
            .or_insert_with(Vec::new);
        entry.push(MetricEntry {
            name: name.to_string(),
            formatted: format!("{:.6}", value),
            serialize: value.to_string(),
        });
    }

    pub fn update_lr(&mut self, lr: f64) {
        self.current_lr = lr;
        self.record_metric("learning_rate", lr as f32);

        // Also record directly as a learning rate metric entry for easier tracking
        self.record_metric("lr", lr as f32);
    }

    pub fn update_progress(&mut self, progress: f32) {
        self.epoch_progress = progress;
    }

    pub fn avg_batch_loss(&self) -> f32 {
        if self.batch_losses.is_empty() {
            0.0
        } else {
            self.batch_losses.iter().sum::<f32>() / self.batch_losses.len() as f32
        }
    }

    pub fn avg_chunk_loss(&self) -> f32 {
        if self.chunk_losses.is_empty() {
            0.0
        } else {
            self.chunk_losses.iter().sum::<f32>() / self.chunk_losses.len() as f32
        }
    }

    pub fn clear_chunk_losses(&mut self) {
        self.chunk_losses.clear();
    }
}

// Custom metrics interface
pub trait CustomMetrics {
    fn get(&self, key: &str) -> Option<&Vec<MetricEntry>>;
    fn keys(&self) -> Vec<String>;
}

impl CustomMetrics for TBPTTMetrics {
    fn get(&self, key: &str) -> Option<&Vec<MetricEntry>> {
        self.metrics.get(key)
    }

    fn keys(&self) -> Vec<String> {
        self.metrics.keys().cloned().collect()
    }
}

/// Main TBPTT Trainer implementation
// No Module derive - we'll handle the model manually
pub struct TBPTTTrainer<B: AutodiffBackend> {
    model: MinGRULM<B>,
    hidden_states: HashMap<usize, Vec<Tensor<B, 2>>>,
    metrics: TBPTTMetrics,
    tbptt_k1: usize,
    tbptt_k2: usize,
    preserve_hidden_states: bool,
    chunk_size: usize,
    grad_clip: f32,
    learning_rate: f64,
}

impl<B: AutodiffBackend> std::fmt::Debug for TBPTTTrainer<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TBPTTTrainer")
            .field("model", &self.model)
            .field("hidden_states", &self.hidden_states)
            .field("metrics", &self.metrics)
            .field("tbptt_k1", &self.tbptt_k1)
            .field("tbptt_k2", &self.tbptt_k2)
            .field("preserve_hidden_states", &self.preserve_hidden_states)
            .field("chunk_size", &self.chunk_size)
            .field("grad_clip", &self.grad_clip)
            .field("learning_rate", &self.learning_rate)
            .finish_non_exhaustive()
    }
}

impl<B: AutodiffBackend> TBPTTTrainer<B> {
    // Helper method to save the model
    pub fn save_file<P: AsRef<Path> + Into<std::path::PathBuf>>(
        &self,
        path: P,
        recorder: &impl FileRecorder<B>,
    ) -> io::Result<()> {
        self.model
            .clone()
            .save_file(path, recorder)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))
    }

    pub fn new(model: MinGRULM<B>, config: &TBPTTConfig) -> Self {
        // Initialize the trainer

        Self {
            model,
            hidden_states: HashMap::new(),
            metrics: TBPTTMetrics::new(),
            tbptt_k1: config.tbptt_k1,
            tbptt_k2: config.tbptt_k2,
            preserve_hidden_states: config.preserve_hidden_states,
            chunk_size: config.chunk_size,
            grad_clip: config.grad_clip,
            learning_rate: config.learning_rate,
        }
    }

    /// Reset hidden states for all documents
    pub fn reset_hidden_states(&mut self) {
        self.hidden_states.clear();
    }

    /// Get metrics for external tracking
    pub fn metrics(&self) -> &TBPTTMetrics {
        &self.metrics
    }

    /// Process a single document chunk with TBPTT or random sampling
    pub fn process_chunk<O: Optimizer<MinGRULM<B>, B>>(
        &mut self,
        chunk: &ChunkedTextBatch<B>,
        optimizer: &mut O,
        accumulator: &mut GradientsAccumulator<MinGRULM<B>>,
        accumulation_current: &mut usize,
        do_update: bool,
        random_sampling: bool,
    ) -> f32 {
        // Safely get tensor dimensions with error handling
        if chunk.input.dims().len() < 2 {
            return 0.0; // Return zero loss for invalid tensors
        }
        
        let batch_size = chunk.input.dims()[0];
        let _seq_len = chunk.input.dims()[1];
        let device = chunk.input.device();

        // Get current hidden states for this document if available
        let mut current_hidden_states = Vec::new();

        // Skip hidden state handling if using random sampling mode
        if !random_sampling && self.preserve_hidden_states {
            // Process each position in the batch
            for i in 0..batch_size {
                let doc_id = if i < chunk.doc_ids.len() {
                    chunk.doc_ids[i]
                } else {
                    continue; // Skip if out of bounds
                };
                
                let chunk_idx = if i < chunk.chunk_indices.len() {
                    chunk.chunk_indices[i]
                } else {
                    continue; // Skip if out of bounds
                };
                
                let is_padded = chunk.is_padded.get(i).cloned().unwrap_or(false);

                // Retrieve or initialize hidden state for this position
                let doc_hidden = if !is_padded && self.hidden_states.contains_key(&doc_id) {
                    let state = self.hidden_states.get(&doc_id).unwrap().clone();

                    // Detach if we're beyond the backprop window (tbptt_k2)
                    if chunk_idx % self.tbptt_k2 == 0 {
                        // Detach to truncate backpropagation
                        state.iter().map(|h| h.clone().detach()).collect()
                    } else {
                        state
                    }
                } else {
                    // No previous state or padded chunk - initialize with zeros
                    Vec::new()
                };

                current_hidden_states.push(doc_hidden);
            }
        }

        // Determine whether to use hidden states or not
        let batch_hidden = if random_sampling || !self.preserve_hidden_states {
            // In random sampling mode or when hidden states are not preserved,
            // we always start with fresh hidden states (None)
            None
        } else if current_hidden_states.is_empty() || current_hidden_states[0].is_empty() {
            None
        } else {
            // Make sure we have valid hidden states with consistent dimensions
            let mut all_valid = true;
            let hidden_dim = match current_hidden_states
                .get(0)
                .and_then(|hs| hs.get(0).map(|h| h.dims()[0]))
            {
                Some(dim) => dim,
                None => {
                    all_valid = false;
                    96 // Default dimension from model config, adjust if needed
                }
            };

            if !all_valid {
                None
            } else {
                // Collect hidden states for each layer
                let mut merged_states = Vec::new();
                let num_layers = current_hidden_states[0].len();

                for layer in 0..num_layers {
                    let layer_states: Vec<_> = current_hidden_states
                        .iter()
                        .map(|doc_state| {
                            if doc_state.len() > layer {
                                // Ensure tensor has correct dimensions
                                doc_state[layer].clone()
                            } else {
                                // If layer doesn't exist, create zero tensor
                                Tensor::zeros([1, hidden_dim], &device)
                            }
                        })
                        .collect();

                    if layer_states.len() == batch_size {
                        // Make sure we're not stacking empty tensors
                        if !layer_states.is_empty() {
                            // Stack along batch dimension
                            merged_states.push(Tensor::cat(layer_states, 0));
                        } else {
                            // Skip this batch due to empty tensors
                            return 0.0;
                        }
                    } else {
                        // Skip this batch due to dimension mismatch
                        return 0.0;
                    }
                }

                Some(merged_states)
            }
        };

        // Forward pass with hidden state (or None for random sampling)
        let (logits, next_hidden) = self.model.forward(chunk.input.clone(), batch_hidden);

        // Only update hidden states if we're not in random sampling mode and preservation is enabled
        if !random_sampling && self.preserve_hidden_states {
            // Update hidden states for each document
            for i in 0..batch_size {
                let doc_id = chunk.doc_ids[i];
                let is_last_chunk = chunk.is_last_chunks[i];

                // Extract this document's hidden state from the batch with safe dimension handling
                let doc_next_hidden: Vec<Tensor<B, 2>> = next_hidden
                    .iter()
                    .map(|h| {
                        // Check dimensions and ensure safe slicing
                        let h_dims = h.dims();

                        if h_dims.len() != 2 {
                            // Create a tensor with the expected dimensions (must be 2D)
                            return Tensor::zeros([1, self.model.dim()], &device);
                        }

                        // Make sure we have valid dimensions for slicing
                        if i < h_dims[0] {
                            // Instead of squeezing, explicitly reshape to ensure correct 2D dimensions
                            let hidden_dim = h_dims[1];
                            let extracted = h.clone().slice([i..i + 1, 0..hidden_dim]);
                            // Already a 2D tensor with shape [1, hidden_dim]
                            extracted
                        } else {
                            // Create a properly sized tensor if index is out of bounds (must be 2D)
                            Tensor::zeros([1, h_dims[1]], &device)
                        }
                    })
                    .collect();

                // Apply tanh nonlinearity to hidden states before storing them (if feature enabled)
                #[cfg(feature = "tanh")]
                let doc_next_hidden_processed: Vec<Tensor<B, 2>> =
                    doc_next_hidden.iter().map(|h| h.clone().tanh()).collect();

                #[cfg(not(feature = "tanh"))]
                let doc_next_hidden_processed = doc_next_hidden;

                // Store unless this is the last chunk of a document
                if !is_last_chunk {
                    self.hidden_states.insert(doc_id, doc_next_hidden_processed);
                } else if is_last_chunk {
                    // Remove hidden state for completed documents
                    self.hidden_states.remove(&doc_id);
                }
            }
        }

        // Calculate loss
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let [target_batch_size, target_seq_len] = chunk.target.dims();

        // Variable to hold the calculated loss
        let loss;

        // Check for dimension mismatch
        if batch_size != target_batch_size || seq_len != target_seq_len {
            // println!(
            //     "Warning: Dimension mismatch. logits: [{}, {}, {}], target: [{}, {}]",
            //     batch_size, seq_len, vocab_size, target_batch_size, target_seq_len
            // );

            // Use the minimum sequence length
            let min_seq_len = seq_len.min(target_seq_len);

            // Slice the logits to the minimum sequence length
            let logits_sliced = if seq_len > min_seq_len {
                logits.clone().slice([0..batch_size, 0..min_seq_len, 0..vocab_size])
            } else {
                logits.clone()
            };

            // Slice the target to the minimum sequence length
            let target_sliced = if target_seq_len > min_seq_len {
                chunk
                    .target
                    .clone()
                    .slice([0..target_batch_size, 0..min_seq_len])
            } else {
                chunk.target.clone()
            };

            // Reshape for loss calculation
            let logits_reshaped = logits_sliced.reshape([batch_size * min_seq_len, vocab_size]);
            let targets_reshaped = target_sliced.clone().reshape([target_batch_size * min_seq_len]);

            // Calculate loss with the adjusted tensors
            let loss_fn = CrossEntropyLossConfig::new().init(&device);
            loss = loss_fn.forward(logits_reshaped.clone(), targets_reshaped.clone());
        } else {
            // Calculate loss normally
            let logits_reshaped = logits.reshape([batch_size * seq_len, vocab_size]);
            let targets_reshaped = chunk.target.clone().reshape([batch_size * seq_len]);

            let loss_fn = CrossEntropyLossConfig::new().init(&device);
            loss = loss_fn.forward(logits_reshaped.clone(), targets_reshaped.clone());
        }

        // Get loss value for metrics
        let scalar_value = loss.clone().into_scalar();
        let loss_value = scalar_value.to_f32();

        // Record metrics
        self.metrics.update_chunk_loss(loss_value);

        // Count tokens processed - batch_size * seq_len tokens per chunk
        let tokens_in_chunk = batch_size * seq_len;
        self.metrics.add_tokens(tokens_in_chunk);

        // Don't reset the counter until we're about to display the rate

        // Backward pass and accumulate gradients
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.model);

        // Gradient clipping is now handled by the optimizer configuration
        // Accumulate gradients directly
        accumulator.accumulate(&self.model, grads_params);
        *accumulation_current += 1;

        // Apply gradients if needed
        if do_update && *accumulation_current >= self.tbptt_k1 {
            let grads = accumulator.grads();

            // All optimizers need learning rate parameter
            let model = optimizer.step(self.learning_rate, self.model.clone(), grads);
            self.model = model;
                
            // Reset accumulation counter and create a fresh accumulator
            *accumulation_current = 0;
            *accumulator = GradientsAccumulator::<MinGRULM<B>>::new();

            // Update metrics with current learning rate (for both Adam and SGD)
            self.metrics.update_lr(self.learning_rate);
        }

        loss_value
    }

    /// Train the model for one epoch with continuous chunks or random sampling
    pub fn train_epoch<O: Optimizer<MinGRULM<B>, B>>(
        &mut self,
        dataloader: &mut ContinuousChunkedTextDataset,
        batcher: &TextBatcher<B>,
        optimizer: &mut O,
        epoch: usize,
        random_sampling: bool,
    ) -> f32 {
        println!("Training epoch {}", epoch);

        if random_sampling {
            println!("Using random sequence sampling mode");
        } else {
            println!("Using TBPTT with hidden state passing");
        }

        // Initialize for the epoch
        let (mut accumulator, progress_bar, total_steps) = self.prepare_for_epoch::<O>(dataloader);
        let mut accumulation_current = 0;
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        // Reset hidden states if using random sampling - we don't preserve state between sequences
        if random_sampling {
            self.reset_hidden_states();
        }

        // Process chunks using a definite counter to ensure we respect max_chunks_per_epoch
        let mut step = 0;
        while step < total_steps {
            let result = self.process_batch(
                dataloader, 
                batcher, 
                optimizer, 
                &mut accumulator, 
                &mut accumulation_current,
                step, 
                total_steps, 
                epoch, 
                &progress_bar,
                random_sampling,
            );
            
            // Handle batch result
            match result {
                BatchResult::Skip => {
                    step += 1;
                    continue;
                },
                BatchResult::Success { loss } => {
                    total_loss += loss;
                    batch_count += 1;
                    
                    // Update progress
                    self.update_progress_for_batch(step, total_steps, loss, &progress_bar);
                },
            }

            // If using random sampling, resample for the next batch
            if random_sampling {
                // Always reset and resample for random sampling
                dataloader.reset();
                dataloader.resample_positions(self.metrics.batch_count() as u64 + step as u64 + epoch as u64);
                
                // Clear hidden states between sequences when using random sampling
                self.reset_hidden_states();
            } else {
                // For TBPTT, proceed to next chunk or reset if needed
                if !dataloader.next_chunk() {
                    // If we can't advance, reset and resample to get fresh chunks
                    dataloader.reset();
                    dataloader.resample_positions(self.metrics.batch_count() as u64 + epoch as u64);
                }
            }

            step += 1;
        }

        // Apply any remaining gradients and finalize the epoch
        self.finalize_epoch(&mut accumulator, accumulation_current, total_loss, batch_count, optimizer, &progress_bar)
    }
    
    /// Prepare for training epoch by initializing metrics, progress bar, etc.
    fn prepare_for_epoch<O: Optimizer<MinGRULM<B>, B>>(
        &mut self,
        dataloader: &mut ContinuousChunkedTextDataset,
    ) -> (GradientsAccumulator<MinGRULM<B>>, ProgressBar, usize) {
        // Reset epoch metrics
        self.metrics.start_epoch();

        let total_steps = dataloader.max_chunks;

        // Progress bar for visualization
        let progress_bar = ProgressBar::new(total_steps as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) - {msg}")
                .expect("Progress bar template error")
                .progress_chars("#>-")
        );

        // Gradient accumulation state
        let accumulator = GradientsAccumulator::<MinGRULM<B>>::new();

        // Reset dataset and hidden states at the beginning of the epoch
        dataloader.reset();
        self.reset_hidden_states();
        
        (accumulator, progress_bar, total_steps)
    }
    
    /// Process a single batch during training
    fn process_batch<O: Optimizer<MinGRULM<B>, B>>(
        &mut self,
        dataloader: &mut ContinuousChunkedTextDataset,
        batcher: &TextBatcher<B>,
        optimizer: &mut O,
        accumulator: &mut GradientsAccumulator<MinGRULM<B>>,
        accumulation_current: &mut usize,
        step: usize,
        total_steps: usize,
        epoch: usize,
        progress_bar: &ProgressBar,
        random_sampling: bool,
    ) -> BatchResult {
        // Get current chunks for all positions
        let chunks = dataloader.get_current_chunks();

        // Create batch from chunks
        let chunked_batcher = ChunkedTextBatcher::new(batcher.vocab.clone(), batcher.device.clone());
        let batch_opt = chunked_batcher.batch(chunks);

        // Check if we got a valid batch
        if batch_opt.is_none() {
            progress_bar.inc(1);
            progress_bar.set_message("Skipped batch - empty");

            // We need to still move to next chunk and count even if we skip
            if !dataloader.next_chunk() {
                // If we can't advance, reset and resample to get fresh chunks
                dataloader.reset();
                dataloader.resample_positions(self.metrics.batch_count() as u64 + epoch as u64);
            }

            return BatchResult::Skip;
        }

        let batch = batch_opt.unwrap();

        // Check if tensors have valid dimensions and handle empty tensors more gracefully
        if batch.input.dims().len() < 2 || batch.input.dims()[0] == 0 || batch.input.dims()[1] == 0 {
            progress_bar.inc(1);
            progress_bar.set_message("Skipped batch - invalid dimensions");

            // We need to still move to next chunk and count even if we skip
            if !dataloader.next_chunk() {
                // If we can't advance, reset and resample to get fresh chunks
                dataloader.reset();
                dataloader.resample_positions(self.metrics.batch_count() as u64 + epoch as u64);
            }

            return BatchResult::Skip;
        }

        // Process chunk and update model if needed
        let do_update = (step + 1) % self.tbptt_k1 == 0 || step == total_steps - 1;
        let loss = self.process_chunk(
            &batch,
            optimizer,
            accumulator,
            accumulation_current,
            do_update,
            random_sampling,
        );

        BatchResult::Success { loss }
    }
    
    /// Update progress display after processing a batch
    fn update_progress_for_batch(
        &mut self, 
        step: usize, 
        total_steps: usize, 
        loss: f32, 
        progress_bar: &ProgressBar
    ) {
        // Update progress
        progress_bar.inc(1);

        // Calculate tokens/s for display and reset counter for next time
        let tokens_per_sec = self.metrics.recent_tokens_per_second();
        self.metrics.update_timing();

        progress_bar.set_message(format!(
            "Chunk {}/{}, Loss: {:.6}, Speed: {:.1} tok/s",
            step + 1,
            total_steps,
            loss,
            tokens_per_sec
        ));

        // Update metrics
        self.metrics.update_progress((step as f32) / (total_steps as f32));
    }
    
    /// Finalize epoch by applying remaining gradients and calculating metrics
    fn finalize_epoch<O: Optimizer<MinGRULM<B>, B>>(
        &mut self,
        accumulator: &mut GradientsAccumulator<MinGRULM<B>>,
        accumulation_current: usize,
        total_loss: f32,
        batch_count: usize,
        optimizer: &mut O,
        progress_bar: &ProgressBar,
    ) -> f32 {
        // Ensure we apply any remaining gradients
        if accumulation_current > 0 {
            let grads = accumulator.grads();
            self.model = optimizer.step(self.learning_rate, self.model.clone(), grads);
            
            // Create a fresh accumulator to ensure clean state for next epoch
            *accumulator = GradientsAccumulator::<MinGRULM<B>>::new();
        }

        // Finalize metrics
        let epoch_loss = if batch_count > 0 {
            total_loss / batch_count as f32
        } else {
            0.0 // Avoid division by zero
        };
        
        self.metrics.update_batch_loss(epoch_loss);

        // Calculate perplexity from loss
        let perplexity = (epoch_loss as f64).exp();
        self.metrics.record_metric("perplexity", perplexity as f32);

        // Calculate tokens per second for the entire epoch
        let epoch_tokens = self.metrics.epoch_tokens();
        let epoch_tok_per_sec = self.metrics.epoch_tokens_per_second();

        // Finish progress bar first, then print the summary on a new line
        progress_bar.finish();
        println!(
            "Epoch complete - Processed {} chunks ({} tokens, {:.1} tok/s), Avg loss: {:.6}, Perplexity: {:.2}",
            batch_count, epoch_tokens, epoch_tok_per_sec, epoch_loss, perplexity
        );

        epoch_loss
    }

    /// Validate the model with continuous chunks
    pub fn validate(
        &self,
        dataloader: &mut ContinuousChunkedTextDataset,
        batcher: &TextBatcher<B::InnerBackend>,
    ) -> f32 {
        println!("Validating model...");

        // Initialize for validation
        let (progress_bar, model, total_steps) = self.prepare_for_validation(dataloader);
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        // Track hidden states per position
        let mut hidden_states_map: HashMap<usize, Vec<Tensor<B::InnerBackend, 2>>> = HashMap::new();

        // Process chunks using a definite counter to ensure we respect max_chunks
        let mut step = 0;
        let validation_limit = 200; // Limit validation to 200 chunks for speed
        while step < total_steps && step < validation_limit {
            let result = self.process_validation_batch(
                dataloader,
                batcher,
                &model,
                step,
                &mut hidden_states_map,
                &progress_bar
            );
            
            match result {
                ValidationResult::Skip => {
                    step += 1;
                    continue;
                },
                ValidationResult::Success { loss } => {
                    total_loss += loss;
                    batch_count += 1;
                    
                    // Update progress
                    progress_bar.inc(1);
                    progress_bar.set_message(format!(
                        "Chunk {}/{}, Loss: {:.6}",
                        step + 1,
                        total_steps,
                        loss
                    ));
                },
            }

            // Move to next chunk and increment step counter
            if !dataloader.next_chunk() {
                // If we can't advance, reset and resample to get fresh chunks
                dataloader.reset();
                dataloader.resample_positions(step as u64 + 10000);
            }

            step += 1;
        }

        // Finalize validation and return average loss
        self.finalize_validation(total_loss, batch_count, &progress_bar)
    }
    
    /// Prepare for validation by creating progress bar and initializing the model
    fn prepare_for_validation(
        &self,
        dataloader: &mut ContinuousChunkedTextDataset,
    ) -> (ProgressBar, MinGRULM<B::InnerBackend>, usize) {
        let total_steps = dataloader.max_chunks;

        let progress_bar = ProgressBar::new(total_steps as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.yellow/blue}] {pos}/{len} ({eta}) - {msg}")
                .expect("Progress bar template error")
                .progress_chars("#>-")
        );

        // AutodiffModule trait provides valid() method
        let model = AutodiffModule::valid(&self.model);

        // Reset to beginning
        dataloader.reset();
        
        (progress_bar, model, total_steps)
    }
    
    /// Process a single batch during validation
    fn process_validation_batch(
        &self,
        dataloader: &mut ContinuousChunkedTextDataset,
        batcher: &TextBatcher<B::InnerBackend>,
        model: &MinGRULM<B::InnerBackend>,
        step: usize,
        hidden_states_map: &mut HashMap<usize, Vec<Tensor<B::InnerBackend, 2>>>,
        progress_bar: &ProgressBar,
    ) -> ValidationResult {
        // Get current chunks for validation
        let chunks = dataloader.get_current_chunks();
        let chunked_batcher = ChunkedTextBatcher::new(batcher.vocab.clone(), batcher.device.clone());
        let batch_opt = chunked_batcher.batch(chunks);

        // Check if we got a valid batch
        if batch_opt.is_none() {
            progress_bar.inc(1);
            progress_bar.set_message("Skipped batch - empty");

            // We need to still move to next chunk and count even if we skip
            if !dataloader.next_chunk() {
                // If we can't advance, reset and resample to get fresh chunks
                dataloader.reset();
                dataloader.resample_positions(step as u64 + 10000);
            }
            return ValidationResult::Skip;
        }

        let batch = batch_opt.unwrap();

        // Skip chunks that don't have valid dimensions with better tensor shape checking
        if batch.input.dims().len() < 2 || batch.input.dims()[0] == 0 || batch.input.dims()[1] == 0 {
            progress_bar.inc(1);
            progress_bar.set_message("Skipped batch - invalid dimensions");

            // We need to still move to next chunk and count even if we skip
            if !dataloader.next_chunk() {
                // If we can't advance, reset and resample to get fresh chunks
                dataloader.reset();
                dataloader.resample_positions(step as u64 + 10000);
            }
            return ValidationResult::Skip;
        }

        // Prepare batch hidden states
        let batch_hidden_states = self.prepare_validation_hidden_states(&batch, hidden_states_map);

        // Forward pass with hidden states
        let (logits, next_hidden) = model.forward(batch.input.clone(), batch_hidden_states);

        // Update hidden states for each position
        self.update_validation_hidden_states(&batch, next_hidden, hidden_states_map);

        // Calculate loss
        let loss = self.calculate_validation_loss(&batch, &logits);
        
        ValidationResult::Success { loss }
    }
    
    /// Prepare hidden states for validation forward pass
    fn prepare_validation_hidden_states(
        &self,
        batch: &ChunkedTextBatch<B::InnerBackend>,
        hidden_states_map: &HashMap<usize, Vec<Tensor<B::InnerBackend, 2>>>,
    ) -> Option<Vec<Tensor<B::InnerBackend, 2>>> {
        let batch_size = batch.doc_ids.len();
        let mut hidden_states_vec = Vec::new();

        if hidden_states_map.is_empty() {
            return None;
        }

        // Collect hidden states for each document in the batch
        for doc_id in &batch.doc_ids {
            if let Some(state) = hidden_states_map.get(doc_id) {
                hidden_states_vec.push(state.clone());
            }
        }

        // If we have hidden states for all documents in the batch, merge them
        if !hidden_states_vec.is_empty() && hidden_states_vec.len() == batch_size {
            // Transpose the structure to get layers of batch hidden states
            let mut merged_states = Vec::new();
            let num_layers = hidden_states_vec[0].len();

            for layer in 0..num_layers {
                let layer_states: Vec<_> = hidden_states_vec
                    .iter()
                    .map(|states| states[layer].clone())
                    .collect();

                merged_states.push(Tensor::cat(layer_states, 0));
            }

            Some(merged_states)
        } else {
            None
        }
    }
    
    /// Update hidden states map after validation forward pass
    fn update_validation_hidden_states(
        &self,
        batch: &ChunkedTextBatch<B::InnerBackend>,
        next_hidden: Vec<Tensor<B::InnerBackend, 2>>,
        hidden_states_map: &mut HashMap<usize, Vec<Tensor<B::InnerBackend, 2>>>,
    ) {
        for (i, &doc_id) in batch.doc_ids.iter().enumerate() {
            // Check if this is a padded position (end of text)
            let is_padded = batch.is_last_chunks.get(i).cloned().unwrap_or(false)
                || batch.is_padded.get(i).cloned().unwrap_or(false);

            if !is_padded {
                // Extract individual hidden state for this position
                let doc_next_hidden: Vec<Tensor<B::InnerBackend, 2>> = next_hidden
                    .iter()
                    .map(|h| {
                        // Safe slice with dimension check
                        if i < h.dims()[0] {
                            h.clone().slice([i..i + 1, 0..h.dims()[1]])
                        } else {
                            // Create a 2D tensor with proper dimensions
                            Tensor::zeros([1, h.dims()[1]], &h.device())
                        }
                    })
                    .collect();

                // Apply tanh nonlinearity to hidden states before storing (if feature enabled)
                #[cfg(feature = "tanh")]
                let doc_next_hidden_processed: Vec<
                    Tensor<B::InnerBackend, 2>,
                > = doc_next_hidden.iter().map(|h| h.clone().tanh()).collect();

                #[cfg(not(feature = "tanh"))]
                let doc_next_hidden_processed = doc_next_hidden;

                // Store for next chunk
                hidden_states_map.insert(doc_id, doc_next_hidden_processed);
            } else {
                // Remove hidden state for padded positions
                hidden_states_map.remove(&doc_id);
            }
        }
    }
    
    /// Calculate loss for a validation batch
    fn calculate_validation_loss(
        &self,
        batch: &ChunkedTextBatch<B::InnerBackend>,
        logits: &Tensor<B::InnerBackend, 3>,
    ) -> f32 {
        // Check for valid tensor dimensions
        if logits.dims().len() != 3 || batch.target.dims().len() != 2 {
            println!("Warning: Invalid tensor dimensions in validation.");
            return 0.0; // Return zero loss for invalid tensors
        }
        
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let [target_batch_size, target_seq_len] = batch.target.dims();

        // Check for dimension mismatch
        let (loss_reshaped, targets_reshaped) = if batch_size != target_batch_size || seq_len != target_seq_len {
            // println!("Warning: Dimension mismatch in validation. logits: [{}, {}, {}], target: [{}, {}]",
            //          batch_size, seq_len, vocab_size, target_batch_size, target_seq_len);

            // Get the minimum dimensions to avoid out-of-bounds issues
            let min_batch_size = batch_size.min(target_batch_size);
            let min_seq_len = seq_len.min(target_seq_len);
            
            if min_batch_size == 0 || min_seq_len == 0 {
                println!("Warning: Zero-sized tensor dimension detected, skipping loss calculation");
                return 0.0;
            }

            // Slice the logits to the minimum dimensions
            let logits_sliced = logits.clone().slice([0..min_batch_size, 0..min_seq_len, 0..vocab_size]);

            // Slice the target to the minimum dimensions
            let target_sliced = batch.target.clone().slice([0..min_batch_size, 0..min_seq_len]);

            // Reshape for loss calculation
            let logits_reshaped = logits_sliced.reshape([min_batch_size * min_seq_len, vocab_size]);
            let targets_reshaped = target_sliced.reshape([min_batch_size * min_seq_len]);

            (logits_reshaped, targets_reshaped)
        } else {
            // Reshape normally
            let logits_reshaped = logits.clone().reshape([batch_size * seq_len, vocab_size]);
            let targets_reshaped = batch.target.clone().reshape([batch_size * seq_len]);

            (logits_reshaped, targets_reshaped)
        };

        // Additional check for empty tensors
        if loss_reshaped.dims().iter().any(|&d| d == 0) || targets_reshaped.dims().iter().any(|&d| d == 0) {
            println!("Warning: Empty tensor detected during loss calculation");
            return 0.0;
        }

        let device = loss_reshaped.device();
        let loss_fn = CrossEntropyLossConfig::new().init(&device);
        let loss = loss_fn.forward(loss_reshaped, targets_reshaped);

        // Safely extract scalar value, returning 0 if anything goes wrong
        loss.into_scalar().to_f32()
    }
    
    /// Finalize validation and compute average loss
    fn finalize_validation(
        &self,
        total_loss: f32,
        batch_count: usize,
        progress_bar: &ProgressBar,
    ) -> f32 {
        // Handle the case where no batches were valid
        if batch_count == 0 {
            progress_bar.finish_with_message("No valid validation batches found");
            return 0.0;
        }

        let avg_loss = total_loss / batch_count as f32;

        // Calculate perplexity from validation loss
        let perplexity = (avg_loss as f64).exp();

        // Finish progress bar first, then print the summary on a new line
        progress_bar.finish();
        println!(
            "Validation complete - Processed {} chunks, Avg loss: {:.6}, Perplexity: {:.2}",
            batch_count, avg_loss, perplexity
        );

        avg_loss
    }
}

/// Train a language model using Truncated Backpropagation Through Time (TBPTT)
pub fn train_with_tbptt<B: AutodiffBackend>(
    config: &TBPTTConfig,
    device: &B::Device,
    input_data: &str,
    vocab: &CharVocab,
    artifact_dir: &str,
) -> MinGRULM<B> {
    // Set random seed for reproducibility (if supported by backend)
    match std::panic::catch_unwind(|| {
        <B as burn::tensor::backend::Backend>::seed(config.seed);
    }) {
        Ok(_) => println!("Random seed set to {}", config.seed),
        Err(_) => println!(
            "Warning: This backend doesn't support manual seed setting. Random results may vary."
        ),
    }

    // Initialize model
    let model = config
        .model
        .clone()
        .with_chunk_size(config.chunk_size)
        .init::<B>(device);

    // Initialize optimizer (feature flag is handled at the type level)
    #[cfg(feature = "optimizer-adam")]
    let _optimizer = {
        // Initialize Adam optimizer with customized hyperparameters
        // from the config, following the pattern in custom_training_loop.rs
        let mut adam_config = AdamConfig::new()
            .with_beta_1(config.adam_beta1)
            .with_beta_2(config.adam_beta2)
            .with_epsilon(config.adam_epsilon);

        // Apply weight decay if configured (using our stored copy from CLI args)
        if let Some(penalty) = config.weight_decay {
            adam_config = adam_config.with_weight_decay(Some(WeightDecayConfig::new(penalty)));
        }

        println!("Initializing Adam optimizer:");
        println!("  - beta_1: {}", config.adam_beta1);
        println!("  - beta_2: {}", config.adam_beta2);
        println!("  - epsilon: {}", config.adam_epsilon);
        println!(
            "  - weight_decay: {}",
            config
                .weight_decay
                .map_or("None".to_string(), |wd| format!("{}", wd))
        );

        // Initialize optimizer - learning rate will be applied during step() calls
        adam_config.init::<B, MinGRULM<B>>()
    };

    #[cfg(feature = "optimizer-sgd")]
    let mut optimizer = config.optimizer.init();

    // Create TBPTT trainer
    let mut trainer = TBPTTTrainer::new(model, config);

    // Calculate a reasonable number of chunks to process based on data size
    let input_length = input_data.len();
    let tokens_per_epoch = config.max_chunks_per_epoch * config.chunk_size;
    let total_processed_tokens = tokens_per_epoch * config.batch_size;
    let coverage_percentage = (total_processed_tokens as f64 / input_length as f64) * 100.0;

    println!("Input data size: {} characters", input_length);
    println!(
        "Will process {} tokens per epoch ({} chunks)",
        tokens_per_epoch, config.max_chunks_per_epoch
    );
    println!(
        "With batch size {}, processing ~{} total tokens per epoch ({:.2}% of dataset)",
        config.batch_size, total_processed_tokens, coverage_percentage
    );
    println!("Training configuration:");
    println!("- Batch size: {} parallel sequences", config.batch_size);
    println!("- Chunk size: {} characters per step", config.chunk_size);
    println!(
        "- Max chunks: {} steps per sequence",
        config.max_chunks_per_epoch
    );
    println!(
        "- Effective context length: {} characters",
        config.chunk_size * config.max_chunks_per_epoch
    );
    println!(
        "- Backpropagation window: {} characters",
        config.chunk_size * config.tbptt_k2
    );

    // Create continuous chunked dataset for training
    // This creates 'batch_size' different starting positions in the text
    // Each position will process up to 'max_chunks_per_epoch' consecutive chunks
    let batch_size = config.batch_size;
    let mut train_dataset = ContinuousChunkedTextDataset::new(
        input_data.to_string(),
        batch_size,
        config.chunk_size,
        config.max_chunks_per_epoch,
        config.seed,
    );
    println!(
        "Created training dataset with {} positions, chunk size {}, max chunks {}",
        batch_size, config.chunk_size, config.max_chunks_per_epoch
    );

    // Create validation dataset with different seed
    let mut valid_dataset = ContinuousChunkedTextDataset::new(
        input_data.to_string(),
        batch_size / 2,
        config.chunk_size,
        config.max_chunks_per_epoch / 5, // Use fewer chunks for validation
        config.seed + 1,
    );
    println!(
        "Created validation dataset with {} positions",
        batch_size / 2
    );

    // Create batchers
    let train_batcher = TextBatcher::<B>::new(vocab.clone(), device.clone());
    let valid_batcher = TextBatcher::<B::InnerBackend>::new(vocab.clone(), device.clone());

    // Create recorder for checkpoints
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    std::fs::create_dir_all(artifact_dir).expect("Failed to create artifact directory");

    println!("Starting TBPTT training");
    println!(
        "Parameters: k1={}, k2={}, chunk_size={}",
        config.tbptt_k1, config.tbptt_k2, config.chunk_size
    );

    // Training loop
    let mut best_loss = f32::MAX;
    let mut best_model = trainer.model.clone();

    // Decide on the maximum number of epochs
    let max_training_epochs = if config.target_valid_loss > 0.0 || config.target_test_loss > 0.0 {
        config.max_epochs
    } else {
        config.num_epochs
    };

    // Initialize learning rate scheduler
    let mut lr_scheduler = LearningRateScheduler::new(
        config.learning_rate,
        config.min_lr_factor,
        config.lr_scheduler,
        config.warmup_epochs,
        max_training_epochs,
        config.plateau_threshold,
        config.plateau_factor,
        config.stall_epochs,
        config.stall_threshold,
        config.plateau_epochs,
    );

    println!("Training for up to {} epochs", max_training_epochs);
    
    // Log training mode
    if config.random_sampling {
        println!("Using RANDOM SEQUENCE SAMPLING mode (no hidden state passing)");
        println!("- Each batch contains randomly sampled sequences");
        println!("- No hidden state is passed between batches");
    } else {
        println!("Using TBPTT mode with hidden state passing");
        println!("- Truncated Backpropagation Through Time with k1={}, k2={}", config.tbptt_k1, config.tbptt_k2);
        println!("- Hidden states preserved between chunks: {}", config.preserve_hidden_states);
    }
    if config.target_valid_loss > 0.0 {
        println!(
            "Will stop when validation loss reaches {:.6}",
            config.target_valid_loss
        );
    }
    if config.target_test_loss > 0.0 {
        println!(
            "Will stop when test loss reaches {:.6}",
            config.target_test_loss
        );
    }

    // Print learning rate schedule info
    println!("Learning rate configuration:");
    println!("- Base learning rate: {}", config.learning_rate);
    println!(
        "- Minimum learning rate: {}",
        config.learning_rate * config.min_lr_factor
    );
    println!("- Scheduler type: {:?}", config.lr_scheduler);

    if config.warmup_epochs > 0 {
        println!(
            "- Using {} warmup epochs with linear ramp",
            config.warmup_epochs
        );
    }

    if config.plateau_threshold > 0.0 {
        println!("- Reduce on plateau: enabled");
        println!(
            "  - Improvement threshold: {:.2}%",
            config.plateau_threshold * 100.0
        );
        println!("  - Reduction factor: {:.2}x", config.plateau_factor);
        println!(
            "  - Plateau detection: will reduce LR after {} consecutive epochs of <{}% improvement",
            config.plateau_epochs,
            config.plateau_threshold * 100.0
        );
        println!("  - Stall detection: will increase LR after {} epochs of <{}% improvement following a reduction",
                 config.stall_epochs, config.stall_threshold * 100.0);
    } else {
        println!("- Reduce on plateau: disabled");
    }

    #[cfg(feature = "optimizer-adam")]
    {
        println!("- Using Adam optimizer with adaptive parameter-specific learning rates");
        println!("- Adam applies base learning rate to all parameters and adapts based on gradient history");
    }

    #[cfg(feature = "optimizer-sgd")]
    {
        println!("- Using SGD optimizer with SGD configuration");
    }

    // Create optimizer - Adam or SGD based on enabled feature
    #[cfg(feature = "optimizer-adam")]
    let _optimizer = config.optimizer.init::<B, MinGRULM<B>>();

    #[cfg(feature = "optimizer-sgd")]
    let _optimizer = config.optimizer.init();

    for epoch in 1..=max_training_epochs {
        // Get learning rate for this epoch from the scheduler
        let current_lr = lr_scheduler.get_lr_for_epoch(epoch);

        // Update trainer's learning rate
        trainer.learning_rate = current_lr;

        // Print learning rate information
        #[cfg(feature = "optimizer-sgd")]
        println!(
            "Epoch {}/{} - Learning rate: {:.6e}",
            epoch, max_training_epochs, current_lr
        );

        #[cfg(feature = "optimizer-adam")]
        println!(
            "Epoch {}/{} - Adam base learning rate: {:.6e}",
            epoch, max_training_epochs, current_lr
        );

        // Resample starting positions for each epoch with different seeds
        println!("Resampling positions for epoch {}...", epoch);
        train_dataset.resample_positions(config.seed + epoch as u64 * 1000);
        valid_dataset.resample_positions(config.seed + epoch as u64 * 1000 + 500);

        // Create optimizer - Adam or SGD based on enabled feature
        #[cfg(feature = "optimizer-adam")]
        let mut adam_optimizer = config.optimizer.init::<B, MinGRULM<B>>();
        
        #[cfg(feature = "optimizer-sgd")]
        let mut sgd_optimizer = config.optimizer.init();
        
        // Training phase - handle different optimizer types directly
        let train_loss = {
            #[cfg(feature = "optimizer-adam")]
            {
                trainer.train_epoch(&mut train_dataset, &train_batcher, &mut adam_optimizer, epoch, config.random_sampling)
            }
            #[cfg(feature = "optimizer-sgd")]
            {
                let loss = trainer.train_epoch(&mut train_dataset, &train_batcher, &mut sgd_optimizer, epoch, config.random_sampling);
                loss
            }
            #[cfg(not(any(feature = "optimizer-adam", feature = "optimizer-sgd")))]
            {
                let mut default_optimizer = AdamConfig::new().init::<B, MinGRULM<B>>();
                trainer.train_epoch(&mut train_dataset, &train_batcher, &mut default_optimizer, epoch, config.random_sampling)
            }
        };

        // Validation phase (only if we have validation data)
        let valid_loss = if valid_dataset.len() > 0 {
            trainer.validate(&mut valid_dataset, &valid_batcher)
        } else {
            println!("Warning: No validation data available, skipping validation");
            // Use training loss as a fallback
            train_loss
        };

        // Calculate perplexity values
        let train_ppl = (train_loss as f64).exp();
        let valid_ppl = (valid_loss as f64).exp();

        println!(
            "Epoch {}/{} - Train Loss: {:.6} (PPL: {:.2}), Valid Loss: {:.6} (PPL: {:.2})",
            epoch, max_training_epochs, train_loss, train_ppl, valid_loss, valid_ppl
        );

        // Check if we should reduce learning rate based on validation loss
        // This is separate from the normal learning rate schedule
        if lr_scheduler.check_reduce_on_plateau(valid_loss) {
            // Update trainer's learning rate
            trainer.learning_rate = lr_scheduler.get_current_lr();
        }

        // Save best model
        if valid_loss < best_loss {
            best_loss = valid_loss;
            best_model = trainer.model.clone();

            // Save checkpoint
            let model_path = format!("{}/model_best.bin", artifact_dir);
            best_model
                .clone()
                .save_file(&model_path, &recorder)
                .expect("Failed to save best model");
            println!("Saved new best model with loss {:.6}", best_loss);
        }

        // Save checkpoint
        if epoch % 1 == 0 {
            let model_path = format!("{}/model_epoch_{}.bin", artifact_dir, epoch);
            trainer
                .model
                .clone()
                .save_file(&model_path, &recorder)
                .expect("Failed to save checkpoint");
        }

        // Check if we've reached the target loss
        let target_reached = (config.target_valid_loss > 0.0
            && valid_loss <= config.target_valid_loss)
            || (config.target_test_loss > 0.0 && train_loss <= config.target_test_loss);

        if target_reached {
            println!(
                "🎉 Target loss reached at epoch {}! Stopping training.",
                epoch
            );
            // Save final model at target
            let model_path = format!("{}/model_target_reached.bin", artifact_dir);
            trainer
                .model
                .clone()
                .save_file(&model_path, &recorder)
                .expect("Failed to save target model");
            println!("Target model saved to {}", model_path);
            break;
        }

        // Reset hidden states between epochs
        trainer.reset_hidden_states();
    }

    // Save final model
    let model_path = format!("{}/model_final.bin", artifact_dir);
    trainer
        .model
        .clone()
        .save_file(&model_path, &recorder)
        .expect("Failed to save final model");

    // Calculate total training statistics
    let total_tokens = trainer.metrics().total_tokens();
    let total_time = trainer
        .metrics()
        .start_time
        .map(|t| t.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    let avg_throughput = if total_time > 0.0 {
        total_tokens as f64 / total_time
    } else {
        0.0
    };

    println!("Final model saved to {}", model_path);
    println!("Best validation loss: {:.6}", best_loss);
    println!(
        "Training processed {} tokens in {:.1} seconds ({:.1} tok/s average, including validation)",
        total_tokens, total_time, avg_throughput
    );

    best_model
}
