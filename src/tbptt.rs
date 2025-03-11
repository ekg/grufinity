use burn::{
    config::Config,
    module::{Module, AutodiffModule},
    nn::loss::CrossEntropyLossConfig,
    optim::{SgdConfig, GradientsAccumulator, GradientsParams, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{backend::AutodiffBackend, Tensor, cast::ToElement},
    train::{
        metric::MetricEntry,
    },
};
use std::io;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::Path;
use indicatif::{ProgressBar, ProgressStyle};
use burn::data::dataloader::batcher::Batcher;

use crate::dataset::{CharVocab, TextBatcher, ChunkedTextBatch, ChunkedTextBatcher, ContinuousChunkedTextDataset};
use burn::data::dataset::Dataset;
use crate::model::{MinGRULM, MinGRULMConfig};
use burn::record::FileRecorder;

/// Configuration for TBPTT training
#[derive(Config)]
pub struct TBPTTConfig {
    /// Model configuration
    pub model: MinGRULMConfig,
    
    /// Optimizer configuration
    pub optimizer: SgdConfig,
    
    /// Learning rate
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
    
    /// Learning rate scheduler type ("none", "cosine", "linear")
    #[config(default = "none")]
    pub lr_scheduler: String,
    
    /// Minimum learning rate for scheduler (as fraction of base lr)
    #[config(default = 0.1)]
    pub min_lr_factor: f64,
    
    /// Number of warmup epochs
    #[config(default = 0)]
    pub warmup_epochs: usize,
    
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
}

impl TBPTTMetrics {
    pub fn new() -> Self {
        Self::default()
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
    
    pub fn record_metric(&mut self, name: &str, value: f32) {
        let entry = self.metrics.entry(name.to_string())
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
    pub fn save_file<P: AsRef<Path> + Into<std::path::PathBuf>>(&self, path: P, recorder: &impl FileRecorder<B>) -> io::Result<()> {
        self.model.clone().save_file(path, recorder)
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
    
    /// Process a single document chunk with TBPTT handling padded chunks
    pub fn process_chunk<O: Optimizer<MinGRULM<B>, B>>(
        &mut self,
        chunk: &ChunkedTextBatch<B>,
        optimizer: &mut O,
        accumulator: &mut GradientsAccumulator<MinGRULM<B>>,
        accumulation_current: &mut usize,
        do_update: bool
    ) -> f32 {
        let batch_size = chunk.input.dims()[0];
        let _seq_len = chunk.input.dims()[1];
        let device = chunk.input.device();
        
        // Get current hidden states for this document if available
        let mut current_hidden_states = Vec::new();
        
        // Process each position in the batch
        for i in 0..batch_size {
            let doc_id = chunk.doc_ids[i];
            let chunk_idx = chunk.chunk_indices[i];
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
        
        // Flatten into vector of tensors for the batch
        let batch_hidden = if current_hidden_states.is_empty() || current_hidden_states[0].is_empty() {
            None
        } else {
            // Make sure we have valid hidden states with consistent dimensions
            let mut all_valid = true;
            let hidden_dim = match current_hidden_states.get(0).and_then(|hs| hs.get(0).map(|h| h.dims()[0])) {
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
                    let layer_states: Vec<_> = current_hidden_states.iter()
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
                        // Stack along batch dimension
                        merged_states.push(Tensor::cat(layer_states, 0));
                    } else {
                        // Skip this batch due to dimension mismatch
                        return 0.0;
                    }
                }
                
                Some(merged_states)
            }
        };
        
        // Forward pass with hidden state
        let (logits, next_hidden) = self.model.forward(chunk.input.clone(), batch_hidden);
        
        // Update hidden states for each document
        for i in 0..batch_size {
            let doc_id = chunk.doc_ids[i];
            let is_last_chunk = chunk.is_last_chunks[i];
            

            // Extract this document's hidden state from the batch with safe dimension handling
            let doc_next_hidden: Vec<Tensor<B, 2>> = next_hidden.iter()
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
                        let extracted = h.clone().slice([i..i+1, 0..hidden_dim]);
                        // Already a 2D tensor with shape [1, hidden_dim]
                        extracted
                    } else {
                        // Create a properly sized tensor if index is out of bounds (must be 2D)
                        Tensor::zeros([1, h_dims[1]], &device)
                    }
                })
                .collect();
            
            // Store unless this is the last chunk of a document
            if !is_last_chunk && self.preserve_hidden_states {
                self.hidden_states.insert(doc_id, doc_next_hidden);
            } else if is_last_chunk {
                // Remove hidden state for completed documents
                self.hidden_states.remove(&doc_id);
            }
        }
        
        // Calculate loss
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let [target_batch_size, target_seq_len] = chunk.target.dims();
        
        // Variable to hold the calculated loss
        let loss;
        
        // Check for dimension mismatch
        if batch_size != target_batch_size || seq_len != target_seq_len {
            println!("Warning: Dimension mismatch. logits: [{}, {}, {}], target: [{}, {}]",
                     batch_size, seq_len, vocab_size, target_batch_size, target_seq_len);
            
            // Use the minimum sequence length
            let min_seq_len = seq_len.min(target_seq_len);
            
            // Slice the logits to the minimum sequence length
            let logits_sliced = if seq_len > min_seq_len {
                logits.slice([0..batch_size, 0..min_seq_len, 0..vocab_size])
            } else {
                logits
            };
            
            // Slice the target to the minimum sequence length
            let target_sliced = if target_seq_len > min_seq_len {
                chunk.target.clone().slice([0..target_batch_size, 0..min_seq_len])
            } else {
                chunk.target.clone()
            };
            
            // Reshape for loss calculation
            let logits_reshaped = logits_sliced.reshape([batch_size * min_seq_len, vocab_size]);
            let targets_reshaped = target_sliced.reshape([target_batch_size * min_seq_len]);
            
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
        
        // Backward pass and accumulate gradients
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.model);
        
        // Apply gradient clipping if configured
        let effective_grads = if self.grad_clip > 0.0 {
            // Implement gradient clipping here
            // For simplicity, just using the original grads for now
            grads_params
        } else {
            grads_params
        };
        
        // Accumulate gradients
        accumulator.accumulate(&self.model, effective_grads);
        *accumulation_current += 1;
        
        // Apply gradients if needed
        if do_update && *accumulation_current >= self.tbptt_k1 {
            let grads = accumulator.grads();
            self.model = optimizer.step(self.learning_rate, self.model.clone(), grads);
            *accumulation_current = 0;
        }
        
        loss_value
    }
    
    /// Train the model for one epoch with continuous chunks
    pub fn train_epoch<O: Optimizer<MinGRULM<B>, B>>(
        &mut self,
        dataloader: &mut ContinuousChunkedTextDataset,
        batcher: &TextBatcher<B>,
        optimizer: &mut O,
        epoch: usize,
    ) -> f32 {
        println!("Training epoch {}", epoch);
        
        let total_steps = dataloader.max_chunks;
        
        // Progress bar for visualization
        let progress_bar = ProgressBar::new(total_steps as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) - {msg}")
                .expect("Progress bar template error")
                .progress_chars("#>-")
        );
        
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        
        // Gradient accumulation state
        let mut accumulator = GradientsAccumulator::<MinGRULM<B>>::new();
        let mut accumulation_current = 0;
        
        // Reset dataset and hidden states at the beginning of the epoch
        dataloader.reset();
        self.reset_hidden_states();
        
        // Process chunks using a definite counter to ensure we respect max_chunks_per_epoch
        let mut step = 0;
        while step < total_steps {
            // Get current chunks for all positions
            let chunks = dataloader.get_current_chunks();
            
            // Create batch from chunks
            let chunked_batcher = ChunkedTextBatcher::new(batcher.vocab.clone(), batcher.device.clone());
            let batch = chunked_batcher.batch(chunks);
            
            // Ensure we have valid input data
            if batch.input.dims()[0] == 0 || batch.input.dims()[1] == 0 {
                progress_bar.inc(1);
                progress_bar.set_message("Skipped batch");
                
                // We need to still move to next chunk and count even if we skip
                if !dataloader.next_chunk() {
                    // If we can't advance, reset and resample to get fresh chunks
                    dataloader.reset();
                    dataloader.resample_positions(self.metrics.batch_count() as u64 + epoch as u64);
                }
                
                step += 1;
                continue;
            }
            
            // Process chunk and update model if needed
            let do_update = (step + 1) % self.tbptt_k1 == 0 || step == total_steps - 1;
            let loss = self.process_chunk(&batch, optimizer, &mut accumulator, &mut accumulation_current, do_update);
            
            total_loss += loss;
            batch_count += 1;
            
            // Update progress
            progress_bar.inc(1);
            progress_bar.set_message(format!("Chunk {}/{}, Loss: {:.6}", step + 1, total_steps, loss));
            
            // Update metrics
            self.metrics.update_progress((step as f32) / (total_steps as f32));
            
            // Move to next chunk and increment step counter
            if !dataloader.next_chunk() {
                // If we can't advance, reset and resample to get fresh chunks
                dataloader.reset();
                dataloader.resample_positions(self.metrics.batch_count() as u64 + epoch as u64);
            }
            
            step += 1;
        }
        
        // Ensure we apply any remaining gradients
        if accumulation_current > 0 {
            let grads = accumulator.grads();
            self.model = optimizer.step(self.learning_rate, self.model.clone(), grads);
        }
        
        // Finalize metrics
        let epoch_loss = total_loss / batch_count as f32;
        self.metrics.update_batch_loss(epoch_loss);
        
        progress_bar.finish_with_message(format!("Epoch complete - Processed {} chunks, Avg loss: {:.6}", 
                                               batch_count, epoch_loss));
        
        epoch_loss
    }
    
    /// Validate the model with continuous chunks
    pub fn validate(
        &self,
        dataloader: &mut ContinuousChunkedTextDataset,
        batcher: &TextBatcher<B::InnerBackend>,
    ) -> f32 {
        println!("Validating model...");
    
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
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        
        // Reset to beginning
        dataloader.reset();
        
        // Track hidden states per position
        let mut hidden_states_map: HashMap<usize, Vec<Tensor<B::InnerBackend, 2>>> = HashMap::new();
    
        // Process chunks using a definite counter to ensure we respect max_chunks
        let mut step = 0;
        while step < total_steps && step < 200 { // Add a validation limit of 200 chunks for speed
            let chunks = dataloader.get_current_chunks();
            let chunked_batcher = ChunkedTextBatcher::new(batcher.vocab.clone(), batcher.device.clone());
            let batch = chunked_batcher.batch(chunks);
            
            // Skip chunks that don't have valid dimensions
            if batch.input.dims()[0] == 0 || batch.input.dims()[1] == 0 {
                progress_bar.inc(1);
                // We need to still move to next chunk and count even if we skip
                if !dataloader.next_chunk() {
                    // If we can't advance, reset and resample to get fresh chunks
                    dataloader.reset();
                    dataloader.resample_positions(step as u64 + 10000);
                }
                step += 1;
                continue;
            }
            
            // Prepare batch hidden states
            let batch_size = batch.doc_ids.len();
            let mut batch_hidden_states = None;
            let mut hidden_states_vec = Vec::new();
            
            if !hidden_states_map.is_empty() {
                for doc_id in &batch.doc_ids {
                    if let Some(state) = hidden_states_map.get(doc_id) {
                        hidden_states_vec.push(state.clone());
                    }
                }
                
                if !hidden_states_vec.is_empty() && hidden_states_vec.len() == batch_size {
                    // Transpose the structure to get layers of batch hidden states
                    let mut merged_states = Vec::new();
                    let num_layers = hidden_states_vec[0].len();
                    
                    for layer in 0..num_layers {
                        let layer_states: Vec<_> = hidden_states_vec.iter()
                            .map(|states| states[layer].clone())
                            .collect();
                        
                        merged_states.push(Tensor::cat(layer_states, 0));
                    }
                    
                    batch_hidden_states = Some(merged_states);
                }
            }
            
            // Forward pass with hidden states
            let (logits, next_hidden) = model.forward(batch.input.clone(), batch_hidden_states);
            
            // Update hidden states for each position
            for (i, &doc_id) in batch.doc_ids.iter().enumerate() {
                // Check if this is a padded position (end of text)
                let is_padded = batch.is_last_chunks.get(i).cloned().unwrap_or(false) || 
                                batch.is_padded.get(i).cloned().unwrap_or(false);
                
                if !is_padded {
                    // Extract individual hidden state for this position
                    let doc_next_hidden: Vec<Tensor<B::InnerBackend, 2>> = next_hidden.iter()
                        .map(|h| {
                            // Safe slice with dimension check
                            if i < h.dims()[0] {
                                h.clone().slice([i..i+1, 0..h.dims()[1]])
                            } else {
                                // Create a 2D tensor with proper dimensions
                                Tensor::zeros([1, h.dims()[1]], &h.device())
                            }
                        })
                        .collect();
                    
                    // Store for next chunk
                    hidden_states_map.insert(doc_id, doc_next_hidden);
                } else {
                    // Remove hidden state for padded positions
                    hidden_states_map.remove(&doc_id);
                }
            }
            
            // Calculate loss
            let [batch_size, seq_len, vocab_size] = logits.dims();
            let [target_batch_size, target_seq_len] = batch.target.dims();
            
            // Check for dimension mismatch
            let (loss_reshaped, targets_reshaped) = if batch_size != target_batch_size || seq_len != target_seq_len {
                println!("Warning: Dimension mismatch in validation. logits: [{}, {}, {}], target: [{}, {}]",
                         batch_size, seq_len, vocab_size, target_batch_size, target_seq_len);
                
                // Use the minimum sequence length
                let min_seq_len = seq_len.min(target_seq_len);
                
                // Slice the logits to the minimum sequence length
                let logits_sliced = if seq_len > min_seq_len {
                    logits.slice([0..batch_size, 0..min_seq_len, 0..vocab_size])
                } else {
                    logits
                };
                
                // Slice the target to the minimum sequence length
                let target_sliced = if target_seq_len > min_seq_len {
                    batch.target.clone().slice([0..target_batch_size, 0..min_seq_len])
                } else {
                    batch.target.clone()
                };
                
                // Reshape for loss calculation
                let logits_reshaped = logits_sliced.reshape([batch_size * min_seq_len, vocab_size]);
                let targets_reshaped = target_sliced.reshape([target_batch_size * min_seq_len]);
                
                (logits_reshaped, targets_reshaped)
            } else {
                // Reshape normally
                let logits_reshaped = logits.reshape([batch_size * seq_len, vocab_size]);
                let targets_reshaped = batch.target.reshape([batch_size * seq_len]);
                
                (logits_reshaped, targets_reshaped)
            };
            
            let device = loss_reshaped.device();
            let loss_fn = CrossEntropyLossConfig::new().init(&device);
            let loss = loss_fn.forward(loss_reshaped, targets_reshaped);
            
            let loss_value = loss.into_scalar().to_f32();
            total_loss += loss_value;
            batch_count += 1;
            
            // Update progress
            progress_bar.inc(1);
            progress_bar.set_message(format!("Chunk {}/{}, Loss: {:.6}", step + 1, total_steps, loss_value));
            
            // Move to next chunk and increment step counter
            if !dataloader.next_chunk() {
                // If we can't advance, reset and resample to get fresh chunks
                dataloader.reset();
                dataloader.resample_positions(step as u64 + 10000);
            }
            
            step += 1;
        }
        
        // Handle the case where no batches were valid
        if batch_count == 0 {
            progress_bar.finish_with_message("No valid validation batches found");
            return 0.0;
        }
    
        let avg_loss = total_loss / batch_count as f32;
        progress_bar.finish_with_message(format!("Validation complete - Processed {} chunks, Avg loss: {:.6}", 
                                               batch_count, avg_loss));
    
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
    // Set random seed for reproducibility
    <B as burn::tensor::backend::Backend>::seed(config.seed);
    
    // Initialize model
    let model = config.model.clone()
        .with_chunk_size(config.chunk_size)
        .init::<B>(device);
    
    // Initialize optimizer
    let mut optimizer = config.optimizer.init();
    
    // Create TBPTT trainer
    let mut trainer = TBPTTTrainer::new(model, config);
    
    // Calculate a reasonable number of chunks to process based on data size
    let input_length = input_data.len();
    let tokens_per_epoch = config.max_chunks_per_epoch * config.chunk_size * config.batch_size;
    let coverage_percentage = (tokens_per_epoch as f64 / input_length as f64) * 100.0;
    
    println!("Input data size: {} characters", input_length);
    println!("Will process ~{} tokens per epoch ({:.2}% of dataset)", 
             tokens_per_epoch, coverage_percentage);
    println!("Training configuration:");
    println!("- Batch size: {} parallel sequences", config.batch_size);
    println!("- Chunk size: {} characters per step", config.chunk_size);
    println!("- Max chunks: {} steps per sequence", config.max_chunks_per_epoch);
    println!("- Effective context length: {} characters", config.chunk_size * config.max_chunks_per_epoch);
    println!("- Backpropagation window: {} characters", config.chunk_size * config.tbptt_k2);
    
    // Create continuous chunked dataset for training
    // This creates 'batch_size' different starting positions in the text
    // Each position will process up to 'max_chunks_per_epoch' consecutive chunks
    let batch_size = config.batch_size;
    let mut train_dataset = ContinuousChunkedTextDataset::new(
        input_data.to_string(),
        batch_size,
        config.chunk_size,
        config.max_chunks_per_epoch,
        config.seed
    );
    println!("Created training dataset with {} positions, chunk size {}, max chunks {}", 
             batch_size, config.chunk_size, config.max_chunks_per_epoch);
    
    // Create validation dataset with different seed
    let mut valid_dataset = ContinuousChunkedTextDataset::new(
        input_data.to_string(),
        batch_size / 2,
        config.chunk_size,
        config.max_chunks_per_epoch / 5, // Use fewer chunks for validation
        config.seed + 1
    );
    println!("Created validation dataset with {} positions", batch_size / 2);
    
    // Create batchers
    let train_batcher = TextBatcher::<B>::new(vocab.clone(), device.clone());
    let valid_batcher = TextBatcher::<B::InnerBackend>::new(vocab.clone(), device.clone());
    
    // Create recorder for checkpoints
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    std::fs::create_dir_all(artifact_dir).expect("Failed to create artifact directory");
    
    println!("Starting TBPTT training");
    println!("Parameters: k1={}, k2={}, chunk_size={}", config.tbptt_k1, config.tbptt_k2, config.chunk_size);
    
    // Training loop
    let mut best_loss = f32::MAX;
    let mut best_model = trainer.model.clone();
    
    // Decide on the maximum number of epochs
    let max_training_epochs = if config.target_valid_loss > 0.0 || config.target_test_loss > 0.0 {
        config.max_epochs
    } else {
        config.num_epochs
    };
    
    // Setup learning rate scheduling
    let use_cosine = config.lr_scheduler.to_lowercase() == "cosine";
    let use_linear = config.lr_scheduler.to_lowercase() == "linear";
    let warmup_epochs = config.warmup_epochs;
    let min_lr = config.learning_rate * config.min_lr_factor;
    
    println!("Training for up to {} epochs", max_training_epochs);
    if config.target_valid_loss > 0.0 {
        println!("Will stop when validation loss reaches {:.6}", config.target_valid_loss);
    }
    if config.target_test_loss > 0.0 {
        println!("Will stop when test loss reaches {:.6}", config.target_test_loss);
    }
    
    // Print learning rate schedule info
    if warmup_epochs > 0 {
        println!("Using {} warmup epochs with linear ramp", warmup_epochs);
    }
    if use_cosine {
        println!("Using cosine annealing schedule from lr={} to min_lr={}", 
                config.learning_rate, min_lr);
    } else if use_linear {
        println!("Using linear decay schedule from lr={} to min_lr={}", 
                config.learning_rate, min_lr);
    } else {
        println!("Using constant learning rate: {}", config.learning_rate);
    }
    
    for epoch in 1..=max_training_epochs {
        // Calculate learning rate for this epoch
        let current_lr = if epoch <= warmup_epochs {
            // Linear warmup
            config.learning_rate * (epoch as f64 / warmup_epochs.max(1) as f64)
        } else if use_cosine {
            // Cosine annealing
            let progress = (epoch - warmup_epochs) as f64 / 
                          (max_training_epochs - warmup_epochs).max(1) as f64;
            let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            min_lr + (config.learning_rate - min_lr) * cosine_decay
        } else if use_linear {
            // Linear decay
            let progress = (epoch - warmup_epochs) as f64 / 
                          (max_training_epochs - warmup_epochs).max(1) as f64;
            config.learning_rate - (config.learning_rate - min_lr) * progress
        } else {
            // Constant learning rate
            config.learning_rate
        };
        
        // Update trainer's learning rate
        trainer.learning_rate = current_lr;
        trainer.metrics.update_lr(current_lr);
        
        println!("Epoch {}/{} - Learning rate: {:.6e}", epoch, max_training_epochs, current_lr);
        
        // Resample starting positions for each epoch with different seeds
        println!("Resampling positions for epoch {}...", epoch);
        train_dataset.resample_positions(config.seed + epoch as u64 * 1000);
        valid_dataset.resample_positions(config.seed + epoch as u64 * 1000 + 500);
        
        // Training phase
        let train_loss = trainer.train_epoch(&mut train_dataset, &train_batcher, &mut optimizer, epoch);
        
        // Validation phase (only if we have validation data)
        let valid_loss = if valid_dataset.len() > 0 {
            trainer.validate(&mut valid_dataset, &valid_batcher)
        } else {
            println!("Warning: No validation data available, skipping validation");
            // Use training loss as a fallback
            train_loss
        };
        
        println!("Epoch {}/{} - Train Loss: {:.6}, Valid Loss: {:.6}", 
            epoch, max_training_epochs, train_loss, valid_loss);
        
        // Save best model
        if valid_loss < best_loss {
            best_loss = valid_loss;
            best_model = trainer.model.clone();
            
            // Save checkpoint
            let model_path = format!("{}/model_best.bin", artifact_dir);
            best_model.clone()
                .save_file(&model_path, &recorder)
                .expect("Failed to save best model");
            println!("Saved new best model with loss {:.6}", best_loss);
        }
        
        // Save checkpoint
        if epoch % 1 == 0 {
            let model_path = format!("{}/model_epoch_{}.bin", artifact_dir, epoch);
            trainer.model.clone()
                .save_file(&model_path, &recorder)
                .expect("Failed to save checkpoint");
        }
        
        // Check if we've reached the target loss
        let target_reached = 
            (config.target_valid_loss > 0.0 && valid_loss <= config.target_valid_loss) ||
            (config.target_test_loss > 0.0 && train_loss <= config.target_test_loss);
        
        if target_reached {
            println!("🎉 Target loss reached at epoch {}! Stopping training.", epoch);
            // Save final model at target
            let model_path = format!("{}/model_target_reached.bin", artifact_dir);
            trainer.model.clone()
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
    trainer.model.clone()
        .save_file(&model_path, &recorder)
        .expect("Failed to save final model");
    
    println!("Final model saved to {}", model_path);
    println!("Best validation loss: {:.6}", best_loss);
    
    best_model
}

