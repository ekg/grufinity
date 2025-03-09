use burn::{
    config::Config,
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsAccumulator, GradientsParams, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{backend::AutodiffBackend, Tensor, cast::ToElement},
    train::{
        ClassificationOutput, TrainOutput,
        metric::MetricEntry,
    },
};
use std::collections::HashMap;
use std::fmt::Debug;
use indicatif::{ProgressBar, ProgressStyle};
use burn::data::dataset::Dataset;
use burn::data::dataloader::batcher::Batcher;

use crate::dataset::{CharVocab, TextBatcher, TextDataset, ChunkedTextDataset, ChunkedTextBatch};
use crate::model::{MinGRULM, MinGRULMConfig, TextBatch};

/// Configuration for TBPTT training
#[derive(Config)]
pub struct TBPTTConfig {
    /// Model configuration
    pub model: MinGRULMConfig,
    
    /// Optimizer configuration
    pub optimizer: AdamConfig,
    
    /// Learning rate
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    
    /// Frequency of parameter updates (k1 parameter)
    #[config(default = 4)]
    pub tbptt_k1: usize,
    
    /// Length of backpropagation window (k2 parameter)
    #[config(default = 8)]
    pub tbptt_k2: usize,
    
    /// Size of each chunk in tokens
    #[config(default = 64)]
    pub chunk_size: usize,
    
    /// Whether to preserve hidden states between batches
    #[config(default = true)]
    pub preserve_hidden_states: bool,
    
    /// Random seed for reproducibility
    #[config(default = 42)]
    pub seed: u64,
    
    /// Number of epochs to train
    #[config(default = 10)]
    pub num_epochs: usize,
    
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
#[derive(Module, Debug)]
pub struct TBPTTTrainer<B: AutodiffBackend> {
    model: MinGRULM<B>,
    #[module(skip)]
    optimizer: AdamConfig,
    #[module(skip)]
    hidden_states: HashMap<usize, Vec<Tensor<B, 2>>>,
    #[module(skip)]
    metrics: TBPTTMetrics,
    #[module(skip)]
    tbptt_k1: usize,
    #[module(skip)]
    tbptt_k2: usize,
    #[module(skip)]
    preserve_hidden_states: bool,
    #[module(skip)]
    chunk_size: usize,
    #[module(skip)]
    grad_clip: f32,
    #[module(skip)]
    learning_rate: f64,
}

impl<B: AutodiffBackend> TBPTTTrainer<B> {
    pub fn new(model: MinGRULM<B>, config: &TBPTTConfig) -> Self {
        Self {
            model,
            optimizer: config.optimizer.clone(),
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
    
    /// Process a single document chunk with TBPTT
    pub fn process_chunk<O: Optimizer<MinGRULM<B>, B>>(
        &mut self,
        chunk: &ChunkedTextBatch<B>,
        optimizer: &mut O,
        accumulator: &mut GradientsAccumulator<MinGRULM<B>>,
        accumulation_current: &mut usize,
        do_update: bool
    ) -> f32 {
        let batch_size = chunk.input.dims()[0];
        let seq_len = chunk.input.dims()[1];
        let device = chunk.input.device();
        
        // Get current hidden states for this document if available
        let mut current_hidden_states = Vec::new();
        
        // Process each document in the batch
        for i in 0..batch_size {
            let doc_id = chunk.doc_ids[i];
            let chunk_idx = chunk.chunk_indices[i];
            
            // Retrieve or initialize hidden state for this document
            let doc_hidden = if chunk_idx > 0 && self.hidden_states.contains_key(&doc_id) {
                let state = self.hidden_states.get(&doc_id).unwrap().clone();
                
                // Detach if we're beyond the backprop window (tbptt_k2)
                if chunk_idx % self.tbptt_k2 == 0 {
                    // Detach to truncate backpropagation
                    state.iter().map(|h| h.clone().detach()).collect()
                } else {
                    state
                }
            } else {
                // No previous state, initialize with zeros
                Vec::new()
            };
            
            current_hidden_states.push(doc_hidden);
        }
        
        // Flatten into vector of tensors for the batch
        let batch_hidden = if current_hidden_states.is_empty() || current_hidden_states[0].is_empty() {
            None
        } else {
            // Merge hidden states across batch dimension
            let hidden_dim = current_hidden_states[0][0].dims()[1];
            
            // Collect hidden states for each layer
            let mut merged_states = Vec::new();
            let num_layers = current_hidden_states[0].len();
            
            for layer in 0..num_layers {
                let layer_states: Vec<_> = current_hidden_states.iter()
                    .map(|doc_state| {
                        if doc_state.len() > layer {
                            doc_state[layer].clone()
                        } else {
                            // If layer doesn't exist, create zero tensor
                            Tensor::zeros([1, hidden_dim], &device)
                        }
                    })
                    .collect();
                
                // Stack along batch dimension
                merged_states.push(Tensor::cat(layer_states, 0));
            }
            
            Some(merged_states)
        };
        
        // Forward pass with hidden state
        let (logits, next_hidden) = self.model.forward(chunk.input.clone(), batch_hidden);
        
        // Update hidden states for each document
        for i in 0..batch_size {
            let doc_id = chunk.doc_ids[i];
            let is_last_chunk = chunk.is_last_chunks[i];
            
            // Extract this document's hidden state from the batch
            let doc_next_hidden: Vec<Tensor<B, 2>> = next_hidden.iter()
                .map(|h| {
                    // Extract slice for this document
                    h.clone().slice([i..i+1, 0..h.dims()[1]]).squeeze(0)
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
        let logits_reshaped = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_reshaped = chunk.target.reshape([batch_size * seq_len]);
        
        let loss_fn = CrossEntropyLossConfig::new().init(&device);
        let loss = loss_fn.forward(logits_reshaped.clone(), targets_reshaped.clone());
        
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
    
    /// Train the model for one epoch
    pub fn train_epoch<O: Optimizer<MinGRULM<B>, B>>(
        &mut self,
        dataloader: &ChunkedTextDataset,
        batcher: &TextBatcher<B>,
        optimizer: &mut O,
        epoch: usize,
    ) -> f32 {
        println!("Training epoch {}", epoch);
        
        // Progress bar for visualization
        let progress_bar = ProgressBar::new(dataloader.len() as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) - {msg}")
                .expect("Progress bar template error")
                .progress_chars("#>-")
        );
        
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        
        // Create batches 
        let batch_size = 32; // From config
        
        // Gradient accumulation state
        let mut accumulator = GradientsAccumulator::<MinGRULM<B>>::new();
        let mut accumulation_current = 0;
        
        // Process each item in dataset
        for i in 0..dataloader.len() {
            if let Some(chunk) = dataloader.get(i) {
                // Batch individual chunks
                let chunks = vec![chunk];
                let batch = batcher.batch(chunks);
                
                // Process chunk and update model if needed
                let do_update = (i + 1) % self.tbptt_k1 == 0 || i == dataloader.len() - 1;
                let loss = self.process_chunk(&batch, optimizer, &mut accumulator, &mut accumulation_current, do_update);
                
                total_loss += loss;
                batch_count += 1;
                
                // Update progress
                progress_bar.inc(1);
                progress_bar.set_message(format!("Loss: {:.6}", loss));
                
                // Update metrics
                self.metrics.update_progress((i as f32) / (dataloader.len() as f32));
            }
        }
        
        // Ensure we apply any remaining gradients
        if accumulation_current > 0 {
            let grads = accumulator.grads();
            self.model = optimizer.step(self.learning_rate, self.model.clone(), grads);
        }
        
        // Finalize metrics
        let epoch_loss = total_loss / batch_count as f32;
        self.metrics.update_batch_loss(epoch_loss);
        
        progress_bar.finish_with_message(format!("Epoch complete - Avg loss: {:.6}", epoch_loss));
        
        epoch_loss
    }
    
    /// Validate the model
    pub fn validate(
        &self,
        dataloader: &ChunkedTextDataset,
        batcher: &TextBatcher<B::InnerBackend>,
    ) -> f32 {
        println!("Validating model...");
        
        let progress_bar = ProgressBar::new(dataloader.len() as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.yellow/blue}] {pos}/{len} ({eta}) - {msg}")
                .expect("Progress bar template error")
                .progress_chars("#>-")
        );
        
        let model = self.model.valid();
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        
        // Process validation data
        for i in 0..dataloader.len() {
            if let Some(chunk) = dataloader.get(i) {
                // Batch individual chunks
                let chunks = vec![chunk];
                let batch = batcher.batch(chunks);
                
                // Forward pass (no gradients needed for validation)
                let (logits, _) = model.forward(batch.input.clone(), None);
                
                // Calculate loss
                let [batch_size, seq_len, vocab_size] = logits.dims();
                let logits_reshaped = logits.reshape([batch_size * seq_len, vocab_size]);
                let targets_reshaped = batch.target.reshape([batch_size * seq_len]);
                
                let device = logits_reshaped.device();
                let loss_fn = CrossEntropyLossConfig::new().init(&device);
                let loss = loss_fn.forward(logits_reshaped, targets_reshaped);
                
                let loss_value = loss.into_scalar().to_f32();
                total_loss += loss_value;
                batch_count += 1;
                
                // Update progress
                progress_bar.inc(1);
                progress_bar.set_message(format!("Loss: {:.6}", loss_value));
            }
        }
        
        let avg_loss = total_loss / batch_count as f32;
        progress_bar.finish_with_message(format!("Validation complete - Avg loss: {:.6}", avg_loss));
        
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
    B::seed(config.seed);
    
    // Initialize model
    let model = config.model.clone()
        .with_chunk_size(config.chunk_size)
        .init::<B>(device);
    
    // Initialize optimizer
    let mut optimizer = config.optimizer.init();
    
    // Create TBPTT trainer
    let mut trainer = TBPTTTrainer::new(model, config);
    
    // Create chunked dataset for document-aware processing
    let documents = prepare_documents(input_data, config.chunk_size);
    let train_dataset = ChunkedTextDataset::new(
        documents.clone(),
        config.chunk_size * config.tbptt_k1,
        config.chunk_size
    );
    
    // Create validation dataset (smaller subset)
    let valid_documents = documents.iter()
        .take(documents.len() / 5)
        .cloned()
        .collect();
    let valid_dataset = ChunkedTextDataset::new(
        valid_documents,
        config.chunk_size * config.tbptt_k1,
        config.chunk_size
    );
    
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
    
    for epoch in 1..=config.num_epochs {
        // Training phase
        let train_loss = trainer.train_epoch(&train_dataset, &train_batcher, &mut optimizer, epoch);
        
        // Validation phase
        let valid_loss = trainer.validate(&valid_dataset, &valid_batcher);
        
        println!("Epoch {}/{} - Train Loss: {:.6}, Valid Loss: {:.6}", 
            epoch, config.num_epochs, train_loss, valid_loss);
        
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

/// Helper function to prepare documents from input text
fn prepare_documents(input_text: &str, chunk_size: usize) -> Vec<String> {
    // Simple document preparation - split by paragraphs
    let paragraphs: Vec<String> = input_text
        .split("\n\n")
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect();
    
    // Ensure each document is large enough
    let min_document_size = chunk_size * 2;
    
    paragraphs.into_iter()
        .filter(|p| p.len() >= min_document_size)
        .collect()
}
