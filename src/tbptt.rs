use burn::{
    config::Config,
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsAccumulator, GradientsParams},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{backend::{AutodiffBackend, Backend}, Tensor, cast::ToElement},
    train::{
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep, 
        metric::{LossMetric, Metric, MetricEntry, Metrics}, MetricsConfig,
    },
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::dataset::{CharVocab, TextBatcher, TextDataset};
use crate::model::{MinGRULM, MinGRULMConfig, TextBatch};

/// Configuration for TBPTT training
#[derive(Config)]
pub struct TBPTTConfig {
    /// Model configuration
    pub model: MinGRULMConfig,
    
    /// Optimizer configuration
    pub optimizer: AdamConfig,
    
    /// Learning rate
    #[config(default = "1e-3")]
    pub learning_rate: f64,
    
    /// Number of chunks to process before performing backpropagation
    #[config(default = "4")]
    pub tbptt_chunks: usize,
    
    /// Size of each chunk in tokens
    #[config(default = "64")]
    pub chunk_size: usize,
    
    /// Whether to preserve hidden states between batches
    #[config(default = "true")]
    pub preserve_hidden_states: bool,
    
    /// Random seed for reproducibility
    #[config(default = "42")]
    pub seed: u64,
    
    /// Number of epochs to train
    #[config(default = "10")]
    pub num_epochs: usize,
    
    /// Batch size
    #[config(default = "32")]
    pub batch_size: usize,
    
    /// Log interval (in batches)
    #[config(default = "10")]
    pub log_interval: usize,
    
    /// Checkpoint interval (in epochs)
    #[config(default = "1")]
    pub checkpoint_interval: usize,
    
    /// Metrics configuration
    #[config(default)]
    pub metrics: MetricsConfig,
    
    /// Gradient clipping value (0.0 to disable)
    #[config(default = "0.0")]
    pub grad_clip: f32,
}

/// TBPTT metrics for tracking training progress
#[derive(Clone)]
struct TBPTTMetrics {
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
    fn new() -> Self {
        Self {
            batch_losses: Vec::new(),
            chunk_losses: Vec::new(),
            current_lr: 0.0,
            epoch_progress: 0.0,
            metrics: HashMap::new(),
        }
    }
    
    fn update_batch_loss(&mut self, loss: f32) {
        self.batch_losses.push(loss);
        self.record_metric("batch_loss", loss);
    }
    
    fn update_chunk_loss(&mut self, loss: f32) {
        self.chunk_losses.push(loss);
        self.record_metric("chunk_loss", loss);
    }
    
    fn record_metric(&mut self, name: &str, value: f32) {
        let entry = self.metrics.entry(name.to_string())
            .or_insert_with(Vec::new);
        entry.push(MetricEntry::new(value));
    }
    
    fn update_lr(&mut self, lr: f64) {
        self.current_lr = lr;
        self.record_metric("learning_rate", lr as f32);
    }
    
    fn update_progress(&mut self, progress: f32) {
        self.epoch_progress = progress;
    }
    
    fn avg_batch_loss(&self) -> f32 {
        if self.batch_losses.is_empty() {
            0.0
        } else {
            self.batch_losses.iter().sum::<f32>() / self.batch_losses.len() as f32
        }
    }
    
    fn avg_chunk_loss(&self) -> f32 {
        if self.chunk_losses.is_empty() {
            0.0
        } else {
            self.chunk_losses.iter().sum::<f32>() / self.chunk_losses.len() as f32
        }
    }
    
    fn clear_chunk_losses(&mut self) {
        self.chunk_losses.clear();
    }
}

impl Metrics for TBPTTMetrics {
    fn get(&self, key: &str) -> Option<&Vec<MetricEntry>> {
        self.metrics.get(key)
    }
    
    fn keys(&self) -> Vec<String> {
        self.metrics.keys().cloned().collect()
    }
}

/// Struct for tracking the training state during TBPTT
struct TBPTTState<B: AutodiffBackend> {
    /// The model being trained
    model: MinGRULM<B>,
    
    /// Hidden states carried between chunks
    hidden_states: Option<Vec<Tensor<B, 2>>>,
    
    /// Gradient accumulator for TBPTT
    grad_accumulator: GradientsAccumulator<MinGRULM<B>>,
    
    /// Current chunk in the sequence
    current_chunk: usize,
    
    /// Number of chunks to accumulate before update
    tbptt_chunks: usize,
    
    /// Metrics tracking
    metrics: TBPTTMetrics,
}

/// TBPTT Trainer that implements the TrainStep trait
#[derive(Module, Debug)]
pub struct TBPTTTrainer<B: AutodiffBackend> {
    model: MinGRULM<B>,
    #[module(skip)]
    state: Arc<Mutex<TBPTTTrainerState<B>>>,
}

struct TBPTTTrainerState<B: AutodiffBackend> {
    hidden_states: Option<Vec<Tensor<B, 2>>>,
    grad_accumulator: GradientsAccumulator<MinGRULM<B>>,
    current_chunk: usize,
    tbptt_chunks: usize,
    preserve_hidden_states: bool,
    chunk_size: usize,
    metrics: TBPTTMetrics,
    grad_clip: f32,
}

impl<B: AutodiffBackend> TBPTTTrainer<B> {
    pub fn new(model: MinGRULM<B>, config: &TBPTTConfig) -> Self {
        let state = TBPTTTrainerState {
            hidden_states: None,
            grad_accumulator: GradientsAccumulator::new(),
            current_chunk: 0,
            tbptt_chunks: config.tbptt_chunks,
            preserve_hidden_states: config.preserve_hidden_states,
            chunk_size: config.chunk_size,
            metrics: TBPTTMetrics::new(),
            grad_clip: config.grad_clip,
        };
        
        Self {
            model,
            state: Arc::new(Mutex::new(state)),
        }
    }
    
    pub fn reset_hidden_states(&self) {
        let mut state = self.state.lock().unwrap();
        state.hidden_states = None;
    }
    
    pub fn metrics(&self) -> TBPTTMetrics {
        self.state.lock().unwrap().metrics.clone()
    }
}

impl<B: AutodiffBackend> TrainStep<TextBatch<B>, ClassificationOutput<B>> for TBPTTTrainer<B> {
    fn step(&self, batch: TextBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let mut state = self.state.lock().unwrap();
        let batch_size = batch.input.dims()[0];
        let device = batch.input.device();
        
        // Debug batch shape
        println!("Batch input shape: {:?}", batch.input.dims());
        
        // Split the batch into chunks for TBPTT
        let seq_len = batch.input.dims()[1];
        let chunk_size = seq_len / state.tbptt_chunks;
        
        let mut total_loss = 0.0;
        
        for chunk_idx in 0..state.tbptt_chunks {
            // Get chunk slice
            let start = chunk_idx * chunk_size;
            let end = start + chunk_size;
            
            let chunk_input = batch.input.clone().slice([0..batch_size, start..end]);
            let chunk_target = batch.target.clone().slice([0..batch_size, start..end]);
            
            // Debug tensor shapes
            println!("Chunk input shape: {:?}", chunk_input.dims());
            if let Some(ref hidden_states) = state.hidden_states {
                println!("Hidden states count: {}", hidden_states.len());
                if !hidden_states.is_empty() {
                    println!("Hidden state shape: {:?}", hidden_states[0].dims());
                }
            }
            
            // Forward pass
            let (logits, next_hidden_states) = self.model.forward(chunk_input.clone(), state.hidden_states.clone());
            
            // Debug output shapes
            println!("Output logits shape: {:?}", logits.dims());
            println!("Next hidden states length: {}", next_hidden_states.len());
            if !next_hidden_states.is_empty() {
                println!("Next hidden state shape: {:?}", next_hidden_states[0].dims());
            }
            
            // Calculate loss
            let [batch_size, seq_len, vocab_size] = logits.dims();
            let logits_reshaped = logits.reshape([batch_size * seq_len, vocab_size]);
            let targets_reshaped = chunk_target.reshape([batch_size * seq_len]);
            
            let loss_fn = CrossEntropyLossConfig::new().init(&device);
            let loss = loss_fn.forward(logits_reshaped.clone(), targets_reshaped.clone());
            
            // Convert loss to f32
            let scalar_value = loss.clone().into_scalar();
            let loss_value = scalar_value.to_f32();
            total_loss += loss_value;
            
            // Record metrics
            state.metrics.update_chunk_loss(loss_value);
            
            // Update hidden states for next chunk - detach to prevent backprop through time
            state.hidden_states = Some(next_hidden_states.iter().map(|h| h.clone().detach()).collect());
            
            // Backward pass and accumulate gradients
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &self.model);
            
            // Apply gradient clipping if configured
            let effective_grads = if state.grad_clip > 0.0 {
                // TODO: Implement gradient clipping
                grads_params
            } else {
                grads_params
            };
            
            state.grad_accumulator.accumulate(&self.model, effective_grads);
            state.current_chunk += 1;
        }
        
        // Create output for the full batch
        let output = ClassificationOutput::new(
            Tensor::from_f32(total_loss / state.tbptt_chunks as f32, &device),
            Tensor::zeros([1, 1], &device), // Dummy tensor
            Tensor::zeros([1], &device),    // Dummy tensor
        );
        
        // Record average batch loss
        state.metrics.update_batch_loss(total_loss / state.tbptt_chunks as f32);
        
        // Get accumulated gradients but don't update yet - let the Learner handle it
        let grads = state.grad_accumulator.grads();
        
        // Reset accumulator if we've processed all chunks
        if state.current_chunk >= state.tbptt_chunks {
            state.grad_accumulator = GradientsAccumulator::new();
            state.current_chunk = 0;
            state.metrics.clear_chunk_losses();
        }
        
        TrainOutput::new(self, grads, output)
    }
}

/// Train a language model using Truncated Backpropagation Through Time (TBPTT)
/// with Burn's Learner API for metrics, checkpointing, and visualization
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
    
    // Create TBPTT trainer
    let trainer = TBPTTTrainer::new(model.clone(), config);
    
    // Create dataset with appropriate sequence length
    let seq_length = config.chunk_size * config.tbptt_chunks;
    let dataset = TextDataset::new_with_random_sampling(
        input_data.to_string(),
        seq_length,
        0.5, // Coverage factor
        config.seed,
        config.chunk_size,
    );
    
    // Create validation dataset (smaller)
    let valid_dataset = TextDataset::new_with_random_sampling(
        input_data.to_string(),
        seq_length,
        0.1, // Smaller coverage for validation
        config.seed + 1, // Different seed
        config.chunk_size,
    );
    
    // Create dataloaders
    let batcher = TextBatcher::<B>::new(vocab.clone(), device.clone());
    let train_dataloader = burn::data::dataloader::DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .build(dataset);
        
    let valid_dataloader = burn::data::dataloader::DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .build(valid_dataset);
    
    // Build learner for metrics and checkpointing
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .build(
            trainer,
            config.optimizer.init::<B, TBPTTTrainer<B>>(),
            config.learning_rate,
        );
    
    // Train the model
    println!("Starting TBPTT training with Learner API");
    println!("Chunk size: {}, Chunks per update: {}", config.chunk_size, config.tbptt_chunks);
    
    let trained_trainer = learner.fit(train_dataloader, valid_dataloader);
    
    // Extract the trained model
    let trained_model = trained_trainer.model;
    
    // Save final model
    save_checkpoint(&trained_model, artifact_dir, config.num_epochs);
    
    trained_model
}

// Process_batch is now implemented as part of the TBPTTTrainer

/// Save model checkpoint
fn save_checkpoint<B: Backend>(
    model: &MinGRULM<B>,
    artifact_dir: &str,
    epoch: usize,
) {
    let checkpoint_path = format!("{}/model_epoch_{}.bin", artifact_dir, epoch);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    model.clone()
        .save_file(checkpoint_path, &recorder)
        .expect("Failed to save checkpoint");
}
