use burn::{
    config::Config,
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsAccumulator, GradientsParams},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{backend::{AutodiffBackend, Backend}, Tensor, cast::ToElement},
    train::{
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
        metric::{LossMetric, MetricEntry},
    },
    backend::wgpu::Wgpu,
};
use std::collections::HashMap;
use std::sync::Mutex;
use std::cell::RefCell;

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
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    
    /// Number of chunks to process before performing backpropagation
    #[config(default = 4)]
    pub tbptt_chunks: usize,
    
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
        entry.push(MetricEntry {
            name: name.to_string(),
            formatted: format!("{:.6}", value),  // Formatted string value
            serialize: value.to_string(),        // Raw value for serialization
        });
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

// Custom metrics interface
trait CustomMetrics {
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

/// TBPTT Trainer that implements the TrainStep trait
#[derive(Module, Debug)]
pub struct TBPTTTrainer<B: AutodiffBackend> {
    model: MinGRULM<B>,
    #[module(skip)]
    hidden_states: Option<Vec<Tensor<B, 2>>>,
    #[module(skip)]
    current_chunk: usize,
    #[module(skip)]
    tbptt_chunks: usize,
    #[module(skip)]
    preserve_hidden_states: bool,
    #[module(skip)]
    chunk_size: usize,
    #[module(skip)]
    grad_clip: f32,
}

// State managed externally
// Define backend type
type WgpuBackend = Wgpu<f32, i32>;

#[derive(Clone)]
struct TBPTTTrainerMetrics {
    metrics: TBPTTMetrics,
}

// Global state - normally not ideal but works for this training scenario
static METRICS: Mutex<Option<TBPTTMetrics>> = Mutex::new(None);
// Hidden state management - needs to be thread local for training
thread_local! {
    static HIDDEN_STATES: RefCell<Option<Vec<Vec<f32>>>> = RefCell::new(None);
    static CURRENT_CHUNK: RefCell<usize> = RefCell::new(0);
}

impl<B: AutodiffBackend> TBPTTTrainer<B> {
    pub fn new(model: MinGRULM<B>, config: &TBPTTConfig) -> Self {
        // Initialize global metrics
        let metrics = TBPTTMetrics::new();
        *METRICS.lock().unwrap() = Some(metrics);
        
        // Initialize thread locals
        HIDDEN_STATES.with(|cell| { *cell.borrow_mut() = None; });
        CURRENT_CHUNK.with(|cell| { *cell.borrow_mut() = 0; });
        
        Self {
            model,
            hidden_states: None, // Keep this field for compatibility
            current_chunk: 0,    // Keep this field for compatibility
            tbptt_chunks: config.tbptt_chunks,
            preserve_hidden_states: config.preserve_hidden_states,
            chunk_size: config.chunk_size,
            grad_clip: config.grad_clip,
        }
    }
    
    pub fn reset_hidden_states(&mut self) {
        // Reset thread local hidden states
        HIDDEN_STATES.with(|cell| { *cell.borrow_mut() = None; });
    }
    
    pub fn metrics(&self) -> TBPTTMetrics {
        METRICS.lock().unwrap().clone().unwrap_or_else(TBPTTMetrics::new)
    }
    
    fn update_metrics(&self, loss: f32) {
        if let Some(metrics) = &mut *METRICS.lock().unwrap() {
            metrics.update_chunk_loss(loss);
        }
    }
    
    fn update_batch_metrics(&self, loss: f32) {
        if let Some(metrics) = &mut *METRICS.lock().unwrap() {
            metrics.update_batch_loss(loss);
            metrics.clear_chunk_losses();
        }
    }
}

impl<B: AutodiffBackend> TrainStep<TextBatch<B>, ClassificationOutput<B>> for TBPTTTrainer<B> 
where 
    <B as AutodiffBackend>::InnerBackend: AutodiffBackend
{
    fn step(&self, batch: TextBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let batch_size = batch.input.dims()[0];
        let device = batch.input.device();
        
        // Split the batch into chunks for TBPTT
        let seq_len = batch.input.dims()[1];
        let chunk_size = seq_len / self.tbptt_chunks;
        
        let mut total_loss = 0.0;
        
        // Create a new accumulator for this batch if needed
        let mut grad_accumulator = GradientsAccumulator::new();
        
        // Retrieve current hidden states from thread local storage
        let current_hidden_states: Option<Vec<Tensor<B, 2>>> = HIDDEN_STATES.with(|cell| {
            cell.borrow().as_ref().map(|vec_of_vecs| {
                vec_of_vecs.iter().map(|vec| {
                    // Reconstruct tensor from stored data
                    let tensor_data: Vec<f32> = vec.clone();
                    // Assuming hidden dim can be inferred from data length
                    let hidden_dim = tensor_data.len() / batch_size;
                    Tensor::<B, 2>::from_data(&*tensor_data.into_iter().map(|x| x as f32).collect::<Vec<f32>>(), &batch.input.device())
                        .reshape([batch_size, hidden_dim])
                }).collect()
            })
        });
        
        for chunk_idx in 0..self.tbptt_chunks {
            // Get chunk slice
            let start = chunk_idx * chunk_size;
            let end = start + chunk_size;
            
            let chunk_input = batch.input.clone().slice([0..batch_size, start..end]);
            let chunk_target = batch.target.clone().slice([0..batch_size, start..end]);
            
            // Forward pass
            let (logits, next_hidden_states) = self.model.forward(chunk_input.clone(), current_hidden_states.clone());
            
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
            self.update_metrics(loss_value);
            
            // Update hidden states for next chunk - detach to prevent backprop through time
            // Use thread local storage since we can't mutate self
            let detached_states: Vec<Tensor<B, 2>> = next_hidden_states.iter().map(|h| h.clone().detach()).collect();
            
            // Store hidden states in thread local storage
            HIDDEN_STATES.with(|cell| {
                *cell.borrow_mut() = Some(detached_states.iter().map(|t| t.clone().to_data().into_vec().unwrap()).collect());
            });
            
            // Backward pass and accumulate gradients
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &self.model);
            
            // Apply gradient clipping if configured
            let effective_grads = if self.grad_clip > 0.0 {
                // TODO: Implement gradient clipping
                grads_params
            } else {
                grads_params
            };
            
            grad_accumulator.accumulate(&self.model, effective_grads);
            
            // Update current chunk counter
            CURRENT_CHUNK.with(|cell| {
                *cell.borrow_mut() += 1;
            });
        }
        
        // Create output for the full batch
        let output = ClassificationOutput::new(
            Tensor::full([], total_loss / self.tbptt_chunks as f32, &device),
            Tensor::zeros([1, 1], &device), // Dummy tensor
            Tensor::zeros([1], &device),    // Dummy tensor
        );
        
        // Record average batch loss
        self.update_batch_metrics(total_loss / self.tbptt_chunks as f32);
        
        // Get the gradients from the output directly
        let grads = output.loss.backward();
        
        // Create train output
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
) -> MinGRULM<B> 
where 
    B::InnerBackend: Backend
{
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
    let train_batcher = TextBatcher::<B>::new(vocab.clone(), device.clone());
    let valid_batcher = TextBatcher::<B::InnerBackend>::new(vocab.clone(), device.clone());
    
    let train_dataloader = burn::data::dataloader::DataLoaderBuilder::new(train_batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .build(dataset);
        
    let valid_dataloader = burn::data::dataloader::DataLoaderBuilder::new(valid_batcher)
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

// Add ValidStep implementation for validation data
impl<B: AutodiffBackend> ValidStep<TextBatch<B::InnerBackend>, ClassificationOutput<B::InnerBackend>> for TBPTTTrainer<B> 
where
    B::InnerBackend: Backend
{
    fn step(&self, batch: TextBatch<B::InnerBackend>) -> ClassificationOutput<B::InnerBackend> {
        // Use valid() to get a non-autodiff version of the model for validation
        let model = self.model.valid();
        
        // Forward pass through the model
        let (logits, _) = model.forward(batch.input, None);
        
        // Get dimensions and reshape for loss calculation
        let [batch_size, seq_len, vocab_size] = logits.dims();
        
        // Reshape for classification output
        let logits_reshaped = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_reshaped = batch.target.reshape([batch_size * seq_len]);
        
        // Compute cross-entropy loss
        let loss_fn = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits_reshaped.device());
        let loss = loss_fn.forward(logits_reshaped.clone(), targets_reshaped.clone());
        
        // Create classification output
        ClassificationOutput::new(loss, logits_reshaped, targets_reshaped)
    }
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
