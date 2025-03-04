use burn::{
    config::Config,
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    optim::{Adam, AdamConfig, GradientsParams},
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{backend::AutodiffBackend, Backend, Int, Tensor},
    train::{ClassificationOutput, GradientsAccumulator, TrainOutput, TrainStep},
};

use crate::dataset::{CharVocab, TextBatch, TextBatcher, TextDataset};
use crate::model::{MinGRULM, MinGRULMConfig};

/// Configuration for TBPTT training
#[derive(Config, Debug)]
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
}

/// Struct for tracking the training state during TBPTT
struct TBPTTState<B: AutodiffBackend> {
    /// The model being trained
    model: MinGRULM<B>,
    
    /// The optimizer
    optimizer: Adam<MinGRULM<B>, B>,
    
    /// Hidden states carried between chunks
    hidden_states: Option<Vec<Tensor<B, 2>>>,
    
    /// Gradient accumulator for TBPTT
    grad_accumulator: GradientsAccumulator<MinGRULM<B>, B>,
    
    /// Current chunk in the sequence
    current_chunk: usize,
    
    /// Number of chunks to accumulate before update
    tbptt_chunks: usize,
    
    /// Learning rate
    learning_rate: f64,
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
    
    // Initialize model and optimizer
    let model = config.model
        .with_chunk_size(config.chunk_size)
        .init::<B>(device);
    
    let optimizer = config.optimizer.init();
    
    // Create dataset with appropriate sequence length
    let seq_length = config.chunk_size * config.tbptt_chunks;
    let dataset = TextDataset::new_with_random_sampling(
        input_data.to_string(),
        seq_length,
        0.5, // Coverage factor
        config.seed,
        config.chunk_size,
    );
    
    // Create dataloader
    let batcher = TextBatcher::<B>::new(vocab.clone(), device.clone());
    let dataloader = burn::data::dataloader::DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .build(dataset);
    
    // Initialize TBPTT state
    let mut tbptt_state = TBPTTState {
        model,
        optimizer,
        hidden_states: None,
        grad_accumulator: GradientsAccumulator::new(),
        current_chunk: 0,
        tbptt_chunks: config.tbptt_chunks,
        learning_rate: config.learning_rate,
    };
    
    // Training loop
    for epoch in 1..=config.num_epochs {
        println!("Epoch {}/{}", epoch, config.num_epochs);
        
        // Reset hidden states between epochs if not preserving
        if !config.preserve_hidden_states {
            tbptt_state.hidden_states = None;
        }
        
        // Process each batch
        let mut batch_losses = Vec::new();
        for (batch_idx, batch) in dataloader.iter().enumerate() {
            let batch_loss = process_batch(&mut tbptt_state, batch, device);
            batch_losses.push(batch_loss);
            
            if batch_idx % 10 == 0 {
                let avg_loss = batch_losses.iter().sum::<f32>() / batch_losses.len() as f32;
                println!("  Batch {}: Loss = {:.4}", batch_idx, avg_loss);
            }
        }
        
        // Report epoch metrics
        let epoch_loss = batch_losses.iter().sum::<f32>() / batch_losses.len() as f32;
        println!("Epoch {}: Loss = {:.4}", epoch, epoch_loss);
        
        // Save checkpoint after each epoch
        save_checkpoint(&tbptt_state.model, artifact_dir, epoch);
    }
    
    // Return the trained model
    tbptt_state.model
}

/// Process a single batch with TBPTT
fn process_batch<B: AutodiffBackend>(
    state: &mut TBPTTState<B>,
    batch: TextBatch<B>,
    device: &B::Device,
) -> f32 {
    let mut total_loss = 0.0;
    let batch_size = batch.input.dims()[0];
    
    // Split the batch into chunks for TBPTT
    let seq_len = batch.input.dims()[1];
    let chunk_size = seq_len / state.tbptt_chunks;
    
    for chunk_idx in 0..state.tbptt_chunks {
        // Get the chunk slice from the batch
        let start = chunk_idx * chunk_size;
        let end = start + chunk_size;
        
        let chunk_input = batch.input.clone().slice([0..batch_size, start..end]);
        let chunk_target = batch.target.clone().slice([0..batch_size, start..end]);
        
        // Forward pass with hidden state passing
        let (logits, next_hidden_states) = state.model.forward(chunk_input.clone(), state.hidden_states.clone());
        
        // Calculate loss
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let logits_reshaped = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_reshaped = chunk_target.reshape([batch_size * seq_len]);
        
        let loss_fn = CrossEntropyLossConfig::new().init(device);
        let loss = loss_fn.forward(logits_reshaped.clone(), targets_reshaped.clone());
        total_loss += loss.clone().to_scalar();
        
        // Create output for gradient calculation
        let output = ClassificationOutput::new(loss.clone(), logits_reshaped, targets_reshaped);
        
        // Update hidden states for next chunk
        state.hidden_states = Some(next_hidden_states);
        
        // Backward pass and gradient accumulation
        let grads = loss.backward();
        
        // Accumulate gradients
        state.grad_accumulator.accumulate(&state.model, grads);
        
        state.current_chunk += 1;
        
        // Update model if we've accumulated enough chunks
        if state.current_chunk >= state.tbptt_chunks {
            let acc_grads = state.grad_accumulator.gradients();
            state.model = state.optimizer.step(
                state.learning_rate,
                &state.model,
                &acc_grads
            );
            state.grad_accumulator = GradientsAccumulator::new();
            state.current_chunk = 0;
        }
    }
    
    total_loss / state.tbptt_chunks as f32
}

/// Save model checkpoint
fn save_checkpoint<B: Backend>(
    model: &MinGRULM<B>,
    artifact_dir: &str,
    epoch: usize,
) {
    let checkpoint_path = format!("{}/model_epoch_{}.bin", artifact_dir, epoch);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    model
        .save_file(checkpoint_path, &recorder)
        .expect("Failed to save checkpoint");
}
