use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    record::{BinFileRecorder, FullPrecisionSettings},
    train::{LearnerBuilder, metric::LossMetric},
};
use grufinity::{
    model::MinGRULMConfig,
    dataset::{CharVocab, TextDataset, TextBatcher},
    Module,
    use_configured_backend,
};
use burn::data::dataset::Dataset;
use std::{
    fs,
    time::Instant,
};

// Import the backend types from the macro
use grufinity::{RawBackend, BackendWithAutodiff};

#[derive(Config)]
struct TrainingConfig {
    #[config(default = "50")]
    num_epochs: usize,
    #[config(default = "32")]
    batch_size: usize,
    #[config(default = "4")]
    num_workers: usize,
    #[config(default = "42")]
    seed: u64,
    #[config(default = "1.0e-3")]
    learning_rate: f64,
    #[config(default = "128")]
    sequence_length: usize,
    #[config(default = "3")]
    step_size: usize,
    #[config(default = "256")]
    chunk_size: usize,
    #[config(default = "0.5")]
    coverage_factor: f64,
    
    model: MinGRULMConfig,
    optimizer: AdamConfig,
}

fn main() {
    // Parse command-line arguments
    let args: Vec<String> = std::env::args().collect();
    
    // Default values
    let mut data_path = "data/sample.txt".to_string();
    let mut artifact_dir = "mingru_artifacts".to_string();
    let mut config_path = "".to_string();
    
    // Parse arguments
    for i in 1..args.len() {
        match args[i].as_str() {
            "--data" => {
                if i + 1 < args.len() {
                    data_path = args[i + 1].clone();
                }
            },
            "--output" => {
                if i + 1 < args.len() {
                    artifact_dir = args[i + 1].clone();
                }
            },
            "--config" => {
                if i + 1 < args.len() {
                    config_path = args[i + 1].clone();
                }
            },
            _ => {}
        }
    }
    
    // Set up the configured backend
    use_configured_backend!();
    
    // Get the device from the macro
    let device;
    
    #[cfg(feature = "cuda-jit")]
    {
        use burn::backend::cuda_jit::CudaJitDevice;
        device = CudaJitDevice::new(0); // Use first CUDA device with JIT
        println!("Using CUDA JIT device");
    }
    
    #[cfg(all(feature = "candle", feature = "burn/candle-cuda", not(feature = "cuda-jit")))]
    {
        use burn::backend::candle::CandleDevice;
        device = CandleDevice::Cuda(0);  // Use first CUDA device via Candle
        println!("Using Candle CUDA device");
    }
    
    #[cfg(all(feature = "wgpu", not(feature = "cuda-jit"), not(all(feature = "candle", feature = "burn/candle-cuda"))))]
    {
        use burn::backend::wgpu::WgpuDevice;
        device = WgpuDevice::default();
    }
    
    #[cfg(all(feature = "candle", not(any(feature = "cuda", feature = "wgpu"))))]
    {
        use burn::backend::candle::CandleDevice;
        device = CandleDevice::Cpu;
    }
    
    #[cfg(all(feature = "ndarray", not(any(feature = "cuda", feature = "wgpu", feature = "candle"))))]
    {
        use burn::backend::ndarray::NdArrayDevice;
        device = NdArrayDevice;
    }
    
    #[cfg(all(feature = "tch", not(any(feature = "cuda", feature = "wgpu", feature = "candle", feature = "ndarray"))))]
    {
        use burn::backend::libtorch::LibTorchDevice;
        device = LibTorchDevice::Cpu;
    }
    
    // Create a directory for artifacts
    fs::create_dir_all(&artifact_dir).expect("Failed to create artifact directory");
    
    // Load training data
    let text = fs::read_to_string(&data_path).unwrap_or_else(|_| {
        println!("Could not read file {}, using sample text", data_path);
        "Hello world! This is a sample text for the MinGRU model.".to_string()
    });
    
    // Create vocabulary from text
    let mut vocab = CharVocab::new();
    vocab.build_from_text(&text);
    println!("Vocabulary size: {}", vocab.size());
    
    // Save vocabulary for later use
    let vocab_path = format!("{}/vocab.txt", artifact_dir);
    vocab.save_to_file(&vocab_path).expect("Failed to save vocabulary");
    
    // Configure the model and training
    let config = if !config_path.is_empty() {
        match TrainingConfig::load(&config_path) {
            Ok(cfg) => {
                println!("Loaded configuration from {}", config_path);
                cfg
            },
            Err(e) => {
                println!("Error loading config: {}. Using default.", e);
                create_default_config()
            }
        }
    } else {
        create_default_config()
    };
    
    // Save config for reproducibility
    config.save(format!("{}/config.json", artifact_dir))
        .expect("Failed to save config");
    
    println!("Training with sequence length: {}, step size: {}", 
             config.sequence_length, config.step_size);
    println!("Vocabulary size: {}", vocab.size());
    
    // Set up the dataset with random sampling
    let dataset = TextDataset::new_with_random_sampling(
        text.clone(),
        config.sequence_length,
        config.coverage_factor,
        config.seed,
        config.chunk_size,
    );
    
    println!("Dataset size: {} sequences", dataset.len());
    println!("Using random sampling with coverage factor: {}", config.coverage_factor);
    
    // Create training and validation splits with different seeds
    let train_dataset = TextDataset::new_with_random_sampling(
        text.clone(),
        config.sequence_length,
        config.coverage_factor,
        config.seed,
        config.chunk_size,
    );
    
    let valid_dataset = TextDataset::new_with_random_sampling(
        text,
        config.sequence_length,
        config.coverage_factor / 5.0, // Less coverage for validation
        config.seed + 1, // Different seed for validation
        config.chunk_size,
    );
    
    // Set up batchers
    let train_batcher = TextBatcher::<BackendWithAutodiff>::new(vocab.clone(), device.clone());
    let valid_batcher = TextBatcher::<RawBackend>::new(vocab.clone(), device.clone());
    
    // Set up dataloaders
    let train_dataloader = DataLoaderBuilder::new(train_batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);
    
    let valid_dataloader = DataLoaderBuilder::new(valid_batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_dataset);
    
    // Initialize learner with random seed
    // Use the Backend trait method
    <BackendWithAutodiff as burn::tensor::backend::Backend>::seed(config.seed);
    
    let learner = LearnerBuilder::new(&artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .build(
            config.model.init::<BackendWithAutodiff>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );
    
    // Train the model
    println!("Starting training for {} epochs", config.num_epochs);
    let start_time = Instant::now();
    let model_trained = learner.fit(train_dataloader, valid_dataloader);
    let duration = start_time.elapsed();
    println!("Training completed in {:.2} seconds", duration.as_secs_f64());
    
    // Save the trained model
    let model_path = format!("{}/model_final.bin", artifact_dir);
    model_trained.clone()
        .save_file(&model_path, &BinFileRecorder::<FullPrecisionSettings>::new())
        .expect("Failed to save trained model");
    
    println!("Training complete! Model saved to {}\n\n\n", model_path);
    
    // NOTE: Sample generation disabled - use the generate binary instead
    // To generate text with the trained model use:
    // cargo run --release --bin generate -- --model artifacts_dir/model_final.bin --vocab artifacts_dir/vocab.txt
}

fn create_default_config() -> TrainingConfig {
    // Configure the model
    let model_config = MinGRULMConfig::new(
        256,           // num_tokens (all possible byte values)
        128            // dimension (increased from 96)
    )
    .with_depth(4)     // increased from 2
    .with_ff_mult(3.0) // increased from 2.0
    .with_expansion_factor(1.5) // increased from 1.2
    .with_chunk_size(256);
    
    let optimizer_config = AdamConfig::new();
    
    TrainingConfig::new(
        model_config,
        optimizer_config,
    )
    .with_sequence_length(128)
    .with_step_size(3)
    .with_batch_size(32)
    .with_num_epochs(10)
    .with_learning_rate(1e-3)
}
