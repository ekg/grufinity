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
    // Initialize device based on enabled features - each cfg block has its own device variable
    #[allow(unused_assignments)]
    let mut device_initialized = false;
    
    // Use appropriate device type for each backend
    #[cfg(feature = "cuda-jit")]
    let mut device = {
        use burn::backend::cuda_jit::CudaDevice;
        device_initialized = true;
        println!("Using CUDA JIT device");
        CudaDevice::new(0)
    };
    
    #[cfg(all(feature = "candle-cuda", not(feature = "cuda-jit")))]
    let mut device = {
        use burn::backend::candle::CandleDevice;
        device_initialized = true;
        println!("Using Candle CUDA device");
        CandleDevice::cuda(0)
    };
    
    #[cfg(all(feature = "candle-metal", not(feature = "cuda-jit"), 
              not(all(feature = "candle", feature = "candle-cuda"))))]
    let device = {
        use burn::backend::candle::CandleDevice;
        device_initialized = true;
        println!("Using Candle Metal device");
        CandleDevice::metal(0)
    };
    
    #[cfg(all(feature = "wgpu", not(feature = "cuda-jit"),
              not(feature = "candle-cuda"), not(feature = "candle-metal"),
              not(feature = "candle")))]
    let device = {
        use burn::backend::wgpu::WgpuDevice;
        device_initialized = true;
        println!("Using WGPU device");
        WgpuDevice::default()
    };
    
    #[cfg(all(feature = "candle", not(feature = "candle-cuda"), not(feature = "cuda-jit"), 
              not(feature = "wgpu"), not(feature = "candle-metal")))]
    let device = {
        use burn::backend::candle::CandleDevice;
        device_initialized = true;
        println!("Using Candle CPU device");
        CandleDevice::cpu()
    };
    
    #[cfg(all(feature = "ndarray", not(feature = "cuda-jit"), not(feature = "wgpu"), 
              not(feature = "candle"), not(feature = "candle-metal"), not(feature = "candle-cuda")))]
    let device = {
        use burn::backend::ndarray::NdArrayDevice;
        device_initialized = true;
        println!("Using NdArray device");
        NdArrayDevice
    };
    
    #[cfg(all(feature = "tch", not(feature = "cuda-jit"), not(feature = "wgpu"), 
              not(feature = "candle"), not(feature = "ndarray"), not(feature = "candle-metal"), 
              not(feature = "candle-cuda")))]
    let device = {
        use burn::backend::libtorch::LibTorchDevice;
        device_initialized = true;
        println!("Using LibTorch CPU device");
        LibTorchDevice::Cpu
    };
    
    // Error if no backend feature is enabled
    #[cfg(not(any(feature = "cuda-jit", feature = "wgpu", feature = "candle", 
                  feature = "ndarray", feature = "tch", feature = "candle-metal", 
                  feature = "candle-cuda")))]
    compile_error!("No backend feature was enabled. Please enable at least one: cuda-jit, wgpu, candle, ndarray, etc.");
    
    // Removed redundant device initialization blocks
    
    // Fallback to ensure device is always initialized
    // This will only run if none of the above cfg blocks matched
    if !device_initialized {
        #[cfg(feature = "cuda-jit")]
        {
            use burn::backend::cuda_jit::CudaDevice;
            device = CudaDevice::new(0);
            _device_initialized = true;
            println!("Using CUDA JIT device (fallback)");
        }
        
        #[cfg(all(not(feature = "cuda-jit"), feature = "wgpu"))]
        {
            use burn::backend::wgpu::WgpuDevice;
            device = WgpuDevice::default();
            _device_initialized = true;
            println!("Using WGPU device (fallback)");
        }
        
        #[cfg(all(not(feature = "cuda-jit"), not(feature = "wgpu"), feature = "candle"))]
        {
            use burn::backend::candle::CandleDevice;
            device = CandleDevice::Cpu;
            device_initialized = true;
            println!("Using Candle CPU device (fallback)");
        }
        
        #[cfg(all(not(feature = "cuda-jit"), not(feature = "wgpu"), not(feature = "candle"), feature = "ndarray"))]
        {
            use burn::backend::ndarray::NdArrayDevice;
            device = NdArrayDevice;
            let _device_initialized = true;
            println!("Using NdArray device (fallback)");
        }
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
    
    // Initialize learner with random seed (if supported)
    match std::panic::catch_unwind(|| {
        <BackendWithAutodiff as burn::tensor::backend::Backend>::seed(config.seed);
    }) {
        Ok(_) => println!("Random seed set to {}", config.seed),
        Err(_) => println!("Warning: This backend doesn't support manual seed setting. Random results may vary.")
    }
    
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
    let mut last_train_loss = 0.0;
    let mut last_valid_loss = 0.0;
    
    // Custom callback to monitor loss and calculate perplexity
    let callback = |event: burn::train::LearnerEvent<_, _, _>| {
        match event {
            burn::train::LearnerEvent::TrainEnd(result) => {
                if let Some(metrics) = result.metrics.numeric.get("loss") {
                    if !metrics.is_empty() {
                        last_train_loss = metrics.last().unwrap().value;
                    }
                }
            },
            burn::train::LearnerEvent::ValidEnd(result) => {
                if let Some(metrics) = result.metrics.numeric.get("loss") {
                    if !metrics.is_empty() {
                        last_valid_loss = metrics.last().unwrap().value;
                        // Print perplexity after each validation
                        let train_ppl = (last_train_loss as f64).exp();
                        let valid_ppl = (last_valid_loss as f64).exp();
                        println!("Train Loss: {:.6} (PPL: {:.2}), Valid Loss: {:.6} (PPL: {:.2})", 
                                 last_train_loss, train_ppl, last_valid_loss, valid_ppl);
                    }
                }
            },
            _ => {}
        }
    };
    
    let model_trained = learner.with_callback(callback).fit(train_dataloader, valid_dataloader);
    let duration = start_time.elapsed();
    
    // Final perplexity values
    let train_ppl = (last_train_loss as f64).exp();
    let valid_ppl = (last_valid_loss as f64).exp();
    println!("Training completed in {:.2} seconds", duration.as_secs_f64());
    println!("Final metrics - Train Loss: {:.6} (PPL: {:.2}), Valid Loss: {:.6} (PPL: {:.2})",
             last_train_loss, train_ppl, last_valid_loss, valid_ppl);
    
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
        1024           // dimension (increased from 512 to 1024)
    )
    .with_depth(3)     // testing with 3 layers
    .with_ff_mult(3.0) // keeping ff_mult at 3.0
    .with_expansion_factor(1.5) // keeping expansion_factor at 1.5
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
