use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{backend::Backend, Tensor, Int},
    train::{LearnerBuilder, metric::LossMetric},
    backend::wgpu::{Wgpu, WgpuDevice},
    backend::autodiff::Autodiff,
};
use mingru::{
    model::MinGRULMConfig,
    dataset::{CharVocab, TextDataset, TextBatcher},
    Module,
};
use burn::data::dataset::Dataset;
use burn::module::AutodiffModule;
use std::{
    fs,
    time::Instant,
};

type WgpuBackend = Wgpu<f32, i32>;
type MyBackend = Autodiff<WgpuBackend>;

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
    
    model: MinGRULMConfig,
    optimizer: AdamConfig,
}

fn main() {
    // Parse command-line arguments
    let args: Vec<String> = std::env::args().collect();
    
    // Default values
    let mut data_path = "data/sample.txt".to_string();
    let mut artifact_dir = "mingru_artifacts".to_string();
    
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
            _ => {}
        }
    }
    
    // Use the GPU-capable backend
    let device = WgpuDevice::default();
    
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
    let model_config = MinGRULMConfig::new(
        vocab.size(),  // num_tokens
        256            // dimension
    )
    .with_ff_mult(4.0)
    .with_expansion_factor(1.5)
    .with_chunk_size(256);
    
    let optimizer_config = AdamConfig::new();
    
    let config = TrainingConfig::new(
        model_config,
        optimizer_config,
    )
    .with_sequence_length(128)
    .with_step_size(3)
    .with_batch_size(32)
    .with_num_epochs(10)
    .with_learning_rate(1e-3);
    
    // Save config for reproducibility
    config.save(format!("{}/config.json", artifact_dir))
        .expect("Failed to save config");
    
    println!("Training with sequence length: {}, step size: {}", 
             config.sequence_length, config.step_size);
    println!("Vocabulary size: {}", vocab.size());
    
    // Set up the dataset
    let dataset = TextDataset::new(
        text.clone(),
        config.sequence_length,
        config.step_size,
        config.chunk_size,
    );
    
    println!("Dataset size: {} sequences", dataset.len());
    
    // Create training and validation splits (80/20)
    let dataset_len = dataset.len();
    let train_len = (dataset_len as f64 * 0.8) as usize;
    
    let train_dataset = TextDataset::new(
        text.clone(),
        config.sequence_length,
        config.step_size,
        config.chunk_size,
    );
    
    let valid_dataset = TextDataset::new(
        text,
        config.sequence_length,
        config.step_size,
        config.chunk_size,
    );
    
    // Set up batchers
    let train_batcher = TextBatcher::<MyBackend>::new(vocab.clone(), device.clone());
    let valid_batcher = TextBatcher::<WgpuBackend>::new(vocab.clone(), device.clone());
    
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
    
    // Initialize learner
    MyBackend::seed(config.seed);
    
    let learner = LearnerBuilder::new(&artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .build(
            config.model.init::<MyBackend>(&device),
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
    
    println!("Training complete! Model saved to {}", model_path);
    
    // Generate a sample using the trained model
    let seed = "Hello";
    let n_chars = 100;
    println!("\nGenerating sample text with seed: '{}'", seed);
    
    let seed_tokens: Vec<i64> = seed.chars()
        .filter_map(|c| vocab.char_to_index(c).map(|idx| idx as i64))
        .collect();
    
    if !seed_tokens.is_empty() {
        let seed_tensor = Tensor::<WgpuBackend, 1, Int>::from_data(&*seed_tokens, &device).unsqueeze::<2>();
        
        // Generate text using the trained model
        let model_valid = model_trained.valid();
        let (generated_tokens, _) = model_valid.generate(seed_tensor, n_chars, 0.8, None);
        
        // Convert token IDs back to text
        let ids: Vec<usize> = generated_tokens
            .reshape([generated_tokens.dims()[0] * generated_tokens.dims()[1]])
            .to_data()
            .as_slice()
            .iter()
            .map(|x| *x as usize)
            .collect();
        
        let generated_text = vocab.decode_text(&ids);
        println!("Generated sample:\n{}", generated_text);
    } else {
        println!("Failed to tokenize seed text");
    }
}
