use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    backend::autodiff::Autodiff,
    config::Config,
    optim::AdamConfig,
};

use grufinity::{
    model::MinGRULMConfig,
    dataset::CharVocab,
    tbptt::{TBPTTConfig, train_with_tbptt},
    Module,
};

use std::fs;

type WgpuBackend = Wgpu<f32, i32>;
type MyBackend = Autodiff<WgpuBackend>;

fn main() {
    // Parse command-line arguments
    let args: Vec<String> = std::env::args().collect();
    
    // Default values
    let mut data_path = "data/sample.txt".to_string();
    let mut artifact_dir = "tbptt_artifacts".to_string();
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
            "--target-valid-loss" => {
                if i + 1 < args.len() {
                    if let Ok(loss) = args[i + 1].parse::<f32>() {
                        // We'll create a modified config with this value
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.target_valid_loss = loss;
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--target-test-loss" => {
                if i + 1 < args.len() {
                    if let Ok(loss) = args[i + 1].parse::<f32>() {
                        // We'll create a modified config with this value
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.target_test_loss = loss;
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--max-epochs" => {
                if i + 1 < args.len() {
                    if let Ok(epochs) = args[i + 1].parse::<usize>() {
                        // We'll create a modified config with this value
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.max_epochs = epochs;
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--learning-rate" => {
                if i + 1 < args.len() {
                    if let Ok(lr) = args[i + 1].parse::<f64>() {
                        // We'll create a modified config with this value
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.learning_rate = lr;
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                        println!("Set learning rate to {}", lr);
                    }
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
    
    // Create or load configuration
    let config = if !config_path.is_empty() {
        match TBPTTConfig::load(&config_path) {
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
    config.save(format!("{}/tbptt_config.json", artifact_dir))
        .expect("Failed to save config");
    
    println!("Training with TBPTT - chunk size: {}, k1: {}, k2: {}", 
             config.chunk_size, config.tbptt_k1, config.tbptt_k2);
    println!("Learning rate: {}", config.learning_rate);
    println!("Vocabulary size: {}", vocab.size());
    
    // Train the model using TBPTT with Learner API
    println!("Training with TBPTT using Learner API");
    let model = train_with_tbptt::<MyBackend>(
        &config,
        &device,
        &text,
        &vocab,
        &artifact_dir
    );
    
    // Save the final model
    let model_path = format!("{}/model_final.bin", artifact_dir);
    let recorder = burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::new();
    model.save_file(model_path.clone(), &recorder)
        .expect("Failed to save final model");
    
    println!("\nTraining complete! Final model saved to {}", model_path);
    println!("To generate text with the trained model, use:");
    println!("cargo run --release --bin generate -- --model {} --vocab {}/vocab.txt", 
             model_path, artifact_dir);
}

fn create_default_config() -> TBPTTConfig {
    // Configure the model
    let model_config = MinGRULMConfig::new(
        256,           // num_tokens (all possible byte values)
        96             // dimension
    )
    .with_depth(2)
    .with_ff_mult(2.0)
    .with_expansion_factor(1.2)
    .with_chunk_size(64);  // Chunk size for TBPTT
    
    let optimizer_config = AdamConfig::new();
    
    TBPTTConfig::new(
        model_config,
        optimizer_config,
    )
    .with_chunk_size(64)
    .with_tbptt_k1(4)                // Update frequency
    .with_tbptt_k2(8)                // Backprop window length 
    .with_max_chunks_per_epoch(1000) // Process 1000 chunks per epoch
    .with_batch_size(32)
    .with_num_epochs(10)
    .with_learning_rate(1e-3)
    .with_preserve_hidden_states(true)
    .with_target_valid_loss(0.0)  // 0.0 means ignore
    .with_target_test_loss(0.0)   // 0.0 means ignore
    .with_max_epochs(1000)        // Maximum epochs if target not reached
}
