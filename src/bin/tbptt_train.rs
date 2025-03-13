use burn::{
    config::Config,
};

#[cfg(feature = "optimizer-sgd")]
use burn::{
    optim::SgdConfig,
    optim::momentum::MomentumConfig,
    optim::decay::WeightDecayConfig,
};

#[cfg(feature = "optimizer-adam")]
use burn::optim::AdamConfig;

use grufinity::{
    model::MinGRULMConfig,
    dataset::CharVocab,
    tbptt::{TBPTTConfig, train_with_tbptt, LRSchedulerType},
    Module,
    use_configured_backend,
    BackendWithAutodiff,
};

use std::fs;

fn print_help() {
    println!("GRUfinity TBPTT Training");
    println!("========================");
    println!("Usage: cargo run --release --bin tbptt_train -- [OPTIONS]");
    println!("\nOptions:");
    println!("  --data PATH                    Path to training data file");
    println!("  --output DIR                   Directory for output artifacts");
    println!("  --config PATH                  Path to configuration file");
    println!("  --learning-rate RATE           Set learning rate (default: 0.001)");
    println!("  --batch-size SIZE              Number of random start positions (default: 32)");
    println!("  --chunk-size SIZE              Characters per chunk (default: 64)");
    println!("  --max-chunks-per-epoch NUM     Chunks to process per epoch (default: 1000)");
    println!("  --context-length LENGTH        Set context length in characters");
    println!("  --num-epochs NUM               Number of training epochs (default: 10)");
    println!("  --max-epochs NUM               Maximum number of epochs if using target loss (default: 1000)");
    println!("  --target-valid-loss VALUE      Target validation loss to stop at (0.0 to ignore)");
    println!("  --target-test-loss VALUE       Target test loss to stop at (0.0 to ignore)");
    println!("  --update-chunks NUM            Update parameters every NUM chunks (k1 parameter) (default: 4)");
    println!("  --backprop-chunks NUM          Backprop through NUM chunks (k2 parameter) (default: 8)");
    println!("  --update-tokens NUM            Update parameters every ~NUM tokens (converted to chunks)");
    println!("  --backprop-tokens NUM          Backprop through ~NUM tokens (converted to chunks)");
    println!("  --preserve-hidden-states BOOL  Preserve hidden states between batches (default: true)");
    println!("  --grad-clip VALUE              Gradient clipping value (0.0 to disable)");
    println!("  --log-interval NUM             Log interval in batches (default: 10)");
    println!("  --checkpoint-interval NUM      Checkpoint interval in epochs (default: 1)");
    println!("  --momentum VALUE               Momentum factor for SGD (default: 0.0)");
    println!("  --weight-decay VALUE           Weight decay (L2 penalty) for SGD (default: 0.0)");
    println!("  --dampening VALUE              Dampening for momentum (default: 0.0)");
    println!("  --nesterov BOOL                Enable Nesterov momentum (default: false)");
    println!("  --lr-scheduler TYPE            Learning rate scheduler (constant, cosine, linear) (default: constant)");
    println!("  --min-lr-factor FACTOR         Minimum learning rate as a factor of initial lr (default: 0.1)");
    println!("  --warmup-epochs NUM            Number of warmup epochs (default: 0)");
    println!("\nExample:");
    println!("  cargo run --release --bin tbptt_train -- --data input.txt --batch-size 64 --chunk-size 128 --context-length 100000 --update-tokens 512 --backprop-tokens 1024");
    println!("\nThis will train with:");
    println!("  - 64 parallel sequences");
    println!("  - 128 characters per chunk");
    println!("  - Context length of 100,000 characters");
    println!("  - Update parameters every ~512 tokens");
    println!("  - Backpropagate through ~1024 tokens");
}

fn main() {
    // Parse command-line arguments
    let args: Vec<String> = std::env::args().collect();
    
    // Check for help
    if args.len() > 1 && (args[1] == "--help" || args[1] == "-h") {
        print_help();
        return;
    }
    
    // Default values
    let mut data_path = "data/sample.txt".to_string();
    let mut artifact_dir = "tbptt_artifacts".to_string();
    let mut config_path = "".to_string();
    
    // Token-based parameters (will be converted to chunks)
    let mut update_tokens: Option<usize> = None;
    let mut backprop_tokens: Option<usize> = None;
    
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
            "--update-tokens" => {
                if i + 1 < args.len() {
                    if let Ok(tokens) = args[i + 1].parse::<usize>() {
                        update_tokens = Some(tokens);
                    }
                }
            },
            "--backprop-tokens" => {
                if i + 1 < args.len() {
                    if let Ok(tokens) = args[i + 1].parse::<usize>() {
                        backprop_tokens = Some(tokens);
                    }
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
            "--update-chunks" => {
                if i + 1 < args.len() {
                    if let Ok(chunks) = args[i + 1].parse::<usize>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.tbptt_k1 = chunks;
                        println!("Setting update frequency to every {} chunks", chunks);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--backprop-chunks" => {
                if i + 1 < args.len() {
                    if let Ok(chunks) = args[i + 1].parse::<usize>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.tbptt_k2 = chunks;
                        println!("Setting backpropagation window to {} chunks", chunks);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--preserve-hidden-states" => {
                if i + 1 < args.len() {
                    if let Ok(preserve) = args[i + 1].parse::<bool>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.preserve_hidden_states = preserve;
                        println!("Setting preserve hidden states to: {}", preserve);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--grad-clip" => {
                if i + 1 < args.len() {
                    if let Ok(clip) = args[i + 1].parse::<f32>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.grad_clip = clip;
                        println!("Setting gradient clipping to: {}", clip);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--log-interval" => {
                if i + 1 < args.len() {
                    if let Ok(interval) = args[i + 1].parse::<usize>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.log_interval = interval;
                        println!("Setting log interval to: {} batches", interval);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--checkpoint-interval" => {
                if i + 1 < args.len() {
                    if let Ok(interval) = args[i + 1].parse::<usize>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.checkpoint_interval = interval;
                        println!("Setting checkpoint interval to: {} epochs", interval);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--momentum" => {
                if i + 1 < args.len() {
                    if let Ok(momentum) = args[i + 1].parse::<f64>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        #[cfg(feature = "optimizer-sgd")]
                        {
                            // Create a fresh SgdConfig with our custom momentum
                            let momentum_config = MomentumConfig {
                                momentum,
                                dampening: 0.0,
                                nesterov: false,
                            };
                            modified_config.optimizer = SgdConfig::new().with_momentum(Some(momentum_config));
                        }
                        
                        #[cfg(feature = "optimizer-adam")]
                        {
                            println!("Note: Momentum parameter is only applicable when using SGD optimizer (--features=\"optimizer-sgd\")");
                            // Keep the default Adam config
                            modified_config.optimizer = AdamConfig::new();
                        }
                        println!("Setting momentum to: {}", momentum);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--weight-decay" => {
                if i + 1 < args.len() {
                    if let Ok(decay) = args[i + 1].parse::<f64>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        #[cfg(feature = "optimizer-sgd")]
                        {
                            // Create a fresh SgdConfig with our custom weight decay
                            let weight_decay_config = WeightDecayConfig { penalty: decay as f32 };
                            modified_config.optimizer = SgdConfig::new().with_weight_decay(Some(weight_decay_config));
                        }
                        
                        #[cfg(feature = "optimizer-adam")]
                        {
                            println!("Note: Weight decay parameter is only applicable when using SGD optimizer (--features=\"optimizer-sgd\")");
                            // Keep the default Adam config
                            modified_config.optimizer = AdamConfig::new();
                        }
                        println!("Setting weight decay to: {}", decay);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--dampening" => {
                if i + 1 < args.len() {
                    if let Ok(dampening) = args[i + 1].parse::<f64>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        #[cfg(feature = "optimizer-sgd")]
                        {
                            // Create a fresh momentum config with dampening
                            let momentum_config = MomentumConfig {
                                momentum: 0.9, // Default momentum
                                dampening,
                                nesterov: false,
                            };
                            modified_config.optimizer = SgdConfig::new().with_momentum(Some(momentum_config));
                        }
                        
                        #[cfg(feature = "optimizer-adam")]
                        {
                            println!("Note: Dampening parameter is only applicable when using SGD optimizer (--features=\"optimizer-sgd\")");
                            // Keep the default Adam config
                            modified_config.optimizer = AdamConfig::new();
                        }
                        println!("Setting dampening to: {}", dampening);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--nesterov" => {
                if i + 1 < args.len() {
                    if let Ok(nesterov) = args[i + 1].parse::<bool>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        #[cfg(feature = "optimizer-sgd")]
                        {
                            // Create a fresh momentum config with nesterov
                            let momentum_config = MomentumConfig {
                                momentum: 0.9, // Default momentum
                                dampening: 0.0,
                                nesterov,
                            };
                            modified_config.optimizer = SgdConfig::new().with_momentum(Some(momentum_config));
                        }
                        
                        #[cfg(feature = "optimizer-adam")]
                        {
                            println!("Note: Nesterov parameter is only applicable when using SGD optimizer (--features=\"optimizer-sgd\")");
                            // Keep the default Adam config
                            modified_config.optimizer = AdamConfig::new();
                        }
                        println!("Setting nesterov to: {}", nesterov);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--lr-scheduler" => {
                if i + 1 < args.len() {
                    let scheduler_type = args[i + 1].clone();
                    let mut modified_config = create_default_config();
                    if !config_path.is_empty() {
                        if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                            modified_config = cfg;
                        }
                    }
                    
                    // Parse the scheduler type string to the enum
                    let scheduler = match scheduler_type.to_lowercase().as_str() {
                        "cosine" => LRSchedulerType::Cosine,
                        "linear" => LRSchedulerType::Linear,
                        _ => LRSchedulerType::Constant, // Default to constant for any other value
                    };
                    
                    modified_config.lr_scheduler = scheduler;
                    println!("Setting learning rate scheduler to: {:?}", scheduler);
                    modified_config.save("temp_config.json").expect("Failed to save temporary config");
                    config_path = "temp_config.json".to_string();
                }
            },
            "--min-lr-factor" => {
                if i + 1 < args.len() {
                    if let Ok(factor) = args[i + 1].parse::<f64>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.min_lr_factor = factor;
                        println!("Setting minimum learning rate factor to: {}", factor);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--warmup-epochs" => {
                if i + 1 < args.len() {
                    if let Ok(epochs) = args[i + 1].parse::<usize>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.warmup_epochs = epochs;
                        println!("Setting warmup epochs to: {}", epochs);
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
                        println!("Setting target test loss to: {}", loss);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--num-epochs" => {
                if i + 1 < args.len() {
                    if let Ok(epochs) = args[i + 1].parse::<usize>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.num_epochs = epochs;
                        println!("Setting number of epochs to: {}", epochs);
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
            "--max-chunks-per-epoch" => {
                if i + 1 < args.len() {
                    if let Ok(chunks) = args[i + 1].parse::<usize>() {
                        // We'll create a modified config with this value
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.max_chunks_per_epoch = chunks;
                        println!("Setting max chunks per epoch to {}", chunks);
                        println!("Effective context length: {} characters", chunks * modified_config.chunk_size);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--batch-size" => {
                if i + 1 < args.len() {
                    if let Ok(batch_size) = args[i + 1].parse::<usize>() {
                        // We'll create a modified config with this value
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.batch_size = batch_size;
                        println!("Setting batch size to {} random start positions", batch_size);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--chunk-size" => {
                if i + 1 < args.len() {
                    if let Ok(chunk_size) = args[i + 1].parse::<usize>() {
                        // We'll create a modified config with this value
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.chunk_size = chunk_size;
                        // Also update the model's chunk size
                        modified_config.model = modified_config.model.with_chunk_size(chunk_size);
                        println!("Setting chunk size to {} characters", chunk_size);
                        println!("Effective context length: {} characters", 
                                modified_config.max_chunks_per_epoch * chunk_size);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--context-length" => {
                if i + 1 < args.len() {
                    if let Ok(context_length) = args[i + 1].parse::<usize>() {
                        // We'll create a modified config with this value
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        let chunk_size = modified_config.chunk_size;
                        let chunks_needed = calculate_chunks_for_context(chunk_size, context_length);
                        modified_config.max_chunks_per_epoch = chunks_needed;
                        println!("Setting context length to {} characters", context_length);
                        println!("Using {} chunks with chunk size {}", chunks_needed, chunk_size);
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
    
    // Set up the configured backend
    use_configured_backend!();
    
    // Get the device from the appropriate backend
    #[allow(unused_assignments)]
    let device;
    let mut device_initialized = false;
    
    #[cfg(all(feature = "cuda-jit", not(feature = "wgpu"), not(feature = "candle"), not(feature = "tch"), not(feature = "ndarray")))]
    {
        use burn::backend::cuda_jit::CudaDevice;
        device = CudaDevice::new(0); // Use first CUDA device with JIT
        device_initialized = true;
        println!("Using CUDA JIT device");
    }
    
    #[cfg(all(feature = "candle", feature = "candle-cuda", not(feature = "cuda-jit"), not(feature = "wgpu")))]
    {
        use burn::backend::candle::CandleDevice;
        device = CandleDevice::Cuda(0);  // Use first CUDA device via Candle
        println!("Using Candle CUDA device");
    }
    
    #[cfg(all(feature = "candle-metal", not(feature = "cuda-jit"), not(feature = "wgpu"), not(all(feature = "candle", feature = "candle-cuda"))))]
    {
        use burn::backend::candle::CandleDevice;
        device = CandleDevice::Metal(0);  // Use first Metal device
        println!("Using Candle Metal device");
    }
    
    #[cfg(all(feature = "wgpu", not(feature = "cuda-jit"), not(all(feature = "candle", feature = "candle-cuda"))))]
    {
        use burn::backend::wgpu::WgpuDevice;
        device = WgpuDevice::default();
    }
    
    #[cfg(all(feature = "candle", not(any(feature = "cuda-jit", feature = "wgpu"))))]
    {
        use burn::backend::candle::CandleDevice;
        device = CandleDevice::Cpu;
    }
    
    #[cfg(all(feature = "ndarray", not(any(feature = "cuda-jit", feature = "wgpu", feature = "candle"))))]
    {
        use burn::backend::ndarray::NdArrayDevice;
        device = NdArrayDevice;
    }
    
    #[cfg(all(feature = "tch", not(any(feature = "cuda-jit", feature = "wgpu", feature = "candle", feature = "ndarray"))))]
    {
        use burn::backend::libtorch::LibTorchDevice;
        device = LibTorchDevice::Cpu;
    }
    
    // Fallback to ensure device is always initialized
    // This will only run if none of the above cfg blocks matched
    if !device_initialized {
        #[cfg(feature = "cuda-jit")]
        {
            use burn::backend::cuda_jit::CudaDevice;
            device = CudaDevice::new(0);
            device_initialized = true;
            println!("Using CUDA JIT device (fallback)");
        }
        
        #[cfg(all(not(feature = "cuda-jit"), feature = "wgpu"))]
        {
            use burn::backend::wgpu::WgpuDevice;
            device = WgpuDevice::default();
            device_initialized = true;
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
            device_initialized = true;
            println!("Using NdArray device (fallback)");
        }
    }

    // Add a final fallback in case no backend feature is enabled
    if !device_initialized {
        // We need to pick one default backend that will be in the binary to satisfy the compiler
        #[cfg(feature = "ndarray")]
        {
            use burn::backend::ndarray::NdArrayDevice;
            device = NdArrayDevice;
            println!("WARNING: Using NdArray device as last resort fallback");
            println!("No backend feature was enabled - please enable at least one backend feature");
        }
        
        #[cfg(not(any(feature = "ndarray", feature = "cuda-jit", feature = "wgpu", feature = "candle", feature = "tch")))]
        {
            // This is a compile-time error that will be triggered if no backend is enabled
            compile_error!("No backend feature was enabled. Please enable at least one: ndarray, wgpu, candle, etc.");
        }
    }
    
    // Create a directory for artifacts
    fs::create_dir_all(&artifact_dir).expect("Failed to create artifact directory");
    
    // Load training data
    let text = fs::read_to_string(&data_path).unwrap_or_else(|_| {
        println!("Could not read file {}, using sample text", data_path);
        "Hello world! This is a sample text for the MinGRU model.".to_string()
    });
    
    // Process token-based TBPTT parameters
    let mut modified_config = if !config_path.is_empty() {
        match TBPTTConfig::load(&config_path) {
            Ok(cfg) => cfg,
            Err(_) => create_default_config()
        }
    } else {
        create_default_config()
    };
    
    // Handle token-based parameters
    let chunk_size = modified_config.chunk_size;
    
    if let Some(tokens) = update_tokens {
        let k1 = calculate_chunks_for_tokens(chunk_size, tokens);
        modified_config.tbptt_k1 = k1;
        println!("Setting update frequency to every {} tokens (~{} chunks)", tokens, k1);
    }
    
    if let Some(tokens) = backprop_tokens {
        let k2 = calculate_chunks_for_tokens(chunk_size, tokens);
        modified_config.tbptt_k2 = k2;
        println!("Setting backpropagation window to {} tokens (~{} chunks)", tokens, k2);
    }
    
    // Save modified config for use
    if update_tokens.is_some() || backprop_tokens.is_some() {
        modified_config.save("temp_config.json").expect("Failed to save temporary config");
        config_path = "temp_config.json".to_string();
    }
    
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
    let model = train_with_tbptt::<BackendWithAutodiff>(
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

/// Calculate the number of chunks needed for a desired context length
fn calculate_chunks_for_context(chunk_size: usize, desired_context_length: usize) -> usize {
    (desired_context_length + chunk_size - 1) / chunk_size
}

/// Calculate the number of chunks needed for a desired token length
fn calculate_chunks_for_tokens(chunk_size: usize, desired_token_length: usize) -> usize {
    let chunks = (desired_token_length + chunk_size - 1) / chunk_size;
    // Ensure at least 1 chunk
    chunks.max(1)
}

fn create_default_config() -> TBPTTConfig {
    // Chunk size for processing text
    let chunk_size = 64;
    
    // Configure the model
    let model_config = MinGRULMConfig::new(
        256,           // num_tokens (all possible byte values)
        512            // dimension (increased from 128 to 512)
    )
    .with_depth(6)     // increased from 4 to 6
    .with_ff_mult(3.0) // keeping ff_mult at 3.0
    .with_expansion_factor(1.5) // keeping expansion_factor at 1.5
    .with_chunk_size(chunk_size);
    
    #[cfg(feature = "optimizer-sgd")]
    let optimizer_config = SgdConfig::new();
    
    #[cfg(feature = "optimizer-adam")]
    let optimizer_config = AdamConfig::new();
    
    // Calculate chunks for different context lengths
    let desired_context = 64000; // Desired context length in characters
    let chunks_needed = calculate_chunks_for_context(chunk_size, desired_context);
    println!("Default config using {} chunks for ~{} character context", chunks_needed, desired_context);
    
    TBPTTConfig::new(
        model_config,
        optimizer_config,
    )
    .with_chunk_size(chunk_size)
    .with_tbptt_k1(4)                // Update frequency (every 4 chunks)
    .with_tbptt_k2(8)                // Backprop window (8 chunks = 512 characters)
    .with_max_chunks_per_epoch(1000) // Process 1000 chunks per epoch (64K character context)
    .with_batch_size(32)
    .with_num_epochs(10)
    .with_learning_rate(1e-3)
    .with_preserve_hidden_states(true)
    .with_target_valid_loss(0.0)  // 0.0 means ignore
    .with_target_test_loss(0.0)   // 0.0 means ignore
    .with_max_epochs(1000)        // Maximum epochs if target not reached
    .with_lr_scheduler(LRSchedulerType::Constant) // Constant learning rate by default
    .with_min_lr_factor(0.1)       // Minimum LR at 10% of max
    .with_warmup_epochs(0)         // No warmup by default
}
