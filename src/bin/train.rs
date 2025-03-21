#![recursion_limit = "256"]

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
use burn::optim::{AdamConfig, decay::WeightDecayConfig};

use grufinity::{
    model::MinGRULMConfig,
    dataset::CharVocab,
    tbptt::{TBPTTConfig, train_with_tbptt, LRSchedulerType},
    Module,
    use_configured_backend,
    BackendWithAutodiff,
    BackendDevice,
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
    println!("                                 Suffixes k/m/g supported (e.g., 64 or 64k)");
    println!("  --chunk-size SIZE              Characters per chunk (default: 64)");
    println!("                                 Suffixes k/m/g supported (e.g., 128 or 1k)");
    println!("  --max-chunks-per-epoch NUM     Chunks to process per epoch (default: 1000)");
    println!("                                 Suffixes k/m/g supported (e.g., 2k or 10k)");
    println!("  --context-length LENGTH        Set context length in characters");
    println!("                                 Suffixes k/m/g supported (e.g., 16k or 1m)");
    println!("  --num-epochs NUM               Number of training epochs (default: 10)");
    println!("  --max-epochs NUM               Maximum number of epochs if using target loss (default: 1000)");
    println!("  --target-valid-loss VALUE      Target validation loss to stop at (0.0 to ignore)");
    println!("  --target-test-loss VALUE       Target test loss to stop at (0.0 to ignore)");
    println!("  --update-chunks NUM            Update parameters every NUM chunks (k1 parameter) (default: 4)");
    println!("  --backprop-chunks NUM          Backprop through NUM chunks (k2 parameter) (default: 8)");
    println!("  --update-tokens NUM            Update parameters every ~NUM tokens (converted to chunks)");
    println!("                                 Suffixes k/m/g supported (e.g., 8k or 16k)");
    println!("  --backprop-tokens NUM          Backprop through ~NUM tokens (converted to chunks)");
    println!("                                 Suffixes k/m/g supported (e.g., 16k or 32k)");
    println!("  --preserve-hidden-states BOOL  Preserve hidden states between batches (default: true)");
    println!("  --grad-clip VALUE              Gradient clipping value (0.0 to disable)");
    println!("  --log-interval NUM             Log interval in batches (default: 10)");
    println!("  --checkpoint-interval NUM      Checkpoint interval in epochs (default: 1)");
    println!("  --momentum VALUE               Momentum factor for SGD (default: 0.0)");
    println!("  --weight-decay VALUE           Weight decay (L2 penalty) (default: 0.0)");
    println!("  --dampening VALUE              Dampening for momentum (default: 0.0)");
    println!("  --nesterov BOOL                Enable Nesterov momentum (default: false)");
    println!("  --beta1 VALUE                  Beta1 parameter for Adam (default: 0.9)");
    println!("  --beta2 VALUE                  Beta2 parameter for Adam (default: 0.999)");
    println!("  --epsilon VALUE                Epsilon parameter for Adam (default: 1e-5)");
    println!("  --lr-scheduler TYPE            Learning rate scheduler (constant, cosine, linear) (default: constant)");
    println!("  --min-lr-factor FACTOR         Minimum learning rate as a factor of initial lr (default: 0.1)");
    println!("  --warmup-epochs NUM            Number of warmup epochs (default: 0)");
    println!("  --lr-reduce-threshold VALUE    Threshold for reducing LR on plateau (default: 0.001, 0 to disable)");
    println!("  --lr-reduce-factor VALUE       Factor to reduce LR by on plateau (default: 0.1)");
    println!("  --stall-epochs NUM            Epochs with low improvement before increasing LR (default: 2)");
    println!("  --stall-threshold VALUE       Improvement % below which an epoch is considered stalled (default: 0.01)");
    println!("  --device-id ID                 CUDA/GPU device ID to use (default: 0)");
    println!("\nModel Structure Options:");
    println!("  --model-dim SIZE               Model hidden dimension (default: 1024)");
    println!("  --model-depth NUM              Number of MinGRU layers (default: 3)");
    println!("  --model-ff-mult FACTOR         Feed-forward multiplier (default: 3.0)");
    println!("  --model-exp-factor FACTOR      Expansion factor (default: 1.5)");
    println!("\nExample:");
    println!("  cargo run --release --bin tbptt_train -- --data input.txt --batch-size 64 --chunk-size 128 --context-length 100000 --update-tokens 512 --backprop-tokens 1024");
    println!("\nThis will train with:");
    println!("  - 64 parallel sequences");
    println!("  - 128 characters per chunk");
    println!("  - Context length of 100,000 characters");
    println!("  - Update parameters every ~512 tokens");
    println!("  - Backpropagate through ~1024 tokens");
}

/// Parse a string with optional metric suffix (k, m, g) into a number
/// Examples: "1k" -> 1024, "2m" -> 2097152, "1.5g" -> 1610612736
fn parse_with_suffix<T>(s: &str) -> Result<T, String> 
where 
    T: std::str::FromStr + 'static,
    <T as std::str::FromStr>::Err: std::fmt::Display
{
    // Check if the string ends with a known suffix
    let lower_s = s.to_lowercase();
    let (value_str, multiplier) = if lower_s.ends_with('k') {
        (&s[..s.len()-1], 1024.0)
    } else if lower_s.ends_with('m') {
        (&s[..s.len()-1], 1024.0 * 1024.0)
    } else if lower_s.ends_with('g') {
        (&s[..s.len()-1], 1024.0 * 1024.0 * 1024.0)
    } else {
        (s, 1.0)
    };
    
    // Parse the numeric part to f64 first
    match value_str.parse::<f64>() {
        Ok(num) => {
            let result = num * multiplier;
            
            // Handle specific types without using unsafe code
            let type_id = std::any::TypeId::of::<T>();
            
            if type_id == std::any::TypeId::of::<usize>() {
                // Safe conversion for usize
                let val = result as usize;
                // This transmute is safe because we've verified T is usize
                Ok(unsafe { std::mem::transmute_copy(&val) })
            } else if type_id == std::any::TypeId::of::<f64>() {
                // Safe conversion for f64
                // This transmute is safe because we've verified T is f64
                Ok(unsafe { std::mem::transmute_copy(&result) })
            } else if type_id == std::any::TypeId::of::<f32>() {
                // Handle f32 case
                let val = result as f32;
                Ok(unsafe { std::mem::transmute_copy(&val) })
            } else if type_id == std::any::TypeId::of::<i32>() {
                // Handle i32 case
                let val = result as i32;
                Ok(unsafe { std::mem::transmute_copy(&val) })
            } else {
                // For other types, try parsing directly
                s.parse::<T>().map_err(|e| format!("Failed to parse '{}': {}", s, e))
            }
        },
        Err(e) => Err(format!("Failed to parse '{}': {}", s, e))
    }
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
    let mut device_id: usize = 0; // Default to first GPU device
    
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
                    if let Ok(tokens) = parse_with_suffix::<usize>(&args[i + 1]) {
                        update_tokens = Some(tokens);
                    }
                }
            },
            "--backprop-tokens" => {
                if i + 1 < args.len() {
                    if let Ok(tokens) = parse_with_suffix::<usize>(&args[i + 1]) {
                        backprop_tokens = Some(tokens);
                    }
                }
            },
            "--model-dim" => {
                if i + 1 < args.len() {
                    if let Ok(dim) = parse_with_suffix::<usize>(&args[i + 1]) {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        // Update model dimension in the config
                        modified_config.model = MinGRULMConfig::new(
                            modified_config.model.num_tokens(),
                            dim
                        )
                        .with_depth(modified_config.model.depth())
                        .with_ff_mult(modified_config.model.ff_mult())
                        .with_expansion_factor(modified_config.model.expansion_factor())
                        .with_chunk_size(modified_config.model.chunk_size());
                        
                        println!("Setting model dimension to: {}", dim);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--model-depth" => {
                if i + 1 < args.len() {
                    if let Ok(depth) = args[i + 1].parse::<usize>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        // Update model depth in the config
                        modified_config.model = MinGRULMConfig::new(
                            modified_config.model.num_tokens(),
                            modified_config.model.dim()
                        )
                        .with_depth(depth)
                        .with_ff_mult(modified_config.model.ff_mult())
                        .with_expansion_factor(modified_config.model.expansion_factor())
                        .with_chunk_size(modified_config.model.chunk_size());
                        
                        println!("Setting model depth to: {} layers", depth);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--model-ff-mult" => {
                if i + 1 < args.len() {
                    if let Ok(mult) = args[i + 1].parse::<f64>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        // Update feed-forward multiplier in the config
                        modified_config.model = MinGRULMConfig::new(
                            modified_config.model.num_tokens(),
                            modified_config.model.dim()
                        )
                        .with_depth(modified_config.model.depth())
                        .with_ff_mult(mult)
                        .with_expansion_factor(modified_config.model.expansion_factor())
                        .with_chunk_size(modified_config.model.chunk_size());
                        
                        println!("Setting feed-forward multiplier to: {}", mult);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--model-exp-factor" => {
                if i + 1 < args.len() {
                    if let Ok(factor) = args[i + 1].parse::<f64>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        // Update expansion factor in the config
                        modified_config.model = MinGRULMConfig::new(
                            modified_config.model.num_tokens(),
                            modified_config.model.dim()
                        )
                        .with_depth(modified_config.model.depth())
                        .with_ff_mult(modified_config.model.ff_mult())
                        .with_expansion_factor(factor)
                        .with_chunk_size(modified_config.model.chunk_size());
                        
                        println!("Setting expansion factor to: {}", factor);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
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
                            // For Adam, store the weight decay in our own copy
                            modified_config.weight_decay = Some(decay as f32);
                            
                            // Update the optimizer config with weight decay and preserve existing Adam parameters
                            modified_config.optimizer = AdamConfig::new()
                                .with_beta_1(modified_config.adam_beta1)
                                .with_beta_2(modified_config.adam_beta2)
                                .with_epsilon(modified_config.adam_epsilon)
                                .with_weight_decay(Some(WeightDecayConfig::new(decay as f32)));
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
            "--beta1" => {
                if i + 1 < args.len() {
                    if let Ok(beta1) = args[i + 1].parse::<f64>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        #[cfg(feature = "optimizer-adam")]
                        {
                            // Update beta1 parameter for Adam
                            // Store our own copy of beta1
                            modified_config.adam_beta1 = beta1 as f32;
                            
                            // Use default values for other parameters if not set
                            if !config_path.is_empty() && modified_config.adam_beta2 == 0.0 {
                                modified_config.adam_beta2 = 0.999; // Default beta2
                            }
                            if !config_path.is_empty() && modified_config.adam_epsilon == 0.0 {
                                modified_config.adam_epsilon = 1e-8; // Default epsilon
                            }

                            // Update the optimizer config
                            modified_config.optimizer = AdamConfig::new()
                                .with_beta_1(modified_config.adam_beta1)
                                .with_beta_2(modified_config.adam_beta2)
                                .with_epsilon(modified_config.adam_epsilon);
                            
                            // Keep existing weight decay if any
                            if let Some(penalty) = modified_config.weight_decay {
                                modified_config.optimizer = modified_config.optimizer
                                    .with_weight_decay(Some(WeightDecayConfig::new(penalty)));
                            }
                        }
                        
                        #[cfg(feature = "optimizer-sgd")]
                        {
                            println!("Note: Beta1 parameter is only applicable when using Adam optimizer (--features=\"optimizer-adam\")");
                        }
                        
                        println!("Setting Adam beta1 to: {}", beta1);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--beta2" => {
                if i + 1 < args.len() {
                    if let Ok(beta2) = args[i + 1].parse::<f64>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        #[cfg(feature = "optimizer-adam")]
                        {
                            // Store our own copy of beta2
                            modified_config.adam_beta2 = beta2 as f32;
                            
                            // Use default values or preserve existing settings
                            if !config_path.is_empty() && modified_config.adam_beta1 == 0.0 {
                                modified_config.adam_beta1 = 0.9; // Default beta1
                            }
                            if !config_path.is_empty() && modified_config.adam_epsilon == 0.0 {
                                modified_config.adam_epsilon = 1e-8; // Default epsilon
                            }
                            
                            // Update the optimizer config
                            modified_config.optimizer = AdamConfig::new()
                                .with_beta_1(modified_config.adam_beta1)
                                .with_beta_2(modified_config.adam_beta2)
                                .with_epsilon(modified_config.adam_epsilon);
                            
                            // Keep existing weight decay if any
                            if let Some(penalty) = modified_config.weight_decay {
                                modified_config.optimizer = modified_config.optimizer
                                    .with_weight_decay(Some(WeightDecayConfig::new(penalty)));
                            }
                        }
                        
                        #[cfg(feature = "optimizer-sgd")]
                        {
                            println!("Note: Beta2 parameter is only applicable when using Adam optimizer (--features=\"optimizer-adam\")");
                        }
                        
                        println!("Setting Adam beta2 to: {}", beta2);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--epsilon" => {
                if i + 1 < args.len() {
                    if let Ok(epsilon) = args[i + 1].parse::<f64>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        #[cfg(feature = "optimizer-adam")]
                        {
                            // Store our own copy of epsilon
                            modified_config.adam_epsilon = epsilon as f32;
                            
                            // Use default values or preserve existing settings
                            if !config_path.is_empty() && modified_config.adam_beta1 == 0.0 {
                                modified_config.adam_beta1 = 0.9; // Default beta1
                            }
                            if !config_path.is_empty() && modified_config.adam_beta2 == 0.0 {
                                modified_config.adam_beta2 = 0.999; // Default beta2
                            }
                            
                            // Update the optimizer config
                            modified_config.optimizer = AdamConfig::new()
                                .with_beta_1(modified_config.adam_beta1)
                                .with_beta_2(modified_config.adam_beta2)
                                .with_epsilon(modified_config.adam_epsilon);
                            
                            // Keep existing weight decay if any
                            if let Some(penalty) = modified_config.weight_decay {
                                modified_config.optimizer = modified_config.optimizer
                                    .with_weight_decay(Some(WeightDecayConfig::new(penalty)));
                            }
                        }
                        
                        #[cfg(feature = "optimizer-sgd")]
                        {
                            println!("Note: Epsilon parameter is only applicable when using Adam optimizer (--features=\"optimizer-adam\")");
                        }
                        
                        println!("Setting Adam epsilon to: {}", epsilon);
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
            "--lr-reduce-threshold" => {
                if i + 1 < args.len() {
                    if let Ok(threshold) = args[i + 1].parse::<f64>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.lr_reduce_threshold = threshold;
                        if threshold <= 0.0 {
                            println!("Disabling learning rate reduction on plateau");
                        } else {
                            println!("Setting learning rate reduction threshold to: {:.4}%", threshold * 100.0);
                        }
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--lr-reduce-factor" => {
                if i + 1 < args.len() {
                    if let Ok(factor) = args[i + 1].parse::<f64>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.lr_reduce_factor = factor;
                        println!("Setting learning rate reduction factor to: {}", factor);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--stall-epochs" => {
                if i + 1 < args.len() {
                    if let Ok(epochs) = args[i + 1].parse::<usize>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.stall_epochs = epochs;
                        println!("Setting stall epochs to: {} epochs", epochs);
                        modified_config.save("temp_config.json").expect("Failed to save temporary config");
                        config_path = "temp_config.json".to_string();
                    }
                }
            },
            "--stall-threshold" => {
                if i + 1 < args.len() {
                    if let Ok(threshold) = args[i + 1].parse::<f64>() {
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        modified_config.stall_threshold = threshold;
                        println!("Setting stall threshold to: {}% improvement", threshold * 100.0);
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
                    if let Ok(chunk_size) = parse_with_suffix::<usize>(&args[i + 1]) {
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
                    if let Ok(context_length) = parse_with_suffix::<usize>(&args[i + 1]) {
                        // We'll create a modified config with this value
                        let mut modified_config = create_default_config();
                        if !config_path.is_empty() {
                            if let Ok(cfg) = TBPTTConfig::load(&config_path) {
                                modified_config = cfg;
                            }
                        }
                        
                        // Get potentially updated chunk size from args if it was specified earlier
                        let mut chunk_size = modified_config.chunk_size;
                        
                        // Look for a chunk-size parameter earlier in the args
                        for j in 1..i {
                            if j + 1 < args.len() && args[j] == "--chunk-size" {
                                if let Ok(size) = parse_with_suffix::<usize>(&args[j + 1]) {
                                    chunk_size = size;
                                    println!("Using previously specified chunk size: {}", chunk_size);
                                    break;
                                }
                            }
                        }
                        
                        // Make sure model chunk size is also updated
                        modified_config.chunk_size = chunk_size;
                        modified_config.model = modified_config.model.with_chunk_size(chunk_size);
                        
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
                    if let Ok(lr) = parse_with_suffix::<f64>(&args[i + 1]) {
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
            "--device-id" => {
                if i + 1 < args.len() {
                    if let Ok(id) = args[i + 1].parse::<usize>() {
                        device_id = id;
                        println!("Using CUDA/GPU device ID: {}", device_id);
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
    let _device: BackendDevice;
    #[allow(unused_assignments)]
    let mut device_initialized = false;
    
    #[cfg(feature = "cuda")]
    let device = {
        use burn::backend::cuda::CudaDevice;
        device_initialized = true;
        println!("Using CUDA device {}", device_id);
        CudaDevice::new(device_id) // Use specified CUDA device
    };
    
    #[cfg(all(feature = "candle-cuda", not(feature = "cuda")))]
    let device = {
        use burn::backend::candle::CandleDevice;
        device_initialized = true;
        println!("Using Candle CUDA device {}", device_id);
        CandleDevice::cuda(device_id)  // Use specified CUDA device via Candle
    };
    
    #[cfg(all(feature = "candle-metal", not(feature = "cuda"), not(all(feature = "candle", feature = "candle-cuda"))))]
    let device = {
        use burn::backend::candle::CandleDevice;
        device_initialized = true;
        println!("Using Candle Metal device {}", device_id);
        CandleDevice::metal(device_id)  // Use specified Metal device
    };
    
    #[cfg(all(feature = "vulkan", not(feature = "cuda"),
              not(feature = "candle-cuda"), not(feature = "candle-metal"),
              not(feature = "candle")))]
    let device = {
        use burn::backend::wgpu::WgpuDevice;
        device_initialized = true;
        if device_id != 0 {
            println!("Warning: Vulkan backend doesn't support explicit device selection by ID");
            println!("Using default Vulkan device (device_id parameter ignored)");
        } else {
            println!("Using Vulkan device");
        }
        WgpuDevice::default()
    };
    
    #[cfg(all(feature = "wgpu", not(feature = "cuda"),
              not(feature = "candle-cuda"), not(feature = "candle-metal"),
              not(feature = "candle"), not(feature = "vulkan")))]
    let device = {
        use burn::backend::wgpu::WgpuDevice;
        device_initialized = true;
        if device_id != 0 {
            println!("Warning: WGPU backend doesn't support explicit device selection by ID");
            println!("Using default WGPU device (device_id parameter ignored)");
        } else {
            println!("Using WGPU device");
        }
        WgpuDevice::default()
    };
    
    #[cfg(all(feature = "candle", not(feature = "candle-cuda"), not(feature = "cuda"), not(feature = "wgpu"), not(feature = "candle-metal")))]
    let device = {
        use burn::backend::candle::CandleDevice;
        device_initialized = true;
        println!("Using Candle CPU device");
        CandleDevice::cpu()
    };
    
    #[cfg(all(feature = "ndarray", not(feature = "cuda"), not(feature = "wgpu"), not(feature = "candle"), not(feature = "candle-metal")))]
    let device = {
        use burn::backend::ndarray::NdArrayDevice;
        device_initialized = true;
        println!("Using NdArray device");
        NdArrayDevice
    };
    
    #[cfg(all(feature = "tch", not(feature = "cuda"), not(feature = "wgpu"), not(feature = "candle"), not(feature = "ndarray"), not(feature = "candle-metal")))]
    let device = {
        use burn::backend::libtorch::LibTorchDevice;
        device_initialized = true;
        println!("Using LibTorch CPU device");
        LibTorchDevice::Cpu
    };
    
    // If no device was initialized yet, provide a fallback
    #[cfg(all(not(feature = "cuda"), not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle"), 
             not(feature = "ndarray"), not(feature = "tch"), not(feature = "candle-metal")))]
    compile_error!("No backend feature was enabled. Please enable at least one: cuda, vulkan, wgpu, candle, ndarray, etc.");

    // This check is now just for logging
    if !device_initialized {
        println!("WARNING: Device initialization flag not set - this should not happen with the current code structure.");
        println!("Please report this as a bug.");
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
    
    // Default update_tokens to 25% of backprop_tokens if not specified, or chunk_size if neither is specified
    let update_tokens = if let Some(bp_tokens) = backprop_tokens {
        update_tokens.or(Some(bp_tokens / 4))
    } else {
        update_tokens.or(Some(chunk_size))
    };
    
    if let Some(tokens) = update_tokens {
        let k1 = calculate_chunks_for_tokens(chunk_size, tokens);
        modified_config.tbptt_k1 = k1;
        println!("Setting update frequency to every {} tokens ({} chunks)", tokens, k1);
    }
    
    if let Some(tokens) = backprop_tokens {
        let k2 = calculate_chunks_for_tokens(chunk_size, tokens);
        modified_config.tbptt_k2 = k2;
        println!("Setting backpropagation window to {} tokens ({} chunks)", tokens, k2);
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
    
    println!("Training with TBPTT - chunk size: {}, k1: {} (update every {} tokens), k2: {} (backprop through {} tokens)", 
             config.chunk_size, config.tbptt_k1, config.tbptt_k1 * config.chunk_size,
             config.tbptt_k2, config.tbptt_k2 * config.chunk_size);
    println!("Learning rate: {}", config.learning_rate);
    println!("Vocabulary size: {}", vocab.size());
    
    // Print summary of final configuration
    println!("\nFinal configuration summary:");
    println!("- Chunk size: {} characters", config.chunk_size);
    println!("- Context length: {} characters ({} chunks)", 
             config.chunk_size * config.max_chunks_per_epoch,
             config.max_chunks_per_epoch);
    println!("- Batch size: {} parallel sequences", config.batch_size);
    println!("- Model dimension: {}", config.model.dim());
    println!("- Learning rate: {}", config.learning_rate);
    println!("- TBPTT parameters:");
    println!("  - Update frequency: k1={} (every {} tokens)", 
             config.tbptt_k1, config.tbptt_k1 * config.chunk_size);
    println!("  - Backprop window: k2={} (through {} tokens)", 
             config.tbptt_k2, config.tbptt_k2 * config.chunk_size);
    
    // Train the model using TBPTT with Learner API
    println!("\nTraining with TBPTT using Learner API");
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
        1024           // dimension (increased from 512 to 1024)
    )
    .with_depth(3)     // testing with 3 layers
    .with_ff_mult(3.0) // keeping ff_mult at 3.0
    .with_expansion_factor(1.5) // keeping expansion_factor at 1.5
    .with_chunk_size(chunk_size);
    
    #[cfg(feature = "optimizer-sgd")]
    let optimizer_config = SgdConfig::new();
    
    #[cfg(feature = "optimizer-adam")]
    // Configure Adam with recommended defaults from custom_training_loop.rs pattern
    let _optimizer_config = AdamConfig::new()
        .with_beta_1(0.9)        // Default: Controls momentum decay rate 
        .with_beta_2(0.999)      // Default: Controls variance decay rate
        .with_epsilon(1e-8);     // Improved stability compared to default 1e-5
    
    // Calculate chunks for different context lengths
    let desired_context = 64000; // Desired context length in characters
    let chunks_needed = calculate_chunks_for_context(chunk_size, desired_context);
    println!("Default config using {} chunks for ~{} character context", chunks_needed, desired_context);
    
    #[cfg(feature = "optimizer-sgd")]
    let config = TBPTTConfig::new(model_config, SgdConfig::new());
    
    #[cfg(feature = "optimizer-adam")]
    let config = TBPTTConfig::new(model_config);
    
    config
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
    .with_lr_scheduler(LRSchedulerType::Cosine) // Cosine learning rate by default
    .with_min_lr_factor(0.1)       // Minimum LR at 10% of max
    .with_warmup_epochs(0)         // No warmup by default
}
