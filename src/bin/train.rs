#![recursion_limit = "256"]

use burn::{
    config::Config,
};
use clap::Parser;

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

/// GRUfinity TBPTT training command line interface
#[derive(Parser, Debug)]
#[command(
    name = "GRUfinity TBPTT Training",
    version, 
    about = "Train a GRUfinity language model using Truncated Backpropagation Through Time",
    long_about = None
)]
struct TrainingArgs {
    /// Path to training data file
    #[arg(long)]
    data: Option<String>,

    /// Directory for output artifacts
    #[arg(long, default_value = "tbptt_artifacts")]
    output: String,

    /// Path to configuration file
    #[arg(long)]
    config: Option<String>,

    /// Set learning rate
    #[arg(long)]
    learning_rate: Option<f64>,

    /// Number of random start positions
    #[arg(long)]
    batch_size: Option<usize>,

    /// Characters per chunk (supports k, m, g suffixes)
    #[arg(long)]
    chunk_size: Option<String>,

    /// Chunks to process per epoch
    #[arg(long)]
    max_chunks_per_epoch: Option<usize>,

    /// Set context length in characters (supports k, m, g suffixes)
    #[arg(long)]
    context_length: Option<String>,

    /// Number of training epochs
    #[arg(long)]
    num_epochs: Option<usize>,

    /// Maximum number of epochs if using target loss
    #[arg(long)]
    max_epochs: Option<usize>,

    /// Target validation loss to stop at (0.0 to ignore)
    #[arg(long)]
    target_valid_loss: Option<f32>,

    /// Target test loss to stop at (0.0 to ignore)
    #[arg(long)]
    target_test_loss: Option<f32>,

    /// Update parameters every NUM chunks (k1 parameter, supports k, m, g suffixes)
    #[arg(long)]
    update_chunks: Option<String>,

    /// Backprop through NUM chunks (k2 parameter, supports k, m, g suffixes)
    #[arg(long)]
    backprop_chunks: Option<String>,

    /// Update parameters every ~NUM tokens (converted to chunks, supports k, m, g suffixes)
    #[arg(long)]
    update_tokens: Option<String>,

    /// Backprop through ~NUM tokens (converted to chunks, supports k, m, g suffixes)
    #[arg(long)]
    backprop_tokens: Option<String>,

    /// Preserve hidden states between batches
    #[arg(long)]
    preserve_hidden_states: Option<bool>,

    /// Gradient clipping value (0.0 to disable)
    #[arg(long)]
    grad_clip: Option<f32>,

    /// Log interval in batches
    #[arg(long)]
    log_interval: Option<usize>,

    /// Checkpoint interval in epochs
    #[arg(long)]
    checkpoint_interval: Option<usize>,

    /// Momentum factor for SGD
    #[arg(long)]
    momentum: Option<f64>,

    /// Weight decay (L2 penalty)
    #[arg(long)]
    weight_decay: Option<f32>,

    /// Dampening for momentum
    #[arg(long)]
    dampening: Option<f64>,

    /// Enable Nesterov momentum
    #[arg(long)]
    nesterov: Option<bool>,

    /// Beta1 parameter for Adam
    #[arg(long)]
    beta1: Option<f32>,

    /// Beta2 parameter for Adam
    #[arg(long)]
    beta2: Option<f32>,

    /// Epsilon parameter for Adam
    #[arg(long)]
    epsilon: Option<f32>,

    /// Learning rate scheduler
    #[arg(long, value_parser = ["constant", "cosine", "linear"])]
    lr_scheduler: Option<String>,

    /// Minimum learning rate as a factor of initial lr
    #[arg(long)]
    min_lr_factor: Option<f64>,

    /// Number of warmup epochs
    #[arg(long)]
    warmup_epochs: Option<usize>,

    /// Threshold for reducing LR on plateau
    #[arg(long)]
    plateau_threshold: Option<f64>,

    /// Factor to reduce LR by on plateau
    #[arg(long)]
    plateau_factor: Option<f64>,

    /// Consecutive epochs below threshold before reducing LR
    #[arg(long)]
    plateau_epochs: Option<usize>,

    /// Epochs with low improvement before increasing LR
    #[arg(long)]
    stall_epochs: Option<usize>,

    /// Improvement % below which an epoch is considered stalled
    #[arg(long)]
    stall_threshold: Option<f64>,

    /// CUDA/GPU device ID to use
    #[arg(long, default_value_t = 0)]
    device_id: usize,

    /// Model hidden dimension
    #[arg(long)]
    model_dim: Option<usize>,

    /// Number of MinGRU layers
    #[arg(long)]
    model_depth: Option<usize>,

    /// Feed-forward multiplier
    #[arg(long)]
    model_ff_mult: Option<f64>,

    /// Expansion factor
    #[arg(long)]
    model_exp_factor: Option<f64>,

    /// Target number of model parameters (supports k, m, g suffixes)
    #[arg(long)]
    model_params: Option<String>,
}

/// Parse a string with optional metric suffix (k, m, g) into a number
/// Examples: "1k" -> 1024, "2m" -> 2097152, "1.5g" -> 1610612736
#[allow(dead_code)]
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
    let args = TrainingArgs::parse();
    
    // Get required data path
    let data_path = args.data.unwrap_or_else(|| {
        eprintln!("Error: No training data file specified. Use --data to specify an input file.");
        std::process::exit(1);
    });
    let artifact_dir = args.output;
    let config_path = args.config.unwrap_or_default();
    #[allow(unused_variables)]
    let device_id = args.device_id;
    
    // Token-based parameters (will be converted to chunks)
    // Note: using references instead of moving the values
    let _update_tokens_ref = &args.update_tokens;
    let _backprop_tokens_ref = &args.backprop_tokens;
    
    // Set up the configured backend
    use_configured_backend!();
    
    // Get the device from the appropriate backend
    #[allow(unused_assignments)]
    let _device: BackendDevice;
    #[allow(unused_assignments)]
    let device_initialized = false;
    
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
        CandleDevice::Cpu
    };
    
    #[cfg(all(feature = "ndarray", not(feature = "cuda"), not(feature = "wgpu"), not(feature = "candle"), not(feature = "candle-metal")))]
    let device = {
        use burn::backend::ndarray::NdArrayDevice;
        device_initialized = true;
        println!("Using NdArray device");
        NdArrayDevice::default()
    };
    
    #[cfg(all(feature = "tch", not(feature = "cuda"), not(feature = "wgpu"), not(feature = "candle"), not(feature = "ndarray"), not(feature = "candle-metal")))]
    let device = {
        device_initialized = true;
        grufinity::create_libtorch_device(device_id)
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
    let text = fs::read_to_string(&data_path).unwrap_or_else(|e| {
        eprintln!("Error: Failed to read training file '{}': {}", data_path, e);
        std::process::exit(1);
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
    
    // Apply command line arguments to the config
    if let Some(learning_rate) = args.learning_rate {
        modified_config.learning_rate = learning_rate;
        println!("Set learning rate to {}", learning_rate);
    
        // For transformer-like models with Adam, typical learning rates range from 1e-4 to 5e-3
        // Values that are too small (e.g., 1e-5) will result in slow convergence
        if learning_rate <= 0.00001 {
            println!("Warning: Learning rate is very small ({}). Consider increasing to 1e-4 to 1e-3 for faster convergence.", learning_rate);
        } else if learning_rate >= 0.01 {
            println!("Warning: Learning rate is very large ({}). Consider decreasing to 1e-3 to 5e-3 for stability.", learning_rate);
        }
    }
    
    if let Some(batch_size) = args.batch_size {
        modified_config.batch_size = batch_size;
        println!("Setting batch size to {} random start positions", batch_size);
    }
    
    if let Some(chunk_size_str) = args.chunk_size {
        match parse_with_suffix::<usize>(&chunk_size_str) {
            Ok(chunk_size) => {
                modified_config.chunk_size = chunk_size;
                // Also update the model's chunk size
                modified_config.model = modified_config.model.with_chunk_size(chunk_size);
                println!("Setting chunk size to {} characters", chunk_size);
                println!("Effective context length: {} characters", 
                       modified_config.max_chunks_per_epoch * chunk_size);
            },
            Err(e) => {
                eprintln!("Error parsing chunk size '{}': {}", chunk_size_str, e);
                std::process::exit(1);
            }
        }
    }
    
    if let Some(max_chunks) = args.max_chunks_per_epoch {
        modified_config.max_chunks_per_epoch = max_chunks;
        println!("Setting max chunks per epoch to {}", max_chunks);
        println!("Effective context length: {} characters", 
               max_chunks * modified_config.chunk_size);
    }
    
    if let Some(context_length_str) = args.context_length {
        match parse_with_suffix::<usize>(&context_length_str) {
            Ok(context_length) => {
                let chunks_needed = calculate_chunks_for_context(modified_config.chunk_size, context_length);
                modified_config.max_chunks_per_epoch = chunks_needed;
                println!("Setting context length to {} characters", context_length);
                println!("Using {} chunks with chunk size {}", chunks_needed, modified_config.chunk_size);
            },
            Err(e) => {
                eprintln!("Error parsing context length '{}': {}", context_length_str, e);
                std::process::exit(1);
            }
        }
    }
    
    if let Some(epochs) = args.num_epochs {
        modified_config.num_epochs = epochs;
        println!("Setting number of epochs to: {}", epochs);
    }
    
    if let Some(epochs) = args.max_epochs {
        modified_config.max_epochs = epochs;
        println!("Setting maximum number of epochs to: {}", epochs);
    }
    
    if let Some(loss) = args.target_valid_loss {
        modified_config.target_valid_loss = loss;
        println!("Setting target validation loss to: {}", loss);
    }
    
    if let Some(loss) = args.target_test_loss {
        modified_config.target_test_loss = loss;
        println!("Setting target test loss to: {}", loss);
    }
    
    if let Some(chunks_str) = args.update_chunks {
        match parse_with_suffix::<usize>(&chunks_str) {
            Ok(chunks) => {
                modified_config.tbptt_k1 = chunks;
                println!("Setting update frequency to every {} chunks", chunks);
            },
            Err(e) => {
                eprintln!("Error parsing update chunks '{}': {}", chunks_str, e);
                std::process::exit(1);
            }
        }
    }
    
    if let Some(chunks_str) = args.backprop_chunks {
        match parse_with_suffix::<usize>(&chunks_str) {
            Ok(chunks) => {
                modified_config.tbptt_k2 = chunks;
                println!("Setting backpropagation window to {} chunks", chunks);
            },
            Err(e) => {
                eprintln!("Error parsing backprop chunks '{}': {}", chunks_str, e);
                std::process::exit(1);
            }
        }
    }
    
    // Handle token-based parameters
    let chunk_size = modified_config.chunk_size;
    
    // Parse and default update_tokens
    let parsed_backprop_tokens = match &args.backprop_tokens {
        Some(bp_tokens_str) => {
            match parse_with_suffix::<usize>(bp_tokens_str) {
                Ok(tokens) => Some(tokens),
                Err(e) => {
                    eprintln!("Error parsing backprop tokens '{}': {}", bp_tokens_str, e);
                    std::process::exit(1);
                }
            }
        },
        None => None
    };
    
    // Default update_tokens to 25% of backprop_tokens if not specified, or chunk_size if neither is specified
    let update_tokens_str = if args.update_tokens.is_none() && parsed_backprop_tokens.is_some() {
        // Create a string representation of 25% of backprop_tokens
        Some((parsed_backprop_tokens.unwrap() / 4).to_string())
    } else if args.update_tokens.is_none() {
        Some(chunk_size.to_string())
    } else {
        args.update_tokens.clone()
    };
    
    if let Some(tokens_str) = update_tokens_str {
        match parse_with_suffix::<usize>(&tokens_str) {
            Ok(tokens) => {
                let k1 = calculate_chunks_for_tokens(chunk_size, tokens);
                modified_config.tbptt_k1 = k1;
                println!("Setting update frequency to every {} tokens ({} chunks)", tokens, k1);
            },
            Err(e) => {
                eprintln!("Error parsing update tokens '{}': {}", tokens_str, e);
                std::process::exit(1);
            }
        }
    }
    
    if let Some(tokens_str) = args.backprop_tokens {
        match parse_with_suffix::<usize>(&tokens_str) {
            Ok(tokens) => {
                let k2 = calculate_chunks_for_tokens(chunk_size, tokens);
                modified_config.tbptt_k2 = k2;
                println!("Setting backpropagation window to {} tokens ({} chunks)", tokens, k2);
            },
            Err(e) => {
                eprintln!("Error parsing backprop tokens '{}': {}", tokens_str, e);
                std::process::exit(1);
            }
        }
    }
    
    if let Some(preserve) = args.preserve_hidden_states {
        modified_config.preserve_hidden_states = preserve;
        println!("Setting preserve hidden states to: {}", preserve);
    }
    
    if let Some(clip) = args.grad_clip {
        modified_config.grad_clip = clip;
        println!("Setting gradient clipping to: {}", clip);
    
        // Add gradient clipping to the optimizer following Burn's patterns
        if clip > 0.0 {
            #[cfg(feature = "optimizer-adam")]
            {
                use burn::grad_clipping::GradientClippingConfig;
                modified_config.optimizer = modified_config.optimizer
                    .with_grad_clipping(Some(GradientClippingConfig::Norm(clip)));
                println!("Applied gradient norm clipping ({}) to Adam optimizer", clip);
            }
        
            #[cfg(feature = "optimizer-sgd")]
            {
                use burn::grad_clipping::GradientClippingConfig;
                modified_config.optimizer = modified_config.optimizer
                    .with_grad_clipping(Some(GradientClippingConfig::Norm(clip)));
                println!("Applied gradient norm clipping ({}) to SGD optimizer", clip);
            }
        }
    }
    
    if let Some(interval) = args.log_interval {
        modified_config.log_interval = interval;
        println!("Setting log interval to: {} batches", interval);
    }
    
    if let Some(interval) = args.checkpoint_interval {
        modified_config.checkpoint_interval = interval;
        println!("Setting checkpoint interval to: {} epochs", interval);
    }
    
    // Handle optimizer-specific parameters
    #[cfg(feature = "optimizer-sgd")]
    if let Some(momentum) = args.momentum {
        // Create a fresh SgdConfig with our custom momentum
        let momentum_config = MomentumConfig {
            momentum,
            dampening: args.dampening.unwrap_or(0.0),
            nesterov: args.nesterov.unwrap_or(false),
        };
        modified_config.optimizer = SgdConfig::new().with_momentum(Some(momentum_config));
        println!("Setting momentum to: {}", momentum);
    }
    
    #[cfg(feature = "optimizer-adam")]
    if let Some(beta1) = args.beta1 {
        // Update Adam parameters
        modified_config.adam_beta1 = beta1;
        modified_config.optimizer = AdamConfig::new()
            .with_beta_1(beta1)
            .with_beta_2(args.beta2.unwrap_or(modified_config.adam_beta2))
            .with_epsilon(args.epsilon.unwrap_or(modified_config.adam_epsilon));
        println!("Setting Adam beta1 to: {}", beta1);
    }
    
    #[cfg(feature = "optimizer-adam")]
    if let Some(beta2) = args.beta2 {
        // Update Adam parameters
        modified_config.adam_beta2 = beta2;
        modified_config.optimizer = AdamConfig::new()
            .with_beta_1(args.beta1.unwrap_or(modified_config.adam_beta1))
            .with_beta_2(beta2)
            .with_epsilon(args.epsilon.unwrap_or(modified_config.adam_epsilon));
        println!("Setting Adam beta2 to: {}", beta2);
    }
    
    #[cfg(feature = "optimizer-adam")]
    if let Some(epsilon) = args.epsilon {
        // Update Adam parameters
        modified_config.adam_epsilon = epsilon;
        modified_config.optimizer = AdamConfig::new()
            .with_beta_1(args.beta1.unwrap_or(modified_config.adam_beta1))
            .with_beta_2(args.beta2.unwrap_or(modified_config.adam_beta2))
            .with_epsilon(epsilon);
        println!("Setting Adam epsilon to: {}", epsilon);
    }
    
    if let Some(weight_decay) = args.weight_decay {
        #[cfg(feature = "optimizer-sgd")]
        {
            let weight_decay_config = WeightDecayConfig { penalty: weight_decay };
            modified_config.optimizer = SgdConfig::new().with_weight_decay(Some(weight_decay_config));
        }
        
        #[cfg(feature = "optimizer-adam")]
        {
            modified_config.weight_decay = Some(weight_decay);
            modified_config.optimizer = modified_config.optimizer.with_weight_decay(
                Some(WeightDecayConfig::new(weight_decay))
            );
        }
        println!("Setting weight decay to: {}", weight_decay);
    }
    
    if let Some(scheduler_type) = args.lr_scheduler {
        let scheduler = match scheduler_type.to_lowercase().as_str() {
            "cosine" => LRSchedulerType::Cosine,
            "linear" => LRSchedulerType::Linear,
            _ => LRSchedulerType::Constant,
        };
        modified_config.lr_scheduler = scheduler;
        println!("Setting learning rate scheduler to: {:?}", scheduler);
    }
    
    if let Some(factor) = args.min_lr_factor {
        modified_config.min_lr_factor = factor;
        println!("Setting minimum learning rate factor to: {}", factor);
    }
    
    if let Some(epochs) = args.warmup_epochs {
        modified_config.warmup_epochs = epochs;
        println!("Setting warmup epochs to: {}", epochs);
    }
    
    if let Some(threshold) = args.plateau_threshold {
        modified_config.plateau_threshold = threshold;
        if threshold <= 0.0 {
            println!("Disabling learning rate reduction on plateau");
        } else {
            println!("Setting plateau threshold to: {:.4}%", threshold * 100.0);
        }
    }
    
    if let Some(factor) = args.plateau_factor {
        modified_config.plateau_factor = factor;
        println!("Setting plateau factor to: {}", factor);
    }
    
    if let Some(epochs) = args.plateau_epochs {
        modified_config.plateau_epochs = epochs;
        println!("Setting plateau epochs to: {} consecutive epochs", epochs);
    }
    
    if let Some(epochs) = args.stall_epochs {
        modified_config.stall_epochs = epochs;
        println!("Setting stall epochs to: {} epochs", epochs);
    }
    
    if let Some(threshold) = args.stall_threshold {
        modified_config.stall_threshold = threshold;
        println!("Setting stall threshold to: {}% improvement", threshold * 100.0);
    }
    
    // Model architecture parameters
    if let Some(dim) = args.model_dim {
        // Round dimension to nearest multiple of 32 for optimal performance
        let rounded_dim = MinGRULMConfig::round_to_multiple_of_32(dim);
        if rounded_dim != dim {
            println!("Rounding model dimension from {} to {} (multiple of 32)", dim, rounded_dim);
        }
        
        modified_config.model = MinGRULMConfig::new(
            modified_config.model.num_tokens(),
            rounded_dim
        )
        .with_depth(modified_config.model.depth())
        .with_ff_mult(modified_config.model.ff_mult())
        .with_expansion_factor(modified_config.model.expansion_factor())
        .with_chunk_size(modified_config.model.chunk_size());
        
        println!("Setting model dimension to: {}", dim);
    }
    
    if let Some(depth) = args.model_depth {
        modified_config.model = MinGRULMConfig::new(
            modified_config.model.num_tokens(),
            modified_config.model.dim()
        )
        .with_depth(depth)
        .with_ff_mult(modified_config.model.ff_mult())
        .with_expansion_factor(modified_config.model.expansion_factor())
        .with_chunk_size(modified_config.model.chunk_size());
        
        println!("Setting model depth to: {} layers", depth);
    }
    
    if let Some(ff_mult) = args.model_ff_mult {
        modified_config.model = MinGRULMConfig::new(
            modified_config.model.num_tokens(),
            modified_config.model.dim()
        )
        .with_depth(modified_config.model.depth())
        .with_ff_mult(ff_mult)
        .with_expansion_factor(modified_config.model.expansion_factor())
        .with_chunk_size(modified_config.model.chunk_size());
        
        println!("Setting feed-forward multiplier to: {}", ff_mult);
    }
    
    if let Some(exp_factor) = args.model_exp_factor {
        modified_config.model = MinGRULMConfig::new(
            modified_config.model.num_tokens(),
            modified_config.model.dim()
        )
        .with_depth(modified_config.model.depth())
        .with_ff_mult(modified_config.model.ff_mult())
        .with_expansion_factor(exp_factor)
        .with_chunk_size(modified_config.model.chunk_size());
        
        println!("Setting expansion factor to: {}", exp_factor);
    }
    
    if let Some(param_count_str) = args.model_params {
        // Parse the parameter count with suffix support
        let param_count = match parse_with_suffix::<usize>(&param_count_str) {
            Ok(count) => count,
            Err(e) => {
                eprintln!("Error parsing model parameters '{}': {}", param_count_str, e);
                std::process::exit(1);
            }
        };
        
        // Compute the dimension needed to meet the parameter target
        let optimal_dim = modified_config.model.compute_dim_for_param_count(param_count);
        
        // Update model dimension in the config
        modified_config.model = MinGRULMConfig::new(
            modified_config.model.num_tokens(),
            optimal_dim
        )
        .with_depth(modified_config.model.depth())
        .with_ff_mult(modified_config.model.ff_mult())
        .with_expansion_factor(modified_config.model.expansion_factor())
        .with_chunk_size(modified_config.model.chunk_size());
        
        // Calculate actual parameter count for display
        let actual_params = modified_config.model.calculate_parameters();
        
        // Format parameter counts with appropriate suffixes
        let format_params = |p: usize| -> String {
            if p >= 1_000_000_000 {
                format!("{:.2}G", p as f64 / 1_000_000_000.0)
            } else if p >= 1_000_000 {
                format!("{:.2}M", p as f64 / 1_000_000.0)
            } else if p >= 1_000 {
                format!("{:.2}K", p as f64 / 1_000.0)
            } else {
                format!("{}", p)
            }
        };
        
        println!("Target parameter count: {}", format_params(param_count));
        println!("Setting model dimension to: {} (gives {} parameters)", 
                 optimal_dim, format_params(actual_params));
    }
    
    // Save modified config
    let temp_config_path = "temp_config.json";
    modified_config.save(temp_config_path).expect("Failed to save temporary config");
    let config_path = temp_config_path;
    
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
    
    // Clean up temporary config file if it was created
    if std::path::Path::new("temp_config.json").exists() {
        std::fs::remove_file("temp_config.json").ok();
    }
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
        1024           // dimension (power of 2 for optimal GPU performance)
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
    
    // Calculate and display the parameter count
    let param_count = model_config.calculate_parameters();
    
    // Format the parameter count with appropriate suffixes
    let formatted_count = if param_count >= 1_000_000 {
        format!("{:.2}M", param_count as f64 / 1_000_000.0)
    } else if param_count >= 1_000 {
        format!("{:.2}K", param_count as f64 / 1_000.0)
    } else {
        format!("{}", param_count)
    };
    
    println!("Total model parameters: {}", formatted_count);
    
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
    .with_min_lr_factor(0.01)      // Minimum LR at 1% of max
    .with_warmup_epochs(0)         // No warmup by default
    .with_stall_threshold(0.01)    // 1% improvement threshold 
    .with_stall_epochs(0)          // Disabled by default, use positive values to enable
}
