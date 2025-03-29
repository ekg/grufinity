#![recursion_limit = "256"]

use burn::{
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{Tensor, Int, backend::Backend},
    module::Module,
};
use clap::Parser;
use rand::Rng;
use std::io::Write;
use grufinity::{
    model::{MinGRULMConfig, MinGRULM},
    dataset::CharVocab,
    Config,
    use_configured_backend,
    RawBackend,
    GRUfinityError,
};

// Global verbosity setting for controlling debug output
static mut VERBOSE: bool = false;

// Helper function to create LibTorch device when B::Device is LibTorchDevice
#[cfg(feature = "tch")]
fn create_device_for_libtorch<B: Backend>(device_id: usize) -> Option<B::Device> {
    use burn::backend::libtorch::LibTorchDevice;
    // No need for type_name import
    
    // Get type name to check if it's LibTorch without using private Elem type
    let type_name = std::any::type_name::<B>();
    if type_name.contains("LibTorch") {
        // Create the appropriate LibTorch device based on features
        let device = grufinity::create_libtorch_device(device_id);
        
        // We need to convert LibTorchDevice to B::Device
        // This is safe because we've verified B contains "LibTorch"
        return Some(unsafe { 
            // We're using a pointer cast instead of transmute to avoid size issues
            let device_ptr = &device as *const LibTorchDevice as *const B::Device;
            std::ptr::read(device_ptr)
        });
    }
    
    None
}

// Helper to print debug info to stderr
fn debug(msg: &str) {
    unsafe {
        if VERBOSE {
            eprintln!("{}", msg);
        }
    }
}

/// GRUfinity text generation command line arguments
#[derive(Parser, Debug)]
#[command(
    name = "GRUfinity Text Generation",
    version,
    about = "Generate text using a trained GRUfinity model",
    long_about = None
)]
struct GenerateArgs {
    /// Path to trained model file
    #[arg(long)]
    model: Option<String>,

    /// Path to vocabulary file
    #[arg(long)]
    vocab: Option<String>,

    /// Initial prompt to seed generation
    #[arg(long, default_value = "Hello")]
    prompt: String,

    /// Number of characters to generate (supports k, m, g suffixes)
    #[arg(long, default_value = "100")]
    length: String,

    /// Characters per chunk for processing (supports k, m, g suffixes)
    #[arg(long, default_value = "64")]
    chunk_size: String,

    /// Sampling temperature (higher = more random)
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Top-k sampling value (0 = disabled)
    #[arg(long, default_value_t = 40)]
    top_k: usize,

    /// Path to model configuration (optional)
    #[arg(long)]
    config: Option<String>,

    /// CUDA/GPU device ID to use
    #[arg(long, default_value_t = 0)]
    device_id: usize,

    /// Enable verbose debug output
    #[arg(short, long)]
    verbose: bool,

    /// Legacy option (use prompt instead)
    #[arg(long, hide = true)]
    seed: Option<String>,
}

// Initialize appropriate device based on enabled features
// Initialize appropriate device based on backend type
fn initialize_device<B: Backend>(_device_id: usize) -> B::Device {
    #[allow(unused_assignments)]
    let mut device_initialized = false;
    
    // Create appropriate device based on backend type
    // Handle LibTorch devices with a separate specialized function
    #[cfg(feature = "tch")]
    {
        if let Some(device) = create_device_for_libtorch::<B>(device_id) {
            debug("Using LibTorch device");
            return device;
        }
    }
    
    // Default device for other backends
    let device = B::Device::default();
    
    // Log the device type being used based on enabled features
    #[cfg(feature = "cuda")]
    {
        device_initialized = true;
        debug(&format!("Using CUDA device {}", device_id));
    }
    
    #[cfg(all(feature = "candle-cuda", not(feature = "cuda")))]
    {
        device_initialized = true;
        debug(&format!("Using Candle CUDA device {}", device_id));
    }
    
    #[cfg(all(feature = "candle-metal", not(feature = "cuda"), 
              not(all(feature = "candle", feature = "candle-cuda"))))]
    {
        device_initialized = true;
        debug(&format!("Using Candle Metal device {}", device_id));
    }
    
    #[cfg(all(feature = "vulkan", not(feature = "cuda"),
              not(feature = "candle-cuda"), not(feature = "candle-metal"),
              not(feature = "candle")))]
    {
        device_initialized = true;
        // Note: Vulkan backend doesn't support direct device_id selection like CUDA
        if device_id != 0 {
            debug("Warning: Vulkan backend doesn't support explicit device selection by ID");
            debug("Using default Vulkan device (device_id parameter ignored)");
        } else {
            debug("Using Vulkan device");
        }
    }
    
    #[cfg(all(feature = "wgpu", not(feature = "cuda"),
              not(feature = "candle-cuda"), not(feature = "candle-metal"),
              not(feature = "candle"), not(feature = "wgpu-spirv-fusion")))]
    {
        device_initialized = true;
        // Note: WGPU backend doesn't support direct device_id selection like CUDA
        if device_id != 0 {
            debug("Warning: WGPU backend doesn't support explicit device selection by ID");
            debug("Using default WGPU device (device_id parameter ignored)");
        } else {
            debug("Using WGPU device");
        }
    }
    
    #[cfg(all(feature = "ndarray", not(feature = "cuda"), not(feature = "wgpu"), 
              not(feature = "candle"), not(feature = "candle-metal"), not(feature = "candle-cuda")))]
    {
        device_initialized = true;
        debug("Using NdArray device");
    }
    
    // Error if no backend feature is enabled
    #[cfg(not(any(feature = "cuda", feature = "wgpu", feature = "candle", 
                  feature = "ndarray", feature = "tch", feature = "candle-metal", 
                  feature = "candle-cuda", feature = "vulkan")))]
    compile_error!("No backend feature was enabled. Please enable at least one: cuda, vulkan, wgpu, candle, ndarray, etc.");
    
    if !device_initialized {
        debug("WARNING: No device was properly initialized. Using fallback if available.");
    }
    
    device
}

// Locate configuration file with fallbacks
fn locate_config_file(config_path: &mut String, model_path: &str) {
    debug("Looking for configuration file...");
    
    // Check if the explicitly provided config path exists
    let explicit_config_exists = std::path::Path::new(config_path).exists();
    if explicit_config_exists {
        debug(&format!("Found explicitly specified config: {}", config_path));
        return;
    }
    
    debug(&format!("Explicit config not found at: {}. Searching for alternatives...", config_path));
    
    // Extract the directory from the model path
    if let Some(last_slash) = model_path.rfind('/') {
        let dir = &model_path[..last_slash];
        debug(&format!("Looking in model directory: {}", dir));
        
        // Try possible config filenames in priority order
        let possible_configs = [
            ("tbptt_config.json", "TBPTT config"),
            ("config.json", "standard config"),
        ];
        
        for (filename, desc) in possible_configs.iter() {
            let candidate_path = format!("{}/{}", dir, filename);
            debug(&format!("Checking for {} at: {}", desc, candidate_path));
            
            if std::path::Path::new(&candidate_path).exists() {
                debug(&format!("Found {} at: {}", desc, candidate_path));
                *config_path = candidate_path;
                return;
            }
        }
        
        debug("No configuration files found in model directory.");
    } else {
        // Also check for Windows-style paths
        if let Some(last_slash) = model_path.rfind('\\') {
            let dir = &model_path[..last_slash];
            debug(&format!("Looking in model directory (Windows path): {}", dir));
            
            // Try possible config filenames in priority order
            let possible_configs = [
                ("tbptt_config.json", "TBPTT config"),
                ("config.json", "standard config"),
            ];
            
            for (filename, desc) in possible_configs.iter() {
                let candidate_path = format!("{}\\{}", dir, filename);
                debug(&format!("Checking for {} at: {}", desc, candidate_path));
                
                if std::path::Path::new(&candidate_path).exists() {
                    debug(&format!("Found {} at: {}", desc, candidate_path));
                    *config_path = candidate_path;
                    return;
                }
            }
            
            debug("No configuration files found in model directory.");
        } else {
            debug(&format!("Could not determine model directory from path: {}", model_path));
        }
    }
}

// Load model configuration with fallbacks
fn load_model_config(config_path: &str, chunk_size: usize, vocab_size: usize) -> Result<MinGRULMConfig, GRUfinityError> {
    if !std::path::Path::new(config_path).exists() {
        debug(&format!("Config file not found at: {}", config_path));
        
        // Use 1024 as dimension (power of 2 for optimal GPU performance)
        let config = MinGRULMConfig::new(vocab_size, 1024)
            .with_depth(3)
            .with_ff_mult(3.0)
            .with_expansion_factor(1.5)
            .with_chunk_size(chunk_size);
            
        debug(&format!("Created default config with chunk size: {}", chunk_size));
        return Ok(config);
    }
    
    debug(&format!("Loading config from: {}", config_path));
    
    // Try to load and parse configuration manually first to check for missing fields
    let content = std::fs::read_to_string(config_path)
        .map_err(|e| GRUfinityError::ConfigLoad {
            path: std::path::PathBuf::from(config_path),
            reason: format!("Failed to read file: {}", e)
        })?;
        
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
        if json.get("num_tokens").is_none() {
            debug(&format!("Adding num_tokens={} to config", vocab_size));
            
            // Create a modified config with the missing field
            let modified_json = match json.as_object() {
                Some(obj) => {
                    let mut obj_clone = obj.clone();
                    obj_clone.insert("num_tokens".to_string(), serde_json::Value::Number(vocab_size.into()));
                    obj_clone
                },
                None => {
                    return Err(GRUfinityError::ConfigLoad {
                        path: std::path::PathBuf::from(config_path),
                        reason: "Config file is not a valid JSON object".to_string()
                    });
                }
            };
            
            // Directly construct config from modified JSON
            let config_str = serde_json::to_string(&modified_json)
                .map_err(|e| GRUfinityError::ConfigLoad {
                    path: std::path::PathBuf::from(config_path),
                    reason: format!("Failed to serialize modified config: {}", e)
                })?;
                
            let mut config = serde_json::from_str::<MinGRULMConfig>(&config_str)
                .map_err(|e| GRUfinityError::ConfigLoad {
                    path: std::path::PathBuf::from(config_path),
                    reason: format!("Failed to parse modified config: {}", e)
                })?;
                
            debug(&format!("Model dims: {}, layers: {}", config.dim(), config.depth()));
            
            // Only update chunk size if explicitly specified on CLI
            if chunk_size != config.chunk_size() {
                debug(&format!("Using chunk size: {} (overriding model's: {})",
                         chunk_size, config.chunk_size()));
                config = config.with_chunk_size(chunk_size);
            }
            return Ok(config);
        }
    }
    
    // If manual parsing failed, try normal loading
    match MinGRULMConfig::load(config_path) {
        Ok(mut config) => {
            debug("Successfully loaded model configuration");
            
            // Update chunk size to match CLI argument
            if chunk_size != config.chunk_size() {
                config = config.with_chunk_size(chunk_size);
            }
            
            Ok(config)
        },
        Err(e) => {
            debug(&format!("Unable to parse config file: {}", e));
            
            // Try to read the file contents to extract chunk_size
            if let Ok(content) = std::fs::read_to_string(config_path) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(chunk_size_value) = json.get("chunk_size") {
                        if let Some(model_chunk_size) = chunk_size_value.as_u64() {
                            debug(&format!("Found chunk_size in config: {}", model_chunk_size));
                        }
                    }
                }
            }
            
            // Create a default config
            let default_config = MinGRULMConfig::new(vocab_size, 1024)
                .with_depth(3)
                .with_ff_mult(3.0)
                .with_expansion_factor(1.5)
                .with_chunk_size(chunk_size);
            
            debug(&format!("Created default config with chunk size: {}", chunk_size));
            Ok(default_config)
        }
    }
}

// Initialize and load model
fn initialize_model<B: Backend>(
    config: &MinGRULMConfig, 
    model_path: &str,
    device: &B::Device
) -> Result<MinGRULM<B>, GRUfinityError> {
    // Initialize model
    let model = config.init::<B>(device);
    
    // Load model weights
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    
    let path = std::path::PathBuf::from(model_path);
    
    // Use ? operator for early return on error
    let record = recorder.load::<<MinGRULM<B> as Module<B>>::Record>(
        model_path.into(), 
        device
    ).map_err(|e| GRUfinityError::ModelLoad {
        path: path.clone(),
        reason: format!("Failed to load model: {}", e)
    })?;
    
    let model = model.load_record(record);
    debug(&format!("Model loaded from: {}", model_path));
    
    // Print chunk size from the loaded model
    let loaded_config = model.config();
    debug(&format!("Model chunk size: {}", loaded_config.chunk_size()));
    
    Ok(model)
}

// Custom implementation of text generation with top-k sampling
fn generate_with_top_k<B: Backend>(
    model: &MinGRULM<B>,
    input_tokens: Tensor<B, 2, Int>,
    max_new_tokens: usize,
    temperature: f64,
    top_k: usize,
    hidden_states: Option<Vec<Tensor<B, 2>>>,
    device: &B::Device
) -> (Tensor<B, 2, Int>, Vec<Tensor<B, 2>>) {
    let [_batch_size, _seq_len] = input_tokens.dims();
    
    // Start with the input tokens
    let mut all_tokens = input_tokens.clone();
    let mut current_hidden = hidden_states;
    
    // Generate one token at a time
    for _ in 0..max_new_tokens {
        // Get next token distribution
        let (logits, next_hidden) = model.forward(all_tokens.clone(), current_hidden);
        current_hidden = Some(next_hidden);
        
        // Sample the next token with top-k
        let next_token = sample_with_top_k(&logits, temperature, top_k, device);
        
        // Add the new token to the sequence
        all_tokens = torch_cat_tokens(all_tokens, next_token, device);
    }
    
    (all_tokens, current_hidden.unwrap_or_default())
}

// Helper function to concatenate tokens, mimicking torch.cat
fn torch_cat_tokens<B: Backend>(
    tokens: Tensor<B, 2, Int>,
    new_token: Tensor<B, 1, Int>,
    device: &B::Device
) -> Tensor<B, 2, Int> {
    let [batch_size, seq_len] = tokens.dims();
    
    // Reshape new token to [batch_size, 1]
    let new_token = new_token.unsqueeze::<2>();
    
    // Copy existing tokens - manually create the tensor by concatenating
    let mut token_data: Vec<i64> = Vec::with_capacity(batch_size * (seq_len + 1));
    
    // Extract data from existing tokens and new token
    let tokens_data = tokens.to_data().into_vec().unwrap();
    let new_token_data = new_token.to_data().into_vec().unwrap();
    
    // Copy data from tokens to result
    for b in 0..batch_size {
        // Copy existing tokens for this batch
        let start = b * seq_len;
        token_data.extend_from_slice(&tokens_data[start..start + seq_len]);
        
        // Add new token at the end
        token_data.push(new_token_data[b]);
    }
    
    // Create new tensor from the collected data
    let result = Tensor::<B, 1, Int>::from_data(&*token_data, device).reshape([batch_size, seq_len + 1]);
    
    result
}

// Sample the next token using top-k sampling
fn sample_with_top_k<B: Backend>(
    logits: &Tensor<B, 3>,
    temperature: f64,
    k: usize,
    device: &B::Device
) -> Tensor<B, 1, Int> {
    let [batch_size, seq_len, vocab_size] = logits.dims();
    
    // Get logits for last position in sequence
    let last_pos_logits = logits.clone().slice([0..batch_size, seq_len-1..seq_len, 0..vocab_size]).squeeze(1);
    
    // Apply temperature scaling
    let scaled_logits = if temperature == 0.0 {
        last_pos_logits.clone()
    } else {
        last_pos_logits / temperature
    };
    
    // If k is 0 or >= vocab_size, use regular sampling (equivalent to top-k with k=vocab_size)
    if k == 0 || k >= vocab_size {
        // Apply softmax manually since we don't have a direct method
        // exp(x) / sum(exp(x))
        let exp_logits = scaled_logits.exp();
        let sum_exp = exp_logits.clone().sum_dim(1).unsqueeze::<2>();
        let probs = exp_logits / sum_exp;
        
        // Simple argmax for deterministic sampling (temperature=0 case)
        if temperature == 0.0 {
            // Convert to Int tensor type to match expected type (using i64 for compatibility)
            let indices_i32: Vec<i32> = probs.argmax(1).to_data().into_vec().unwrap();
            // Convert to i64 for LibTorch compatibility
            let indices: Vec<i64> = indices_i32.into_iter().map(|x| x as i64).collect();
            return Tensor::<B, 1, Int>::from_data(&*indices, device);
        }
        
        // Instead of multinomial, we'll use a simple sampling method
        // Get probabilities and use them to select an index
        let probs_vec: Vec<f32> = probs.to_data().into_vec().unwrap();
        
        // Generate random value
        let mut rng = rand::thread_rng();
        let random: f32 = rng.r#gen();
        
        // Sample based on cumulative distribution
        let mut cumsum = 0.0;
        let mut selected_idx = 0;
        
        for (idx, &prob) in probs_vec.iter().enumerate() {
            cumsum += prob;
            if random < cumsum {
                selected_idx = idx;
                break;
            }
        }
        
        // Create a tensor with the selected index - use i64 to match expected Int type
        let indices: Vec<i64> = vec![selected_idx as i64];
        return Tensor::<B, 1, Int>::from_data(&*indices, device);
    }
    
    // Otherwise, perform top-k sampling
    
    // We need to find the top-k indices and values
    // Since sorting with descending option isn't available, we'll use a different approach
    
    // We'll find the top k values ourselves
    let logits_vec: Vec<f32> = scaled_logits.to_data().into_vec().unwrap();
    let mut top_k_indices = Vec::with_capacity(k);
    let mut top_k_values = Vec::with_capacity(k);
    
    // Create a copy we can modify
    let mut logits_copy = logits_vec.clone();
    
    // Find top-k values by repeatedly finding max
    for _ in 0..k {
        if let Some(max_idx) = logits_copy.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx) 
        {
            top_k_indices.push(max_idx as i64);
            top_k_values.push(logits_vec[max_idx]);
            // Set this value to negative infinity so it's not picked again
            logits_copy[max_idx] = f32::NEG_INFINITY;
        }
    }
    
    // Apply softmax manually to get probabilities for just the top-k
    let mut exp_sum = 0.0;
    let mut top_k_probs = Vec::with_capacity(k);
    
    // Calculate exp and sum
    for &val in &top_k_values {
        let exp_val = (val / temperature as f32).exp();
        top_k_probs.push(exp_val);
        exp_sum += exp_val;
    }
    
    // Normalize to get probabilities
    for prob in &mut top_k_probs {
        *prob /= exp_sum;
    }
    
    // Sample based on the probabilities
    let mut rng = rand::thread_rng();
    let random: f32 = rng.r#gen();
    
    let mut cumsum = 0.0;
    let mut selected_idx = 0;
    
    for (idx, prob) in top_k_probs.iter().enumerate() {
        cumsum += prob;
        if random < cumsum {
            selected_idx = idx;
            break;
        }
    }
    
    // Get the original vocabulary index
    let final_idx = top_k_indices[selected_idx];
    
    // Return as a tensor - convert explicitly to Vec<i64> to match expected Int type for LibTorch
    let indices: Vec<i64> = vec![final_idx];
    Tensor::<B, 1, Int>::from_data(&*indices, device)
}

// Generate text with chunking and hidden state passing
// If stream_output is true, will print each chunk as it's generated
fn generate_text<B: Backend>(
    model: &MinGRULM<B>,
    vocab: &CharVocab,
    prompt: &str,
    length: usize,
    chunk_size: usize,
    temperature: f64,
    top_k: usize,
    device: &B::Device,
    stream_output: bool
) -> String {
    // Handle empty prompt
    if prompt.is_empty() {
        return String::new();
    }
    
    // Start with the prompt
    let mut generated_text = prompt.to_string();
    let mut hidden_states = None;
    
    // Calculate how many tokens we still need to generate
    let mut remaining = length;
    
    // Process the prompt to build initial hidden state
    for i in (0..prompt.len()).step_by(chunk_size) {
        let end = (i + chunk_size).min(prompt.len());
        let chunk = &prompt[i..end];
        
        let chunk_tokens: Vec<i32> = chunk.as_bytes()
            .iter()
            .map(|&b| b as i32)
            .collect();
        
        if chunk_tokens.is_empty() {
            continue;
        }
        
        let chunk_tensor = Tensor::<B, 1, Int>::from_data(&*chunk_tokens, device).unsqueeze::<2>();
        let (_, next_hidden) = model.forward(chunk_tensor, hidden_states);
        hidden_states = Some(next_hidden);
    }
    
    // Generate text in chunks
    while remaining > 0 {
        // Take the last bit of the generated text as the immediate seed
        let last_offset = generated_text.len().saturating_sub(chunk_size);
        let seed = &generated_text[last_offset..];
        
        // Convert seed to tokens
        let seed_tokens: Vec<i32> = seed.as_bytes()
            .iter()
            .map(|&b| b as i32)
            .collect();
        
        if seed_tokens.is_empty() {
            break;
        }
        
        // Determine how many tokens to generate in this step
        let gen_count = remaining.min(chunk_size);
        
        // Create tensor for generation
        let seed_tensor = Tensor::<B, 1, Int>::from_data(&*seed_tokens, device).unsqueeze::<2>();
        
        // Generate the next chunk
        let (generated_tokens, next_hidden) = if top_k > 0 {
            // Use custom top-k sampling if enabled
            generate_with_top_k(&model, seed_tensor, gen_count, temperature, top_k, hidden_states.clone(), device)
        } else {
            // Use regular sampling if top-k is disabled
            model.generate(seed_tensor, gen_count, temperature, hidden_states)
        };
        hidden_states = Some(next_hidden);
        
        // Convert tokens to text
        if generated_tokens.dims()[1] > seed_tokens.len() {
            // Extract only the newly generated tokens (skip seed)
            let new_tokens = generated_tokens.clone()
                .slice([0..1, seed_tokens.len()..generated_tokens.dims()[1]]);
            
            let reshaped = new_tokens.clone().reshape([new_tokens.dims()[0] * new_tokens.dims()[1]]);
            let values: Vec<i64> = reshaped.to_data().into_vec()
                .expect("Failed to convert tensor data to vector");
            let ids: Vec<usize> = values.into_iter().map(|x| x as usize).collect();
            
            let new_text = vocab.decode_text(&ids);
            generated_text.push_str(&new_text);
            
            // Stream output if enabled
            if stream_output {
                print!("{}", new_text);
                std::io::stdout().flush().unwrap();
            }
            
            // Update remaining count
            remaining -= new_text.len();
        } else {
            // If we didn't generate any new tokens, avoid an infinite loop
            break;
        }
    }
    
    generated_text
}

/// Parse a string with optional binary suffix (k, m, g) into a number
/// Using binary units: k=1024, m=1024², g=1024³
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
    // Parse command-line arguments using clap
    let args = GenerateArgs::parse();
    
    // Set default paths if not provided
    let model_path = args.model.unwrap_or_else(|| "mingru_artifacts/model_final.bin".to_string());
    let vocab_path = args.vocab.unwrap_or_else(|| "mingru_artifacts/vocab.txt".to_string());
    let config_path = args.config.unwrap_or_else(|| "mingru_artifacts/config.json".to_string());
    
    // Handle legacy seed option
    let prompt = match args.seed {
        Some(seed) => {
            println!("Using deprecated --seed option. Use --prompt instead.");
            println!("Prompt set to: \"{}\"", seed);
            seed
        },
        None => args.prompt
    };
    
    // Parse generation length and chunk size with suffix support
    let generation_length = match parse_with_suffix::<usize>(&args.length) {
        Ok(len) => len,
        Err(e) => {
            eprintln!("Error parsing generation length '{}': {}", args.length, e);
            eprintln!("Using default value of 100");
            100
        }
    };
    
    let chunk_size = match parse_with_suffix::<usize>(&args.chunk_size) {
        Ok(size) => size,
        Err(e) => {
            eprintln!("Error parsing chunk size '{}': {}", args.chunk_size, e);
            eprintln!("Using default value of 64");
            64
        }
    };
    
    // Print configuration
    println!("Model path set to: {}", model_path);
    println!("Vocabulary path set to: {} (will use default byte vocabulary if not found)", vocab_path);
    println!("Prompt set to: \"{}\"", prompt);
    println!("Generation length set to: {}", generation_length);
    println!("Chunk size explicitly set to: {}", chunk_size);
    println!("Temperature set to: {}", args.temperature);
    println!("Device ID set to: {}", args.device_id);
    debug(&format!("Top-k sampling set to: {}", args.top_k));
    
    // Set verbosity
    unsafe {
        VERBOSE = args.verbose;
    }
    
    // Set up the configured backend
    use_configured_backend!();
    
    // Initialize device based on enabled features
    let device = initialize_device::<RawBackend>(args.device_id);
    
    // Load vocabulary or create default byte vocabulary
    let mut vocab = CharVocab::new();
    if let Err(_) = vocab.load_from_file(&vocab_path) {
        debug("Creating default byte vocabulary (0-255)");
        vocab.build_from_text(""); // Creates a full 256-byte vocabulary
        debug(&format!("Using default byte vocabulary with {} tokens", vocab.size()));
    } else {
        debug(&format!("Loaded vocabulary with {} tokens", vocab.size()));
    }
    
    // Try to locate config file
    let mut config_path_mut = config_path.clone();
    locate_config_file(&mut config_path_mut, &model_path);
    
    // Load model configuration
    let config = match load_model_config(&config_path_mut, chunk_size, vocab.size()) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Warning: Failed to load config: {}", e);
            eprintln!("Using default configuration instead");
            MinGRULMConfig::new(vocab.size(), 1024)
                .with_depth(3)
                .with_ff_mult(3.0)
                .with_expansion_factor(1.5)
                .with_chunk_size(chunk_size)
        }
    };
    
    // Initialize and load model
    let model = match initialize_model::<RawBackend>(&config, &model_path, &device) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!("Failed to load model. Exiting.");
            return;
        }
    };
    
    // Log generation parameters to stderr
    if args.verbose {
        eprintln!("Generating {} characters with temperature {}", args.length, args.temperature);
        if args.top_k > 0 {
            eprintln!("Using top-k sampling with k = {}", args.top_k);
        }
        
        // Get the actual chunk size from the model
        let model_chunk_size = model.config().chunk_size();
        if chunk_size != model_chunk_size {
            eprintln!("Using chunk size {} (model's native size: {})", chunk_size, model_chunk_size);
        } else {
            eprintln!("Using chunk size: {}", chunk_size);
        }
    }
    
    // Print the prompt and generated text directly to stdout without any headers
    // This allows for clean output redirection (e.g., `generate > output.txt`)
    print!("{}", prompt);
    std::io::stdout().flush().unwrap();
    
    // Generate text with streaming enabled
    let _output = generate_text(&model, &vocab, &prompt, generation_length, chunk_size, args.temperature, args.top_k, &device, true);
    
    // Add a final newline
    println!();
}
