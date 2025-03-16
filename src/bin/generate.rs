use burn::{
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{Tensor, Int, backend::Backend},
    module::Module,
};
use rand::Rng;
use grufinity::{
    model::{MinGRULMConfig, MinGRULM},
    dataset::CharVocab,
    Config,
    use_configured_backend,
    RawBackend,
};

// Print help information
fn print_help() {
    println!("GRUfinity Text Generation");
    println!("========================");
    println!("Usage: cargo run --release --bin generate -- [OPTIONS]");
    println!("\nOptions:");
    println!("  --model PATH                   Path to trained model file");
    println!("  --vocab PATH                   Path to vocabulary file");
    println!("  --prompt TEXT                  Initial prompt to seed generation");
    println!("  --length NUM                   Number of characters to generate (default: 100)");
    println!("  --chunk-size NUM               Characters per chunk for processing (default: 64)");
    println!("  --temperature VALUE            Sampling temperature (default: 0.8)");
    println!("  --top-k VALUE                  Top-k sampling value (0 = disabled, default: 0)");
    println!("  --config PATH                  Path to model configuration (optional)");
    println!("  --device-id ID                 CUDA/GPU device ID to use (default: 0)");
    println!("\nExample:");
    println!("  cargo run --release --bin generate -- \\");
    println!("    --model artifacts/model_final.bin \\");
    println!("    --vocab artifacts/vocab.txt \\");
    println!("    --prompt \"Once upon a time\" \\");
    println!("    --length 500 \\");
    println!("    --temperature 0.8");
}

// Initialize appropriate device based on enabled features
fn initialize_device<B: Backend>(device_id: usize) -> B::Device {
    #[allow(unused_assignments)]
    let mut device_initialized = false;
    
    // Create a device based on the backend type
    let device = B::Device::default();
    
    // Log the device type being used based on enabled features
    #[cfg(feature = "cuda")]
    {
        device_initialized = true;
        println!("Using CUDA JIT device {}", device_id);
    }
    
    #[cfg(all(feature = "candle-cuda", not(feature = "cuda")))]
    {
        device_initialized = true;
        println!("Using Candle CUDA device {}", device_id);
    }
    
    #[cfg(all(feature = "candle-metal", not(feature = "cuda"), 
              not(all(feature = "candle", feature = "candle-cuda"))))]
    {
        device_initialized = true;
        println!("Using Candle Metal device {}", device_id);
    }
    
    #[cfg(all(feature = "wgpu", not(feature = "cuda"),
              not(feature = "candle-cuda"), not(feature = "candle-metal"),
              not(feature = "candle")))]
    {
        device_initialized = true;
        println!("Using WGPU device");
    }
    
    #[cfg(all(feature = "candle", not(feature = "candle-cuda"), not(feature = "cuda"), 
              not(feature = "wgpu"), not(feature = "candle-metal")))]
    {
        device_initialized = true;
        println!("Using Candle CPU device");
    }
    
    #[cfg(all(feature = "ndarray", not(feature = "cuda"), not(feature = "wgpu"), 
              not(feature = "candle"), not(feature = "candle-metal"), not(feature = "candle-cuda")))]
    {
        device_initialized = true;
        println!("Using NdArray device");
    }
    
    #[cfg(all(feature = "tch", not(feature = "cuda"), not(feature = "wgpu"), 
              not(feature = "candle"), not(feature = "ndarray"), not(feature = "candle-metal"), 
              not(feature = "candle-cuda")))]
    {
        device_initialized = true;
        println!("Using LibTorch CPU device");
    }
    
    // Error if no backend feature is enabled
    #[cfg(not(any(feature = "cuda", feature = "wgpu", feature = "candle", 
                  feature = "ndarray", feature = "tch", feature = "candle-metal", 
                  feature = "candle-cuda")))]
    compile_error!("No backend feature was enabled. Please enable at least one: cuda, wgpu, candle, ndarray, etc.");
    
    if !device_initialized {
        println!("WARNING: No device was properly initialized. Using fallback if available.");
    }
    
    device
}

// Locate configuration file with fallbacks
fn locate_config_file(config_path: &mut String, model_path: &str) {
    println!("Looking for configuration file...");
    
    // Check if the explicitly provided config path exists
    let explicit_config_exists = std::path::Path::new(config_path).exists();
    if explicit_config_exists {
        println!("Found explicitly specified config: {}", config_path);
        return;
    }
    
    println!("Explicit config not found at: {}. Searching for alternatives...", config_path);
    
    // Extract the directory from the model path
    if let Some(last_slash) = model_path.rfind('/') {
        let dir = &model_path[..last_slash];
        println!("Looking in model directory: {}", dir);
        
        // Try possible config filenames in priority order
        let possible_configs = [
            ("tbptt_config.json", "TBPTT config"),
            ("config.json", "standard config"),
        ];
        
        for (filename, desc) in possible_configs.iter() {
            let candidate_path = format!("{}/{}", dir, filename);
            println!("Checking for {} at: {}", desc, candidate_path);
            
            if std::path::Path::new(&candidate_path).exists() {
                println!("Found {} at: {}", desc, candidate_path);
                *config_path = candidate_path;
                return;
            }
        }
        
        println!("No configuration files found in model directory.");
    } else {
        println!("Could not determine model directory from path: {}", model_path);
    }
    
    println!("Will use config path: {} (may not exist)", config_path);
}

// Load model configuration with fallbacks
fn load_model_config(config_path: &str, chunk_size: usize, vocab_size: usize) -> MinGRULMConfig {
    if !std::path::Path::new(config_path).exists() {
        println!("Config file not found at: {}", config_path);
        println!("Using default configuration instead");
        
        return MinGRULMConfig::new(vocab_size, 1024)
            .with_depth(3)
            .with_ff_mult(3.0)
            .with_expansion_factor(1.5)
            .with_chunk_size(chunk_size);
    }
    
    println!("Attempting to load config from: {}", config_path);
    
    // Try to load and parse configuration manually first to check for missing fields
    if let Ok(content) = std::fs::read_to_string(config_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            if json.get("num_tokens").is_none() {
                println!("Config file is missing 'num_tokens' field. Adding it with value: {}", vocab_size);
                
                // Create a modified config with the missing field
                let mut modified_json = json.as_object().unwrap_or(&serde_json::Map::new()).clone();
                modified_json.insert("num_tokens".to_string(), serde_json::Value::Number(vocab_size.into()));
                
                // Directly construct config from modified JSON
                if let Ok(config_str) = serde_json::to_string(&modified_json) {
                    if let Ok(mut config) = serde_json::from_str::<MinGRULMConfig>(&config_str) {
                        println!("Successfully loaded model configuration with added num_tokens field");
                        println!("Model dimensions: {}, layers: {}", config.dim(), config.depth());
                        
                        // Update chunk size to match CLI argument
                        config = config.with_chunk_size(chunk_size);
                        return config;
                    }
                }
            }
        }
    }
    
    // If manual parsing failed, try normal loading
    match MinGRULMConfig::load(config_path) {
        Ok(mut config) => {
            println!("Successfully loaded model configuration");
            println!("Model dimensions: {}, layers: {}", config.dim(), config.depth());
            
            // Update chunk size to match CLI argument
            config = config.with_chunk_size(chunk_size);
            
            config
        },
        Err(e) => {
            eprintln!("Error loading config file: {}", e);
            eprintln!("The file was found but could not be parsed");
            println!("Using default configuration with vocab size: {}", vocab_size);
            
            // Try to read the file contents to help debug
            match std::fs::read_to_string(config_path) {
                Ok(content) => {
                    if content.trim().is_empty() {
                        println!("Note: Config file is empty");
                    } else if content.len() < 100 {
                        println!("File content: {}", content);
                    } else {
                        println!("File content too large to display ({}b)", content.len());
                    }
                },
                Err(read_err) => {
                    println!("Could not read file: {}", read_err);
                }
            }
            
            // Create a default config
            MinGRULMConfig::new(vocab_size, 1024)
                .with_depth(3)
                .with_ff_mult(3.0)
                .with_expansion_factor(1.5)
                .with_chunk_size(chunk_size)
        }
    }
}

// Initialize and load model
fn initialize_model<B: Backend>(
    config: &MinGRULMConfig, 
    model_path: &str,
    device: &B::Device
) -> Option<MinGRULM<B>> {
    // Initialize model
    let model = config.init::<B>(device);
    
    // Load model weights
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    match recorder.load::<<MinGRULM<B> as Module<B>>::Record>(model_path.into(), device) {
        Ok(record) => {
            let model = model.load_record(record);
            println!("Model loaded from: {}", model_path);
            Some(model)
        },
        Err(e) => {
            eprintln!("Failed to load model file: {}", e);
            None
        }
    }
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
    let mut token_data: Vec<i32> = Vec::with_capacity(batch_size * (seq_len + 1));
    
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
            // Convert to Int tensor type without to_dtype 
            // (which doesn't exist in this version of Burn)
            let indices: Vec<i32> = probs.argmax(1).to_data().into_vec().unwrap();
            return Tensor::<B, 1, Int>::from_data(&*indices, device);
        }
        
        // Instead of multinomial, we'll use a simple sampling method
        // Get probabilities and use them to select an index
        let probs_vec: Vec<f32> = probs.to_data().into_vec().unwrap();
        
        // Generate random value
        let mut rng = rand::thread_rng();
        let random: f32 = rng.gen();
        
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
        
        // Create a tensor with the selected index
        let indices = vec![selected_idx as i32];
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
            top_k_indices.push(max_idx as i32);
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
    let random: f32 = rng.gen();
    
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
    
    // Return as a tensor
    let indices = vec![final_idx];
    Tensor::<B, 1, Int>::from_data(&*indices, device)
}

// Generate text with chunking and hidden state passing
fn generate_text<B: Backend>(
    model: &MinGRULM<B>,
    vocab: &CharVocab,
    prompt: &str,
    length: usize,
    chunk_size: usize,
    temperature: f64,
    top_k: usize,
    device: &B::Device
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
        
        let chunk_tokens: Vec<i64> = chunk.as_bytes()
            .iter()
            .map(|&b| b as i64)
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
        let seed_tokens: Vec<i64> = seed.as_bytes()
            .iter()
            .map(|&b| b as i64)
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
            let values: Vec<i32> = reshaped.to_data().into_vec()
                .expect("Failed to convert tensor data to vector");
            let ids: Vec<usize> = values.into_iter().map(|x| x as usize).collect();
            
            let new_text = vocab.decode_text(&ids);
            generated_text.push_str(&new_text);
            
            // Update remaining count
            remaining -= new_text.len();
        } else {
            // If we didn't generate any new tokens, avoid an infinite loop
            break;
        }
    }
    
    generated_text
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
    let mut model_path = "mingru_artifacts/model_final.bin".to_string();
    let mut vocab_path = "mingru_artifacts/vocab.txt".to_string();
    let mut prompt = "Hello".to_string();
    let mut length = 100;
    let mut chunk_size = 64;
    let mut temperature = 0.8;
    let mut top_k: usize = 0; // 0 means disabled (use full distribution)
    let mut config_path = "mingru_artifacts/config.json".to_string();
    let mut device_id: usize = 0;
    
    // Parse arguments
    for i in 1..args.len() {
        if i + 1 >= args.len() && (args[i].starts_with("--") && args[i] != "--help" && args[i] != "-h") {
            eprintln!("Warning: Option {} has no value", args[i]);
            continue;
        }
        
        match args[i].as_str() {
            "--model" => {
                if i + 1 < args.len() {
                    model_path = args[i + 1].clone();
                    println!("Model path set to: {}", model_path);
                }
            },
            "--vocab" => {
                if i + 1 < args.len() {
                    vocab_path = args[i + 1].clone();
                    println!("Vocabulary path set to: {}", vocab_path);
                }
            },
            "--prompt" => {
                if i + 1 < args.len() {
                    prompt = args[i + 1].clone();
                    println!("Prompt set to: \"{}\"", prompt);
                }
            },
            "--length" => {
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse() {
                        length = n;
                        println!("Generation length set to: {}", length);
                    } else {
                        eprintln!("Warning: Invalid length value: {}", args[i + 1]);
                    }
                }
            },
            "--chunk-size" => {
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse() {
                        chunk_size = n;
                        println!("Chunk size set to: {}", chunk_size);
                    } else {
                        eprintln!("Warning: Invalid chunk size value: {}", args[i + 1]);
                    }
                }
            },
            "--temperature" => {
                if i + 1 < args.len() {
                    if let Ok(t) = args[i + 1].parse() {
                        temperature = t;
                        println!("Temperature set to: {}", temperature);
                    } else {
                        eprintln!("Warning: Invalid temperature value: {}", args[i + 1]);
                    }
                }
            },
            "--config" => {
                if i + 1 < args.len() {
                    config_path = args[i + 1].clone();
                    println!("Config path explicitly set to: {}", config_path);
                }
            },
            "--device-id" => {
                if i + 1 < args.len() {
                    if let Ok(id) = args[i + 1].parse() {
                        device_id = id;
                        println!("Device ID set to: {}", device_id);
                    } else {
                        eprintln!("Warning: Invalid device ID value: {}", args[i + 1]);
                    }
                }
            },
            "--top-k" => {
                if i + 1 < args.len() {
                    if let Ok(k) = args[i + 1].parse() {
                        top_k = k;
                        println!("Top-k sampling set to: {}", top_k);
                    } else {
                        eprintln!("Warning: Invalid top-k value: {}", args[i + 1]);
                    }
                }
            },
            // For backward compatibility
            "--seed" => {
                if i + 1 < args.len() {
                    prompt = args[i + 1].clone();
                    println!("Using deprecated --seed option. Use --prompt instead.");
                    println!("Prompt set to: \"{}\"", prompt);
                }
            },
            _ => {
                // Only warn about unrecognized options, not values
                if args[i].starts_with("--") {
                    eprintln!("Warning: Unrecognized option: {}", args[i]);
                }
            }
        }
    }
    
    // Set up the configured backend
    use_configured_backend!();
    
    // Initialize device based on enabled features
    let device = initialize_device::<RawBackend>(device_id);
    
    // Load vocabulary
    let mut vocab = CharVocab::new();
    if let Err(e) = vocab.load_from_file(&vocab_path) {
        eprintln!("Failed to load vocabulary: {}", e);
        return;
    }
    println!("Loaded vocabulary with {} tokens", vocab.size());
    
    // Try to locate config file
    locate_config_file(&mut config_path, &model_path);
    
    // Load model configuration
    let config = load_model_config(&config_path, chunk_size, vocab.size());
    
    // Initialize and load model
    let model = initialize_model::<RawBackend>(&config, &model_path, &device);
    if model.is_none() {
        return;
    }
    let model = model.unwrap();
    
    println!("Model loaded successfully. Ready to generate text.");
    println!("Prompt: \"{}\"", prompt);
    println!("Generating {} characters with temperature {}", length, temperature);
    if top_k > 0 {
        println!("Using top-k sampling with k = {}", top_k);
    }
    println!("Using chunk size of {} characters", chunk_size);
    
    // Generate text
    let output = generate_text(&model, &vocab, &prompt, length, chunk_size, temperature, top_k, &device);
    
    // Display the generated text
    println!("\nGenerated text:");
    println!("{}", output);
}
