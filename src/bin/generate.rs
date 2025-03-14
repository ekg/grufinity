use burn::{
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{Tensor, Int, backend::Backend},
    module::Module,
};
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
    
    // Use appropriate device type for each backend
    #[cfg(feature = "cuda-jit")]
    let device = {
        use burn::backend::cuda_jit::CudaDevice;
        device_initialized = true;
        println!("Using CUDA JIT device {}", device_id);
        CudaDevice::new(device_id)
    };
    
    #[cfg(all(feature = "candle-cuda", not(feature = "cuda-jit")))]
    let device = {
        use burn::backend::candle::CandleDevice;
        device_initialized = true;
        println!("Using Candle CUDA device {}", device_id);
        CandleDevice::cuda(device_id)
    };
    
    #[cfg(all(feature = "candle-metal", not(feature = "cuda-jit"), 
              not(all(feature = "candle", feature = "candle-cuda"))))]
    let device = {
        use burn::backend::candle::CandleDevice;
        device_initialized = true;
        println!("Using Candle Metal device {}", device_id);
        CandleDevice::metal(device_id)
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
    
    if !device_initialized {
        println!("WARNING: No device was properly initialized. Using fallback if available.");
    }
    
    device
}

// Locate configuration file with fallbacks
fn locate_config_file(config_path: &mut String, model_path: &str) {
    if !std::path::Path::new(config_path).exists() {
        // Try alternate filenames in the same directory
        if let Some(last_slash) = model_path.rfind('/') {
            let dir = &model_path[..last_slash];
            
            // Check for tbptt_config.json
            let tbptt_path = format!("{}/tbptt_config.json", dir);
            if std::path::Path::new(&tbptt_path).exists() {
                println!("Using TBPTT config: {}", tbptt_path);
                *config_path = tbptt_path;
            } else {
                // Check for regular config.json
                let regular_path = format!("{}/config.json", dir);
                if std::path::Path::new(&regular_path).exists() {
                    println!("Using config: {}", regular_path);
                    *config_path = regular_path;
                }
            }
        }
    }
    
    println!("Config path: {}", config_path);
}

// Load model configuration with fallbacks
fn load_model_config(config_path: &str, chunk_size: usize) -> MinGRULMConfig {
    match MinGRULMConfig::load(config_path) {
        Ok(mut config) => {
            println!("Loaded model configuration from: {}", config_path);
            println!("Model dimensions: {}, layers: {}", config.dim(), config.depth());
            
            // Update chunk size to match CLI argument
            config = config.with_chunk_size(chunk_size);
            
            config
        },
        Err(e) => {
            eprintln!("Failed to load model config: {}", e);
            println!("Using default configuration");
            
            // Create a default config
            MinGRULMConfig::new(256, 1024)
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

// Generate text with chunking and hidden state passing
fn generate_text<B: Backend>(
    model: &MinGRULM<B>,
    vocab: &CharVocab,
    prompt: &str,
    length: usize,
    chunk_size: usize,
    temperature: f64,
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
        let (generated_tokens, next_hidden) = model.generate(seed_tensor, gen_count, temperature, hidden_states);
        hidden_states = Some(next_hidden);
        
        // Convert tokens to text
        if generated_tokens.dims()[1] > seed_tokens.len() {
            // Extract only the newly generated tokens (skip seed)
            let new_tokens = generated_tokens.clone()
                .slice([0..1, seed_tokens.len()..generated_tokens.dims()[1]]);
            
            let reshaped = new_tokens.reshape([new_tokens.dims()[0] * new_tokens.dims()[1]]);
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
    let mut config_path = "mingru_artifacts/config.json".to_string();
    let mut device_id: usize = 0;
    
    // Parse arguments
    for i in 1..args.len() {
        match args[i].as_str() {
            "--model" => {
                if i + 1 < args.len() {
                    model_path = args[i + 1].clone();
                }
            },
            "--vocab" => {
                if i + 1 < args.len() {
                    vocab_path = args[i + 1].clone();
                }
            },
            "--prompt" => {
                if i + 1 < args.len() {
                    prompt = args[i + 1].clone();
                }
            },
            "--length" => {
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse() {
                        length = n;
                    }
                }
            },
            "--chunk-size" => {
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse() {
                        chunk_size = n;
                    }
                }
            },
            "--temperature" => {
                if i + 1 < args.len() {
                    if let Ok(t) = args[i + 1].parse() {
                        temperature = t;
                    }
                }
            },
            "--config" => {
                if i + 1 < args.len() {
                    config_path = args[i + 1].clone();
                }
            },
            "--device-id" => {
                if i + 1 < args.len() {
                    if let Ok(id) = args[i + 1].parse() {
                        device_id = id;
                    }
                }
            },
            // For backward compatibility
            "--seed" => {
                if i + 1 < args.len() {
                    prompt = args[i + 1].clone();
                }
            },
            _ => {}
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
    let config = load_model_config(&config_path, chunk_size);
    
    // Initialize and load model
    let model = initialize_model::<RawBackend>(&config, &model_path, &device);
    if model.is_none() {
        return;
    }
    let model = model.unwrap();
    
    println!("Model loaded successfully. Ready to generate text.");
    println!("Prompt: \"{}\"", prompt);
    println!("Generating {} characters with temperature {}", length, temperature);
    println!("Using chunk size of {} characters", chunk_size);
    
    // Generate text
    let output = generate_text(&model, &vocab, &prompt, length, chunk_size, temperature, &device);
    
    // Display the generated text
    println!("\nGenerated text:");
    println!("{}", output);
}
