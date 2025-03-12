use burn::{
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{Tensor, Int},
};
use grufinity::{
    model::{MinGRULMConfig, MinGRULM},
    dataset::CharVocab,
    Config,
    use_configured_backend,
    RawBackend,
    Module,
};

fn main() {
    // Parse command-line arguments
    let args: Vec<String> = std::env::args().collect();
    
    // Default values
    let mut model_path = String::from("out.small/model_final.bin");
    let mut vocab_path = String::from("out.small/vocab.txt");
    let mut seed_text = String::from("Hello world");
    let mut num_chars = 50;
    let mut temperature = 0.8;
    let mut config_path = String::from("out.small/config.json");
    
    // Parse arguments
    for i in 1..args.len() {
        match args[i].as_str() {
            "--model" => {
                if i + 1 < args.len() {
                    model_path = args[i + 1].clone();
                    
                    // Extract directory from model path for vocab and config
                    if let Some(last_slash) = model_path.rfind('/') {
                        let dir = &model_path[..last_slash];
                        vocab_path = format!("{}/vocab.txt", dir);
                        config_path = format!("{}/config.json", dir);
                    }
                }
            },
            "--vocab" => {
                if i + 1 < args.len() {
                    vocab_path = args[i + 1].clone();
                }
            },
            "--seed" => {
                if i + 1 < args.len() {
                    seed_text = args[i + 1].clone();
                }
            },
            "--length" => {
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse() {
                        num_chars = n;
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
            _ => {}
        }
    }
    
    // Set up the configured backend
    use_configured_backend!();
    
    // Get the device from the appropriate backend
    let device;
    
    #[cfg(feature = "cuda")]
    {
        use burn::backend::cuda::CudaDevice;
        device = CudaDevice::new(0); // Use first CUDA device
        println!("Using CUDA device");
    }
    
    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
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
    
    println!("Model path: {}", model_path);
    println!("Vocab path: {}", vocab_path);
    println!("Config path: {}", config_path);
    
    // Load vocabulary
    let mut vocab = CharVocab::new();
    match vocab.load_from_file(&vocab_path) {
        Ok(_) => println!("Loaded vocabulary with {} tokens", vocab.size()),
        Err(e) => {
            eprintln!("Failed to load vocabulary: {}", e);
            // Create an empty vocabulary with all possible bytes
            vocab.build_from_text("");
            println!("Created default byte vocabulary with {} tokens", vocab.size());
        }
    }
    
    // Load model configuration
    let config = match MinGRULMConfig::load(&config_path) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Failed to load model config: {}", e);
            // Create a more robust default config with 4 layers
            MinGRULMConfig::new(256, 128)
                .with_depth(4)  // Increased from 2 to 4 layers
                .with_ff_mult(3.0)  // Increased from 2.0 to 3.0
                .with_expansion_factor(1.5)  // Increased from 1.2 to 1.5
                .with_chunk_size(256)
        }
    };
    
    // Initialize model
    let mut model = config.init::<RawBackend>(&device);
    
    // Load model weights with robust error handling
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    match recorder.load::<<MinGRULM<RawBackend> as Module<RawBackend>>::Record>(model_path.clone().into(), &device) {
        Ok(record) => {
            // Try to load the record directly
            model = model.load_record(record);
            println!("Model loaded successfully");
        }
        Err(e) => {
            eprintln!("Failed to load model file: {}", e);
            return;
        }
    }
    
    // Simple generation without hidden state passing
    let seed_tokens: Vec<i64> = seed_text.as_bytes()
        .iter()
        .map(|&b| b as i64)
        .collect();
    
    if seed_tokens.is_empty() {
        eprintln!("Empty seed text");
        return;
    }
    
    println!("Generating {} characters with seed: \"{}\"", num_chars, seed_text);
    
    let seed_tensor = Tensor::<RawBackend, 1, Int>::from_data(&*seed_tokens, &device).unsqueeze::<2>();
    println!("Input tensor shape: {:?}", seed_tensor.dims());
    
    // Generate text without hidden state passing
    let (generated_tokens, _) = model.generate(seed_tensor, num_chars, temperature, None);
    
    if generated_tokens.dims()[1] > 0 {
        // Convert token IDs back to text
        let reshaped = generated_tokens.clone().reshape([generated_tokens.dims()[0] * generated_tokens.dims()[1]]);
        let values: Vec<i32> = reshaped.to_data().into_vec().expect("Failed to convert tensor data to vector");
        let ids: Vec<usize> = values.into_iter().map(|x| x as usize).collect();
        
        let generated_text = vocab.decode_text(&ids);
        println!("\nGenerated text:");
        println!("{}", generated_text);
    } else {
        println!("Generated an empty sequence");
    }
}
