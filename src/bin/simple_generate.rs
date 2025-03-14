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
    #[allow(unused_assignments)]
    let _device: grufinity::BackendDevice;
    let mut _device_initialized = false;
    
    // Add a final fallback in case no backend feature is enabled
    #[cfg(feature = "ndarray")]
    {
        use burn::backend::ndarray::NdArrayDevice;
        let _fallback_device = NdArrayDevice;
        // We'll set the actual device later based on enabled features
    }
    
    #[cfg(not(any(feature = "ndarray", feature = "cuda-jit", feature = "wgpu", feature = "candle", feature = "tch")))]
    {
        // This is a compile-time error that will be triggered if no backend is enabled
        compile_error!("No backend feature was enabled. Please enable at least one: ndarray, wgpu, candle, etc.");
    }
    
    // Get the device from the appropriate backend
    #[allow(unused_assignments)]
    let mut device;
    #[allow(unused_assignments)]
    let mut _device_initialized = false;
    
    // Initialize device based on enabled features
    #[cfg(all(feature = "cuda-jit", not(feature = "wgpu"), not(feature = "candle"), not(feature = "tch"), not(feature = "ndarray")))]
    {
        use burn::backend::cuda_jit::CudaDevice;
        device = CudaDevice::new(0); // Use first CUDA device with JIT
        device_initialized = true;
        println!("Using CUDA JIT device");
    }
    
    #[cfg(all(feature = "candle-cuda", not(feature = "cuda-jit")))]
    {
        use burn::backend::candle::CandleDevice;
        device = CandleDevice::cuda(0);  // Use first CUDA device via Candle
        _device_initialized = true;
        println!("Using Candle CUDA device");
    }
    
    #[cfg(all(feature = "candle-metal", not(feature = "cuda-jit"), not(feature = "wgpu"), not(all(feature = "candle", feature = "candle-cuda"))))]
    {
        use burn::backend::candle::CandleDevice;
        device = CandleDevice::metal(0);  // Use first Metal device
        println!("Using Candle Metal device");
    }
    
    #[cfg(all(feature = "wgpu", not(feature = "cuda-jit"), 
              not(feature = "candle-cuda"), not(feature = "candle-metal"),
              not(feature = "candle")))]
    {
        use burn::backend::wgpu::WgpuDevice;
        device = WgpuDevice::default();
        _device_initialized = true;
        println!("Using WGPU device");
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
    if !_device_initialized {
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
            device_initialized = true;
            println!("Using NdArray device (fallback)");
        }
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
    
    // Verify that the config path exists, and use fallbacks if it doesn't
    if !std::path::Path::new(&config_path).exists() {
        // Try alternate filenames in the same directory
        if let Some(last_slash) = model_path.rfind('/') {
            let dir = &model_path[..last_slash];
            
            // Check for tbptt_config.json
            let tbptt_path = format!("{}/tbptt_config.json", dir);
            if std::path::Path::new(&tbptt_path).exists() {
                println!("Using TBPTT config: {}", tbptt_path);
                config_path = tbptt_path;
            } else {
                // Check for regular config.json
                let regular_path = format!("{}/config.json", dir);
                if std::path::Path::new(&regular_path).exists() {
                    println!("Using config: {}", regular_path);
                    config_path = regular_path;
                }
            }
        }
    }
    
    println!("Trying to load config from: {}", config_path);
    
    // Load model configuration
    let config = match MinGRULMConfig::load(&config_path) {
        Ok(config) => {
            println!("Loaded model configuration from: {}", config_path);
            println!("Model dimensions: {}, layers: {}", config.dim(), config.depth());
            config
        },
        Err(e) => {
            eprintln!("Failed to load model config: {}", e);
            println!("Using default configuration");
            // Create a default config with 3 layers for testing
            MinGRULMConfig::new(256, 1024)
                .with_depth(3)  // Using 3 layers
                .with_ff_mult(3.0)  // Keeping ff_mult at 3.0
                .with_expansion_factor(1.5)  // Keeping expansion_factor at 1.5
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
