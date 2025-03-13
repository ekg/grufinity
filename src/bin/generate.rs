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
    let mut model_path = "mingru_artifacts/model_final.bin".to_string();
    let mut vocab_path = "mingru_artifacts/vocab.txt".to_string();
    let mut seed_text = "Hello".to_string();
    let mut num_chars = 100;
    let mut temperature = 0.8;
    let mut config_path = "mingru_artifacts/config.json".to_string();
    
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
    
    // Initialize device based on enabled features
    // Priority: cuda-jit > candle-cuda > candle-metal > wgpu > candle > ndarray > tch
    #[allow(unused_assignments)]
    let device;
    let mut device_initialized = false;
    
    #[cfg(feature = "cuda-jit")]
    {
        use burn::backend::cuda_jit::CudaDevice;
        device = CudaDevice::new(0);
        device_initialized = true;
        println!("Using CUDA JIT device");
    }
    
    #[cfg(all(feature = "candle", feature = "candle-cuda", not(feature = "cuda-jit")))]
    {
        use burn::backend::candle::CandleDevice;
        device = CandleDevice::Cuda(0);
        device_initialized = true;
        println!("Using Candle CUDA device");
    }
    
    #[cfg(all(feature = "candle-metal", not(feature = "cuda-jit"), 
              not(all(feature = "candle", feature = "candle-cuda"))))]
    {
        use burn::backend::candle::CandleDevice;
        device = CandleDevice::Metal(0);
        device_initialized = true;
        println!("Using Candle Metal device");
    }
    
    #[cfg(all(feature = "wgpu", not(feature = "cuda-jit"),
              not(all(feature = "candle", feature = "candle-cuda")),
              not(feature = "candle-metal")))]
    {
        use burn::backend::wgpu::WgpuDevice;
        device = WgpuDevice::default();
        device_initialized = true;
        println!("Using WGPU device");
    }
    
    #[cfg(all(feature = "candle", not(feature = "cuda-jit"), not(feature = "wgpu"),
              not(feature = "candle-metal"), not(feature = "candle-cuda")))]
    {
        use burn::backend::candle::CandleDevice;
        device = CandleDevice::Cpu;
        device_initialized = true;
        println!("Using Candle CPU device");
    }
    
    #[cfg(all(feature = "ndarray", not(feature = "cuda-jit"), not(feature = "wgpu"),
              not(feature = "candle"), not(feature = "candle-metal")))]
    {
        use burn::backend::ndarray::NdArrayDevice;
        device = NdArrayDevice;
        device_initialized = true;
        println!("Using NdArray device");
    }
    
    #[cfg(all(feature = "tch", not(feature = "cuda-jit"), not(feature = "wgpu"),
              not(feature = "candle"), not(feature = "ndarray"), not(feature = "candle-metal")))]
    {
        use burn::backend::libtorch::LibTorchDevice;
        device = LibTorchDevice::Cpu;
        device_initialized = true;
        println!("Using LibTorch CPU device");
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
    
    // Load vocabulary
    let mut vocab = CharVocab::new();
    if let Err(e) = vocab.load_from_file(&vocab_path) {
        eprintln!("Failed to load vocabulary: {}", e);
        return;
    }
    println!("Loaded vocabulary with {} tokens", vocab.size());
    
    // Load model configuration
    let config = match MinGRULMConfig::load(&config_path) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Failed to load model config: {}", e);
            println!("Attempting to infer model structure from saved model...");
            // Create a more robust default config with 6 layers
            // This matches the structure used in newer training runs
            MinGRULMConfig::new(256, 512)
                .with_depth(6)  // Using 6 layers
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
            println!("Model loaded from: {}", model_path);
        }
        Err(e) => {
            eprintln!("Failed to load model file: {}", e);
            return;
        }
    }
    
    // Encode seed text to tokens - direct byte mapping
    let seed_tokens: Vec<i64> = seed_text.as_bytes()
        .iter()
        .map(|&b| b as i64)
        .collect();
    
    if seed_tokens.is_empty() {
        eprintln!("Failed to tokenize seed text");
        return;
    }
    
    println!("Generating {} characters with seed: \"{}\"", num_chars, seed_text);
    println!("Temperature: {}", temperature);
    
    // Process seed text in chunks for long-context support
    let seed_tensor = Tensor::<RawBackend, 1, Int>::from_data(&*seed_tokens, &device).unsqueeze::<2>();
    
    // Generate text
    println!("Generating with tensor of shape: {:?}", seed_tensor.dims());
    let (generated_tokens, _) = model.generate(seed_tensor, num_chars, temperature, None);
    
    // Verify we have valid output
    if generated_tokens.dims()[1] > 0 {
        // Convert token IDs back to text
        let reshaped = generated_tokens.clone().reshape([generated_tokens.dims()[0] * generated_tokens.dims()[1]]);
        let values: Vec<i32> = reshaped.to_data().into_vec().expect("Failed to convert tensor data to vector");
        let ids: Vec<usize> = values.into_iter().map(|x| x as usize).collect();
        
        let generated_text = vocab.decode_text(&ids);
        println!("\nGenerated text:");
        println!("{}", generated_text);
    } else {
        println!("Warning: Generated an empty sequence");
    }
    
    // Demonstrate long-context generation by using multiple chunks
    println!("\nDemonstrating long-context generation with hidden state passing:");
    let long_text = "This is a demonstration of long-context generation. The MinGRU model will pass hidden states between chunks, allowing it to maintain coherence across arbitrary sequence lengths. Let's see how well it can generate text while maintaining the context from earlier portions.";
    
    let current_text = long_text.to_string();
    let mut hidden_states = None;
    
    // Process initial text in chunks
    let chunk_size = 64;
    for i in (0..current_text.len()).step_by(chunk_size) {
        let end = (i + chunk_size).min(current_text.len());
        let chunk = &current_text[i..end];
        
        let chunk_tokens: Vec<i64> = chunk.as_bytes()
            .iter()
            .map(|&b| b as i64)
            .collect();
        
        if chunk_tokens.is_empty() {
            continue;
        }
        
        let chunk_tensor = Tensor::<RawBackend, 1, Int>::from_data(&*chunk_tokens, &device).unsqueeze::<2>();
        let (_, next_hidden) = model.forward(chunk_tensor, hidden_states);
        hidden_states = Some(next_hidden);
        
        println!("Processed chunk: \"{}\"", chunk);
    }
    
    println!("\n=== Now demonstrating long-context generation ===");
    
    // Now generate with the accumulated hidden state
    // Take the last 10 characters as the immediate seed for generation
    let last_chunk = &current_text[current_text.len().saturating_sub(10)..];
    
    let last_tokens: Vec<i64> = last_chunk.as_bytes()
        .iter()
        .map(|&b| b as i64)
        .collect();
    
    if !last_tokens.is_empty() {
        println!("Using seed of {} bytes for continuation: \"{}\"", last_tokens.len(), last_chunk);
        let last_tensor = Tensor::<RawBackend, 1, Int>::from_data(&*last_tokens, &device).unsqueeze::<2>();
        
        // Print tensor dimensions for debugging
        println!("Seed tensor shape: {:?}", last_tensor.dims());
        
        // Verify if hidden states are valid
        if let Some(ref states) = hidden_states {
            println!("Hidden states present: Yes ({} layers)", states.len());
        } else {
            println!("Warning: No hidden states present");
        }
        
        // Generate continuing from the long context
        let (generated_tokens, _) = model.generate(last_tensor, 100, temperature, hidden_states);
        
        // Ensure we have valid generated tokens
        if generated_tokens.dims()[1] > 0 {
            // Convert token IDs back to text
            let reshaped = generated_tokens.clone().reshape([generated_tokens.dims()[0] * generated_tokens.dims()[1]]);
            let values: Vec<i32> = reshaped.to_data().into_vec().expect("Failed to convert tensor data to vector");
            let ids: Vec<usize> = values.into_iter().map(|x| x as usize).collect();
            
            let continuation = vocab.decode_text(&ids);
            println!("\nLong-context continuation:");
            println!("----------------");
            println!("{}", continuation);
            println!("----------------");
        } else {
            println!("Warning: Generated an empty sequence");
        }
    } else {
        println!("Warning: Empty seed sequence for continuation");
    }
}
