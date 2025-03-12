use burn::{
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{Tensor, Int, activation},
};
use grufinity::{
    model::{MinGRULMConfig, MinGRULM},
    dataset::CharVocab,
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
    let mut seed_text = String::from("hi");
    let mut max_tokens = 3;
    
    // Parse arguments
    for i in 1..args.len() {
        match args[i].as_str() {
            "--model" => {
                if i + 1 < args.len() {
                    model_path = args[i + 1].clone();
                    
                    // Extract directory from model path for vocab
                    if let Some(last_slash) = model_path.rfind('/') {
                        let dir = &model_path[..last_slash];
                        vocab_path = format!("{}/vocab.txt", dir);
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
                        max_tokens = n;
                    }
                }
            },
            _ => {}
        }
    }
    
    // Set up the configured backend
    use_configured_backend!();
    
    // Get the device from the appropriate backend
    let device;
    
    #[cfg(feature = "cuda-jit")]
    {
        use burn::backend::cuda_jit::CudaDevice;
        device = CudaDevice::new(0); // Use first CUDA device with JIT
        println!("Using CUDA JIT device");
    }
    
    #[cfg(all(feature = "candle", feature = "candle-cuda", not(feature = "cuda-jit")))]
    {
        use burn::backend::candle::CandleDevice;
        device = CandleDevice::Cuda(0);  // Use first CUDA device via Candle
        println!("Using Candle CUDA device");
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
    
    println!("Model path: {}", model_path);
    println!("Vocab path: {}", vocab_path);
    
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
    
    // Create a more robust model configuration to match newer models
    let config = MinGRULMConfig::new(256, 128)
        .with_depth(4)  // Increased from 2 to 4 layers
        .with_ff_mult(3.0)  // Increased from 2.0 to 3.0
        .with_expansion_factor(1.5)  // Increased from 1.2 to 1.5
        .with_chunk_size(256);
    
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
    
    // Manual token-by-token generation
    println!("Seed text: \"{}\"", seed_text);
    
    let seed_tokens: Vec<i64> = seed_text.as_bytes()
        .iter()
        .map(|&b| b as i64)
        .collect();
    
    if seed_tokens.is_empty() {
        eprintln!("Empty seed text");
        return;
    }
    
    // Create initial tensor
    let mut tokens = Tensor::<RawBackend, 1, Int>::from_data(&*seed_tokens, &device).unsqueeze::<2>();
    println!("Initial tokens shape: {:?}", tokens.dims());
    
    let mut hidden_states = None;
    
    // Generate tokens step by step
    for i in 0..max_tokens {
        println!("\n--- Step {} ---", i);
        
        // Get dimensions
        let [batch_size, seq_len] = tokens.dims();
        println!("Current tokens shape: [{}, {}]", batch_size, seq_len);
        
        // Get last token
        println!("Getting last token...");
        let last_token = tokens.clone().slice([0..batch_size, seq_len-1..seq_len]);
        println!("Last token shape: {:?}", last_token.dims());
        
        // Run forward pass
        println!("Running forward pass...");
        let (logits, new_hidden) = model.forward(last_token, hidden_states);
        println!("Forward pass complete");
        println!("Logits shape: {:?}", logits.dims());
        
        // Update hidden states
        hidden_states = Some(new_hidden);
        
        // Manually sample token
        println!("Sampling next token...");
        let [_, logits_seq_len, vocab_size] = logits.dims();
        
        // Get logits for the last position
        println!("Getting last position logits...");
        let last_pos_logits = logits.slice([0..batch_size, logits_seq_len-1..logits_seq_len, 0..vocab_size]);
        println!("Last position logits shape: {:?}", last_pos_logits.dims());
        
        // Squeeze out the sequence dimension
        println!("Squeezing sequence dimension...");
        let squeezed_logits = last_pos_logits.squeeze::<2>(1);
        println!("Squeezed logits shape: {:?}", squeezed_logits.dims());
        
        // Apply softmax
        println!("Applying softmax...");
        let probs = activation::softmax(squeezed_logits, 1);
        println!("Probs shape: {:?}", probs.dims());
        
        // Take argmax
        println!("Taking argmax...");
        let next_token = probs.argmax(1);
        println!("Next token shape: {:?}", next_token.dims());
        
        // Add sequence dimension back
        println!("Unsqueezing next token...");
        let next_token_2d = next_token.unsqueeze::<2>();
        println!("Next token 2D shape: {:?}", next_token_2d.dims());
        
        // Concatenate
        println!("Concatenating with previous tokens...");
        tokens = Tensor::cat(vec![tokens, next_token_2d], 1);
        println!("New tokens shape: {:?}", tokens.dims());
    }
    
    // Convert to text
    println!("\nGenerated tokens:");
    let reshaped = tokens.clone().reshape([tokens.dims()[0] * tokens.dims()[1]]);
    let values: Vec<i32> = reshaped.to_data().into_vec().unwrap();
    let ids: Vec<usize> = values.into_iter().map(|x| x as usize).collect();
    
    println!("Token IDs: {:?}", ids);
    let generated_text = vocab.decode_text(&ids);
    println!("Generated text: {}", generated_text);
}
