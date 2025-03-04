use burn::{
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{Tensor, Int},
    backend::wgpu::{Wgpu, WgpuDevice},
};
use grufinity::{
    model::MinGRULMConfig,
    dataset::CharVocab,
    Config, Module,
};

type MyBackend = Wgpu<f32, i32>;

fn main() {
    // Parse arguments (simplified)
    let model_path = "out.small/model_final.bin";
    let vocab_path = "out.small/vocab.txt";
    let seed_text = "Hello world";
    let num_chars = 50;
    let temperature = 0.8;
    let config_path = "out.small/config.json";
    
    // Use the GPU-capable backend
    let device = WgpuDevice::default();
    
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
            // Create a default config
            MinGRULMConfig::new(256, 96)
                .with_depth(2)
                .with_ff_mult(2.0)
                .with_expansion_factor(1.2)
                .with_chunk_size(256)
        }
    };
    
    // Initialize model
    let mut model = config.init::<MyBackend>(&device);
    
    // Load model weights
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    match recorder.load(model_path.into(), &device) {
        Ok(record) => {
            model = model.load_record(record);
            println!("Model loaded successfully");
        },
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
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
    
    let seed_tensor = Tensor::<MyBackend, 1, Int>::from_data(&*seed_tokens, &device).unsqueeze::<2>();
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
