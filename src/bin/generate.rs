use burn::{
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{Tensor, Int},
    backend::wgpu::{Wgpu, WgpuDevice},
};
use mingru::{
    model::{MinGRULMConfig, MinGRULM},
    dataset::CharVocab,
    Config, Module,
};
use std::fs;

type MyBackend = Wgpu<f32, i32>;

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
    
    // Use the GPU-capable backend
    let device = WgpuDevice::default();
    
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
            // Create a default config
            MinGRULMConfig::new(vocab.size(), 256)
                .with_depth(3)
                .with_ff_mult(4.0)
                .with_expansion_factor(1.5)
                .with_chunk_size(256)
        }
    };
    
    // Initialize model
    let mut model = config.init::<MyBackend>(&device);
    
    // Load model weights
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    match recorder.load(model_path.clone().into(), &device) {
        Ok(record) => {
            model = model.load_record(record);
            println!("Model loaded from: {}", model_path);
        },
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            return;
        }
    }
    
    // Encode seed text to tokens
    let seed_tokens: Vec<i64> = seed_text.chars()
        .filter_map(|c| vocab.char_to_index(c).map(|idx| idx as i64))
        .collect();
    
    if seed_tokens.is_empty() {
        eprintln!("Failed to tokenize seed text");
        return;
    }
    
    println!("Generating {} characters with seed: \"{}\"", num_chars, seed_text);
    println!("Temperature: {}", temperature);
    
    // Process seed text in chunks for long-context support
    let seed_tensor = Tensor::<MyBackend, 1, Int>::from_data(&*seed_tokens, &device).unsqueeze::<2>();
    
    // Generate text
    let (generated_tokens, _) = model.generate(seed_tensor, num_chars, temperature, None);
    
    // Convert token IDs back to text
    let ids: Vec<usize> = generated_tokens
        .reshape([generated_tokens.dims()[0] * generated_tokens.dims()[1]])
        .to_data()
        .as_slice()
        .iter()
        .map(|&x| x as usize)
        .collect();
    
    let generated_text = vocab.decode_text(&ids);
    println!("\nGenerated text:");
    println!("{}", generated_text);
    
    // Demonstrate long-context generation by using multiple chunks
    println!("\nDemonstrating long-context generation with hidden state passing:");
    let long_text = "This is a demonstration of long-context generation. The MinGRU model will pass hidden states between chunks, allowing it to maintain coherence across arbitrary sequence lengths. Let's see how well it can generate text while maintaining the context from earlier portions.";
    
    let mut current_text = long_text.to_string();
    let mut hidden_states = None;
    
    // Process initial text in chunks
    let chunk_size = 64;
    for i in (0..current_text.len()).step_by(chunk_size) {
        let end = (i + chunk_size).min(current_text.len());
        let chunk = &current_text[i..end];
        
        let chunk_tokens: Vec<i64> = chunk.chars()
            .filter_map(|c| vocab.char_to_index(c).map(|idx| idx as i64))
            .collect();
        
        if chunk_tokens.is_empty() {
            continue;
        }
        
        let chunk_tensor = Tensor::<MyBackend, 1, Int>::from_data(&*chunk_tokens, &device).unsqueeze::<2>();
        let (_, next_hidden) = model.forward(chunk_tensor, hidden_states);
        hidden_states = Some(next_hidden);
        
        println!("Processed chunk: \"{}\"", chunk);
    }
    
    // Now generate with the accumulated hidden state
    let last_chunk = &current_text[current_text.len().saturating_sub(10)..];
    
    let last_tokens: Vec<i64> = last_chunk.chars()
        .filter_map(|c| vocab.char_to_index(c).map(|idx| idx as i64))
        .collect();
    
    if !last_tokens.is_empty() {
        let last_tensor = Tensor::<MyBackend, 1, Int>::from_data(&*last_tokens, &device).unsqueeze::<2>();
        
        // Generate continuing from the long context
        let (generated_tokens, _) = model.generate(last_tensor, 100, temperature, hidden_states);
        
        // Convert token IDs back to text
        let ids: Vec<usize> = generated_tokens
            .reshape([generated_tokens.dims()[0] * generated_tokens.dims()[1]])
            .to_data()
            .as_slice()
            .iter()
            .map(|&x| x as usize)
            .collect();
        
        let continuation = vocab.decode_text(&ids);
        println!("\nLong-context continuation:");
        println!("{}", continuation);
    }
}
