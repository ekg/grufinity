use std::{collections::HashMap, path::Path, fs::File, io::{self, BufRead, BufReader, Write}};
use rand::{Rng, SeedableRng, rngs::StdRng};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Int, Tensor},
};
use crate::model::TextBatch;

/// Byte vocabulary for text processing
#[derive(Debug, Clone)]
pub struct CharVocab {
    byte_to_idx: HashMap<u8, usize>,
    idx_to_byte: HashMap<usize, u8>,
    size: usize,
}

impl CharVocab {
    pub fn new() -> Self {
        Self {
            byte_to_idx: HashMap::new(),
            idx_to_byte: HashMap::new(),
            size: 0,
        }
    }
    
    // Public accessor to get a byte index
    pub fn byte_to_index(&self, b: u8) -> Option<usize> {
        self.byte_to_idx.get(&b).copied()
    }
    
    // Public accessor to get a byte from an index
    pub fn index_to_byte(&self, idx: usize) -> Option<u8> {
        self.idx_to_byte.get(&idx).copied()
    }

    pub fn build_from_text(&mut self, _text: &str) {
        // Use all 256 possible byte values for a true byte-level model
        for i in 0..=255u8 {
            self.byte_to_idx.insert(i, i as usize);
            self.idx_to_byte.insert(i as usize, i);
        }
        self.size = 256; // Fixed size for all possible bytes
    }

    pub fn char_to_index(&self, c: char) -> Option<usize> {
        // Convert char to bytes and use the first byte
        let mut buf = [0u8; 4];
        c.encode_utf8(&mut buf);
        self.byte_to_idx.get(&buf[0]).copied()
    }

    pub fn index_to_char(&self, idx: usize) -> Option<char> {
        // Convert byte to char (for backward compatibility)
        self.idx_to_byte.get(&idx).map(|&b| b as char)
    }

    pub fn size(&self) -> usize {
        self.size
    }
    
    pub fn encode_text(&self, text: &str) -> Vec<usize> {
        text.as_bytes()
            .iter()
            .map(|&b| b as usize)
            .collect()
    }
    
    pub fn decode_text(&self, indices: &[usize]) -> String {
        // Convert indices to bytes and then to a valid UTF-8 string
        // If any bytes are invalid UTF-8, they'll be replaced with the replacement character
        let bytes: Vec<u8> = indices.iter()
            .map(|&idx| idx.min(255) as u8)
            .collect();
        
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = File::create(path)?;
        for (&b, &idx) in &self.byte_to_idx {
            writeln!(file, "{} {}", b, idx)?;
        }
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> io::Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        self.byte_to_idx.clear();
        self.idx_to_byte.clear();
        
        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                if let (Ok(byte), Ok(idx)) = (parts[0].parse::<u8>(), parts[1].parse::<usize>()) {
                    self.byte_to_idx.insert(byte, idx);
                    self.idx_to_byte.insert(idx, byte);
                }
            }
        }
        self.size = self.byte_to_idx.len();
        Ok(())
    }
}

/// Text dataset that supports chunking for long sequences
#[derive(Clone)]
pub struct TextDataset {
    text: String,
    sequence_length: usize,
    start_positions: Vec<usize>,
    chunk_size: usize, // For long context processing
}

impl TextDataset {
    /// Legacy constructor with fixed step size (for backward compatibility)
    pub fn new(text: String, sequence_length: usize, step_size: usize, chunk_size: usize) -> Self {
        // Generate start positions using the old method
        let bytes = text.as_bytes();
        let num_positions = if bytes.len() <= sequence_length {
            0
        } else {
            (bytes.len() - sequence_length - 1) / step_size + 1
        };
        
        let mut start_positions = Vec::with_capacity(num_positions);
        for i in 0..num_positions {
            start_positions.push(i * step_size);
        }
        
        Self {
            text,
            sequence_length,
            start_positions,
            chunk_size,
        }
    }
    
    /// New constructor with random sampling for better coverage
    pub fn new_with_random_sampling(
        text: String, 
        sequence_length: usize, 
        coverage_factor: f64, 
        seed: u64,
        chunk_size: usize
    ) -> Self {
        let bytes = text.as_bytes();
        if bytes.len() <= sequence_length {
            return Self {
                text,
                sequence_length,
                start_positions: Vec::new(),
                chunk_size,
            };
        }
        
        // Determine number of samples based on coverage
        let valid_range = bytes.len() - sequence_length;
        let num_samples = (valid_range as f64 * coverage_factor).max(1.0) as usize;
        
        // Initialize RNG with seed for reproducibility
        let mut rng = StdRng::seed_from_u64(seed);
        
        // Generate random start positions
        let mut start_positions = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            let pos = rng.gen_range(0..valid_range);
            start_positions.push(pos);
        }
        
        Self {
            text,
            sequence_length,
            start_positions,
            chunk_size,
        }
    }
    
    /// Get all chunks that form a single document
    pub fn get_document_chunks(&self, index: usize) -> Vec<(String, String, usize, bool)> {
        // Ensure the index is within the range of start positions
        if index >= self.start_positions.len() {
            return Vec::new();
        }
        
        let start_idx = self.start_positions[index];
        if start_idx + self.sequence_length > self.text.len() {
            return Vec::new();
        }
        
        let doc_text = &self.text[start_idx..start_idx + self.sequence_length];
        let num_chunks = (doc_text.len() + self.chunk_size - 1) / self.chunk_size;
        let mut chunks = Vec::with_capacity(num_chunks);
        
        for chunk_idx in 0..num_chunks {
            let chunk_start = chunk_idx * self.chunk_size;
            let chunk_end = (chunk_start + self.chunk_size + 1).min(doc_text.len());
            
            if chunk_end <= chunk_start + 1 {
                break;
            }
            
            let is_last_chunk = chunk_idx == num_chunks - 1;
            let input = &doc_text[chunk_start..chunk_end-1];
            let target = &doc_text[chunk_start+1..chunk_end];
            
            chunks.push((
                input.to_string(),
                target.to_string(),
                chunk_idx,
                is_last_chunk
            ));
        }
        
        chunks
    }
}

impl Dataset<(String, String)> for TextDataset {
    fn get(&self, index: usize) -> Option<(String, String)> {
        if index >= self.start_positions.len() {
            return None;
        }
        
        let start_idx = self.start_positions[index];
        let bytes = self.text.as_bytes();
        
        if start_idx + self.sequence_length + 1 > bytes.len() {
            return None;
        }
        
        // Use byte slices instead of character slices
        let input_bytes = &bytes[start_idx..start_idx + self.sequence_length];
        let target_bytes = &bytes[start_idx + 1..start_idx + self.sequence_length + 1];
        
        // Convert bytes to strings (needed for the return type)
        let input = String::from_utf8_lossy(input_bytes).into_owned();
        let target = String::from_utf8_lossy(target_bytes).into_owned();
        
        Some((input, target))
    }
    
    fn len(&self) -> usize {
        self.start_positions.len()
    }
}

/// Document-aware dataset for chunked processing with hidden state passing
#[derive(Clone)]
pub struct ChunkedTextDataset {
    chunks: Vec<TextChunk>,
}

#[derive(Clone)]
pub struct TextChunk {
    pub text: String,
    pub doc_id: usize,
    pub chunk_idx: usize,
    pub is_last_chunk: bool,
}

impl ChunkedTextDataset {
    pub fn new(documents: Vec<String>, sequence_length: usize, chunk_size: usize) -> Self {
        let mut chunks = Vec::new();
        
        for (doc_id, document) in documents.into_iter().enumerate() {
            // Skip documents that are too short
            if document.len() < sequence_length {
                continue;
            }
            
            // Split document into chunks
            let num_chunks = (sequence_length + chunk_size - 1) / chunk_size;
            
            for chunk_idx in 0..num_chunks {
                let start_idx = chunk_idx * chunk_size;
                let end_idx = (start_idx + chunk_size).min(sequence_length);
                
                if end_idx <= start_idx {
                    break;
                }
                
                // Create chunk with document tracking information
                chunks.push(TextChunk {
                    text: document[start_idx..end_idx].to_string(),
                    doc_id,
                    chunk_idx,
                    is_last_chunk: chunk_idx == num_chunks - 1,
                });
            }
        }
        
        Self { chunks }
    }
}

impl Dataset<TextChunk> for ChunkedTextDataset {
    fn get(&self, index: usize) -> Option<TextChunk> {
        self.chunks.get(index).cloned()
    }
    
    fn len(&self) -> usize {
        self.chunks.len()
    }
}

/// Batcher for text data
#[derive(Clone)]
pub struct TextBatcher<B: Backend> {
    pub vocab: CharVocab,
    pub device: B::Device,
}

impl<B: Backend> TextBatcher<B> {
    pub fn new(vocab: CharVocab, device: B::Device) -> Self {
        Self { vocab, device }
    }
}

impl<B: Backend> Batcher<(String, String), TextBatch<B>> for TextBatcher<B> {
    fn batch(&self, items: Vec<(String, String)>) -> TextBatch<B> {
        if items.is_empty() {
            // Return empty batch
            return TextBatch {
                input: Tensor::zeros([0, 0], &self.device),
                target: Tensor::zeros([0, 0], &self.device),
            };
        }
        
        // Find the maximum sequence length we'll use for all items
        let sequence_length = items[0].0.len();
        
        // Convert strings to token indices with uniform length
        let mut inputs = Vec::with_capacity(items.len());
        let mut targets = Vec::with_capacity(items.len());
        
        for (input_str, target_str) in items {
            // Process input sequence - direct byte to index mapping
            let input_indices: Vec<i64> = input_str.as_bytes()
                .iter()
                .map(|&b| b as i64)
                .collect();
            
            // Process target sequence - direct byte to index mapping
            let target_indices: Vec<i64> = target_str.as_bytes()
                .iter()
                .map(|&b| b as i64)
                .collect();
            
            // Ensure all sequences have the same length by padding or truncating
            let mut padded_input = Vec::with_capacity(sequence_length);
            let mut padded_target = Vec::with_capacity(sequence_length);
            
            // Fill with valid indices (truncate if too long, pad with 0 if too short)
            for i in 0..sequence_length {
                padded_input.push(if i < input_indices.len() { input_indices[i] } else { 0 });
                padded_target.push(if i < target_indices.len() { target_indices[i] } else { 0 });
            }
            
            inputs.push(Tensor::<B, 1, Int>::from_data(&*padded_input, &self.device));
            targets.push(Tensor::<B, 1, Int>::from_data(&*padded_target, &self.device));
        }
        
        // Batch tensors (now all have the same shape)
        let input = Tensor::stack(inputs, 0);
        let target = Tensor::stack(targets, 0);
        
        TextBatch { input, target }
    }
}

/// Batcher for chunked text data that preserves document structure
#[derive(Clone)]
pub struct ChunkedTextBatcher<B: Backend> {
    pub vocab: CharVocab,
    pub device: B::Device,
}

impl<B: Backend> ChunkedTextBatcher<B> {
    pub fn new(vocab: CharVocab, device: B::Device) -> Self {
        Self { vocab, device }
    }
}

// Implementation moved above - remove this duplicate

/// Batch structure with document tracking information
#[derive(Debug, Clone)]
pub struct ChunkedTextBatch<B: Backend> {
    pub input: Tensor<B, 2, Int>,
    pub target: Tensor<B, 2, Int>,
    pub doc_ids: Vec<usize>,
    pub chunk_indices: Vec<usize>,
    pub is_last_chunks: Vec<bool>,
}

impl<B: Backend> Batcher<TextChunk, ChunkedTextBatch<B>> for ChunkedTextBatcher<B> {
    fn batch(&self, items: Vec<TextChunk>) -> ChunkedTextBatch<B> {
        if items.is_empty() {
            // Return empty batch
            return ChunkedTextBatch {
                input: Tensor::zeros([0, 0], &self.device),
                target: Tensor::zeros([0, 0], &self.device),
                doc_ids: Vec::new(),
                chunk_indices: Vec::new(),
                is_last_chunks: Vec::new(),
            };
        }
        
        // Extract document tracking information
        let doc_ids: Vec<_> = items.iter().map(|chunk| chunk.doc_id).collect();
        let chunk_indices: Vec<_> = items.iter().map(|chunk| chunk.chunk_idx).collect();
        let is_last_chunks: Vec<_> = items.iter().map(|chunk| chunk.is_last_chunk).collect();
        
        // Find maximum sequence length for this batch
        // Each chunk should ideally have the same length, but let's be safe
        let max_seq_len = items.iter()
            .map(|chunk| chunk.text.len().saturating_sub(1))
            .max()
            .unwrap_or(0);
        
        // Convert strings to token indices with uniform length
        let mut inputs = Vec::with_capacity(items.len());
        let mut targets = Vec::with_capacity(items.len());
        
        for chunk in items {
            // Use byte slices instead of character slices
            let bytes = chunk.text.as_bytes();
            if bytes.len() < 2 {
                continue;
            }
            
            let input_bytes = &bytes[..bytes.len()-1];
            let target_bytes = &bytes[1..];
            
            let input_indices: Vec<i64> = input_bytes.iter()
                .map(|&b| b as i64)
                .collect();
            
            let target_indices: Vec<i64> = target_bytes.iter()
                .map(|&b| b as i64)
                .collect();
            
            // Create padded vectors with consistent length
            let mut padded_input = Vec::with_capacity(max_seq_len);
            let mut padded_target = Vec::with_capacity(max_seq_len);
            
            for i in 0..max_seq_len {
                padded_input.push(if i < input_indices.len() { input_indices[i] } else { 0 });
                padded_target.push(if i < target_indices.len() { target_indices[i] } else { 0 });
            }
            
            inputs.push(Tensor::<B, 1, Int>::from_data(&*padded_input, &self.device));
            targets.push(Tensor::<B, 1, Int>::from_data(&*padded_target, &self.device));
        }
        
        // Batch tensors (now all have the same shape)
        let input = Tensor::stack(inputs, 0);
        let target = Tensor::stack(targets, 0);
        
        ChunkedTextBatch { 
            input, 
            target, 
            doc_ids, 
            chunk_indices, 
            is_last_chunks 
        }
    }
}
