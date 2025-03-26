use std::{collections::HashMap, path::Path, fs::File, io::{BufRead, BufReader, Write}};
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
        
        self.size = 256; // All 256 byte values
    }
    
    // Get a padding token index (using null byte 0)
    pub fn padding_token(&self) -> usize {
        0 // Using null byte (0) as padding token
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

    #[cfg(test)]
    pub fn get_byte_to_idx(&self) -> &HashMap<u8, usize> {
        &self.byte_to_idx
    }

    #[cfg(test)]
    pub fn get_idx_to_byte(&self) -> &HashMap<usize, u8> {
        &self.idx_to_byte
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> crate::Result<()> {
        let mut file = File::create(&path)
            .map_err(|e| crate::GRUfinityError::Io(e))?;
            
        for (&b, &idx) in &self.byte_to_idx {
            writeln!(file, "{} {}", b, idx)
                .map_err(|e| crate::GRUfinityError::Io(e))?;
        }
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> crate::Result<()> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref)
            .map_err(|e| crate::GRUfinityError::VocabLoad {
                path: path_ref.to_path_buf(),
                reason: format!("Could not open file: {}", e)
            })?;
            
        let reader = BufReader::new(file);
        
        self.byte_to_idx.clear();
        self.idx_to_byte.clear();
        
        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result.map_err(|e| crate::GRUfinityError::VocabLoad {
                path: path_ref.to_path_buf(),
                reason: format!("Error reading line {}: {}", line_num + 1, e)
            })?;
            
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                let byte = parts[0].parse::<u8>().map_err(|_| crate::GRUfinityError::VocabLoad {
                    path: path_ref.to_path_buf(),
                    reason: format!("Invalid byte value on line {}: '{}'", line_num + 1, parts[0])
                })?;
                
                let idx = parts[1].parse::<usize>().map_err(|_| crate::GRUfinityError::VocabLoad {
                    path: path_ref.to_path_buf(),
                    reason: format!("Invalid index value on line {}: '{}'", line_num + 1, parts[1])
                })?;
                
                self.byte_to_idx.insert(byte, idx);
                self.idx_to_byte.insert(idx, byte);
            }
        }
        
        self.size = self.byte_to_idx.len();
        
        if self.size == 0 {
            return Err(crate::GRUfinityError::VocabLoad {
                path: path_ref.to_path_buf(),
                reason: "Vocabulary is empty after loading".to_string()
            });
        }
        
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

/// Continuous chunked dataset for TBPTT with hidden state passing between chunks
#[derive(Clone)]
pub struct ContinuousChunkedTextDataset {
    text: String,
    start_positions: Vec<usize>,
    chunk_size: usize,
    pub max_chunks: usize,
    current_chunk: usize,
}

// Implement Dataset trait for ContinuousChunkedTextDataset
impl burn::data::dataset::Dataset<TextChunk> for ContinuousChunkedTextDataset {
    fn get(&self, index: usize) -> Option<TextChunk> {
        if index < self.max_chunks {
            // Clone current state, move to specific chunk, get chunks
            let mut clone = self.clone();
            clone.current_chunk = index;
            let chunks = clone.get_current_chunks();
            chunks.get(0).cloned()
        } else {
            None
        }
    }
    
    fn len(&self) -> usize {
        self.max_chunks
    }
}

#[derive(Clone)]
pub struct TextChunk {
    pub text: String,
    pub doc_id: usize,
    pub chunk_idx: usize,
    pub is_last_chunk: bool,
    pub is_padded: bool,
}

/// Legacy document-aware dataset for chunked processing
#[derive(Clone)]
pub struct ChunkedTextDataset {
    chunks: Vec<TextChunk>,
}

impl ContinuousChunkedTextDataset {
    pub fn new(text: String, num_positions: usize, chunk_size: usize, max_chunks: usize, seed: u64) -> Self {
        // Initialize RNG with seed for reproducibility
        let mut rng = StdRng::seed_from_u64(seed);
        
        // Generate random start positions
        let valid_range = text.len().saturating_sub(chunk_size * max_chunks);
        let mut start_positions = Vec::with_capacity(num_positions);
        
        let dataset = Self {
            text: text.clone(),
            start_positions: Vec::new(),
            chunk_size,
            max_chunks,
            current_chunk: 0,
        };
        
        if valid_range > 0 {
            for _ in 0..num_positions {
                let raw_pos = rng.gen_range(0..valid_range);
                // Ensure position is at a valid character boundary
                let pos = dataset.find_char_boundary(raw_pos);
                start_positions.push(pos);
            }
        } else {
            // If text is too short, just use beginning positions
            for i in 0..num_positions.min(dataset.text.len()) {
                // Ensure position is at a valid character boundary
                let pos = dataset.find_char_boundary(i);
                start_positions.push(pos);
            }
        }
        
        Self {
            text,
            start_positions,
            chunk_size,
            max_chunks,
            current_chunk: 0,
        }
    }
    
    /// Reset to the first chunk
    pub fn reset(&mut self) {
        self.current_chunk = 0;
    }
    
    /// Resample random starting positions with a new seed
    pub fn resample_positions(&mut self, seed: u64) {
        // Initialize RNG with new seed
        let mut rng = StdRng::seed_from_u64(seed);
        
        // Store the capacity for when we regenerate
        let capacity = self.start_positions.capacity().max(1);
        
        // Clear existing positions
        self.start_positions.clear();
        
        // Generate new random start positions
        let valid_range = self.text.len().saturating_sub(self.chunk_size * self.max_chunks);
        
        if valid_range > 0 {
            for _ in 0..capacity {
                let raw_pos = rng.gen_range(0..valid_range);
                // Ensure position is at a valid character boundary
                let pos = self.find_char_boundary(raw_pos);
                self.start_positions.push(pos);
            }
        } else {
            // If text is too short, just use beginning positions
            for i in 0..capacity.min(self.text.len()) {
                // Ensure position is at a valid character boundary
                let pos = self.find_char_boundary(i);
                self.start_positions.push(pos);
            }
        }
        
        // Reset current chunk position
        self.current_chunk = 0;
    }
    
    /// Advance to the next chunk
    pub fn next_chunk(&mut self) -> bool {
        if self.current_chunk < self.max_chunks - 1 {
            self.current_chunk += 1;
            true
        } else {
            false
        }
    }
    
    /// Get the current chunk index
    pub fn current_chunk_index(&self) -> usize {
        self.current_chunk
    }
    
    /// Check if we're at the last chunk
    pub fn is_last_chunk(&self) -> bool {
        self.current_chunk == self.max_chunks - 1
    }
    
    /// Helper function to find a valid UTF-8 character boundary
    fn find_char_boundary(&self, index: usize) -> usize {
        if index >= self.text.len() {
            return self.text.len();
        }
        
        if self.text.is_char_boundary(index) {
            return index;
        }
        
        // Find the next valid boundary
        let mut i = index + 1;
        while i < self.text.len() && !self.text.is_char_boundary(i) {
            i += 1;
        }
        i.min(self.text.len())
    }
    
    /// Get chunks for the current position in the sequence
    pub fn get_current_chunks(&self) -> Vec<TextChunk> {
        let mut chunks = Vec::with_capacity(self.start_positions.len());
        
        for (doc_id, &start_pos) in self.start_positions.iter().enumerate() {
            let raw_chunk_start = start_pos + (self.current_chunk * self.chunk_size);
            let raw_chunk_end = raw_chunk_start + self.chunk_size + 1; // +1 for target
            
            // Ensure we're at valid UTF-8 character boundaries
            let chunk_start = self.find_char_boundary(raw_chunk_start);
            let chunk_end = self.find_char_boundary(raw_chunk_end);
            
            let is_last_chunk = self.current_chunk == self.max_chunks - 1;
            let mut is_padded = false;
            
            // Create the chunk text, handling end-of-text with padding
            let chunk_text = if chunk_end <= self.text.len() {
                self.text[chunk_start..chunk_end].to_string()
            } else if chunk_start < self.text.len() {
                // Partial text + padding (just repeat the last char to reach chunk_size)
                // Use a valid char boundary for the start
                let safe_start = self.find_char_boundary(chunk_start);
                let mut text = self.text[safe_start..].to_string();
                let last_char = text.chars().last().unwrap_or(' ');
                while text.len() < self.chunk_size + 1 {
                    text.push(last_char);
                }
                is_padded = true;
                text
            } else {
                // Completely beyond text - all padding
                let padding_char = ' ';
                let padding = std::iter::repeat(padding_char)
                    .take(self.chunk_size + 1)
                    .collect::<String>();
                is_padded = true;
                padding
            };
            
            chunks.push(TextChunk {
                text: chunk_text,
                doc_id,
                chunk_idx: self.current_chunk,
                is_last_chunk,
                is_padded,
            });
        }
        
        chunks
    }
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
                    is_padded: false,
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

impl<B: Backend> Batcher<(String, String), Option<TextBatch<B>>> for TextBatcher<B> {
    fn batch(&self, items: Vec<(String, String)>) -> Option<TextBatch<B>> {
        if items.is_empty() {
            // Return None for empty batch
            return None;
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
        
        // Check if we have any valid inputs after potential skips
        if inputs.is_empty() {
            return None;
        }
        
        // Add safety check to ensure we're not stacking empty tensors
        if inputs.len() == 0 {
            return None;
        }
        
        // Create placeholder batch if needed (instead of returning None)
        // This is a better approach than letting the stack operation fail
        if inputs.len() < 1 {
            // Create a single dummy sample with padding tokens
            let pad_idx = self.vocab.padding_token() as i64;
            let dummy = vec![pad_idx; sequence_length];
            
            let dummy_tensor = Tensor::<B, 1, Int>::from_data(&*dummy, &self.device);
            inputs = vec![dummy_tensor.clone()];
            targets = vec![dummy_tensor];
        }
        
        // Batch tensors (now all have the same shape)
        let input = Tensor::stack(inputs, 0);
        let target = Tensor::stack(targets, 0);
        
        Some(TextBatch { input, target })
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
    pub is_padded: Vec<bool>,
}

impl<B: Backend> Batcher<TextChunk, Option<ChunkedTextBatch<B>>> for ChunkedTextBatcher<B> {
    fn batch(&self, items: Vec<TextChunk>) -> Option<ChunkedTextBatch<B>> {
        if items.is_empty() {
            // Return None for empty batch
            return None;
        }
        
        // Extract document tracking information
        let mut doc_ids: Vec<_> = items.iter().map(|chunk| chunk.doc_id).collect();
        let mut chunk_indices: Vec<_> = items.iter().map(|chunk| chunk.chunk_idx).collect();
        let mut is_last_chunks: Vec<_> = items.iter().map(|chunk| chunk.is_last_chunk).collect();
        let mut is_padded: Vec<_> = items.iter().map(|chunk| chunk.is_padded).collect();
        
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
                // If we don't have enough content for input/target, skip this chunk
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
        
        // Check if we have any valid inputs after processing
        if inputs.is_empty() {
            return None;
        }
        
        // Create placeholder batch if needed
        if inputs.len() < 1 {
            // Create a single dummy sample with padding tokens
            let pad_idx = self.vocab.padding_token() as i64;
            let dummy = vec![pad_idx; max_seq_len];
            
            let dummy_tensor = Tensor::<B, 1, Int>::from_data(&*dummy, &self.device);
            inputs = vec![dummy_tensor.clone()];
            targets = vec![dummy_tensor];
            
            // Add dummy document tracking info
            if doc_ids.is_empty() {
                doc_ids = vec![0];
                chunk_indices = vec![0];
                is_last_chunks = vec![true];
                is_padded = vec![true];
            }
        }
        
        // Batch tensors (now all have the same shape)
        let input = Tensor::stack(inputs, 0);
        let target = Tensor::stack(targets, 0);
        
        Some(ChunkedTextBatch { 
            input, 
            target, 
            doc_ids, 
            chunk_indices, 
            is_last_chunks,
            is_padded
        })
    }
}
