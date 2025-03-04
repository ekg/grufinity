use std::{collections::HashMap, path::Path, fs::File, io::{self, BufRead, BufReader, Write}};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Int, Tensor},
};
use crate::model::TextBatch;

/// Character vocabulary for text processing
#[derive(Debug, Clone)]
pub struct CharVocab {
    char_to_idx: HashMap<char, usize>,
    idx_to_char: HashMap<usize, char>,
    size: usize,
}

impl CharVocab {
    pub fn new() -> Self {
        Self {
            char_to_idx: HashMap::new(),
            idx_to_char: HashMap::new(),
            size: 0,
        }
    }

    pub fn build_from_text(&mut self, text: &str) {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars.dedup();
        
        for (i, c) in chars.into_iter().enumerate() {
            self.char_to_idx.insert(c, i);
            self.idx_to_char.insert(i, c);
        }
        self.size = self.char_to_idx.len();
    }

    pub fn char_to_index(&self, c: char) -> Option<usize> {
        self.char_to_idx.get(&c).copied()
    }

    pub fn index_to_char(&self, idx: usize) -> Option<char> {
        self.idx_to_char.get(&idx).copied()
    }

    pub fn size(&self) -> usize {
        self.size
    }
    
    pub fn encode_text(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.char_to_index(c))
            .collect()
    }
    
    pub fn decode_text(&self, indices: &[usize]) -> String {
        indices.iter()
            .filter_map(|&idx| self.index_to_char(idx))
            .collect()
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = File::create(path)?;
        for (c, idx) in &self.char_to_idx {
            writeln!(file, "{} {}", *c as u32, idx)?;
        }
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> io::Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        self.char_to_idx.clear();
        self.idx_to_char.clear();
        
        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                if let (Ok(char_code), Ok(idx)) = (parts[0].parse::<u32>(), parts[1].parse::<usize>()) {
                    if let Some(c) = std::char::from_u32(char_code) {
                        self.char_to_idx.insert(c, idx);
                        self.idx_to_char.insert(idx, c);
                    }
                }
            }
        }
        self.size = self.char_to_idx.len();
        Ok(())
    }
}

/// Text dataset that supports chunking for long sequences
#[derive(Clone)]
pub struct TextDataset {
    text: String,
    sequence_length: usize,
    step_size: usize,
    chunk_size: usize, // For long context processing
}

impl TextDataset {
    pub fn new(text: String, sequence_length: usize, step_size: usize, chunk_size: usize) -> Self {
        Self {
            text,
            sequence_length,
            step_size,
            chunk_size,
        }
    }
    
    /// Get all chunks that form a single document
    pub fn get_document_chunks(&self, doc_id: usize) -> Vec<(String, String, usize, bool)> {
        let start_idx = doc_id * self.step_size;
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
        let start_idx = index * self.step_size;
        if start_idx + self.sequence_length + 1 > self.text.len() {
            return None;
        }
        
        let input = &self.text[start_idx..start_idx + self.sequence_length];
        let target = &self.text[start_idx + 1..start_idx + self.sequence_length + 1];
        
        Some((input.to_string(), target.to_string()))
    }
    
    fn len(&self) -> usize {
        if self.text.len() <= self.sequence_length {
            return 0;
        }
        (self.text.len() - self.sequence_length - 1) / self.step_size + 1
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
    vocab: CharVocab,
    device: B::Device,
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
        
        // Convert strings to token indices
        let mut inputs = Vec::with_capacity(items.len());
        let mut targets = Vec::with_capacity(items.len());
        
        for (input_str, target_str) in items {
            let input_indices: Vec<i64> = input_str.chars()
                .filter_map(|c| self.vocab.char_to_index(c).map(|idx| idx as i64))
                .collect();
            
            let target_indices: Vec<i64> = target_str.chars()
                .filter_map(|c| self.vocab.char_to_index(c).map(|idx| idx as i64))
                .collect();
                
            inputs.push(Tensor::<B, 1, Int>::from_data(input_indices.as_slice(), &self.device));
            targets.push(Tensor::<B, 1, Int>::from_data(target_indices.as_slice(), &self.device));
        }
        
        // Batch tensors
        let input = Tensor::stack(inputs, 0);
        let target = Tensor::stack(targets, 0);
        
        TextBatch { input, target }
    }
}

/// Batcher for chunked text data that preserves document structure
#[derive(Clone)]
pub struct ChunkedTextBatcher<B: Backend> {
    vocab: CharVocab,
    device: B::Device,
}

impl<B: Backend> ChunkedTextBatcher<B> {
    pub fn new(vocab: CharVocab, device: B::Device) -> Self {
        Self { vocab, device }
    }
}

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
        
        // Convert strings to token indices
        let mut inputs = Vec::with_capacity(items.len());
        let mut targets = Vec::with_capacity(items.len());
        
        for chunk in items {
            // For simplicity, use the chunk text as input and shift by one for target
            let chars: Vec<char> = chunk.text.chars().collect();
            if chars.len() < 2 {
                continue;
            }
            
            let input_chars = &chars[..chars.len()-1];
            let target_chars = &chars[1..];
            
            let input_indices: Vec<i64> = input_chars.iter()
                .filter_map(|&c| self.vocab.char_to_index(c).map(|idx| idx as i64))
                .collect();
            
            let target_indices: Vec<i64> = target_chars.iter()
                .filter_map(|&c| self.vocab.char_to_index(c).map(|idx| idx as i64))
                .collect();
                
            inputs.push(Tensor::<B, 1, Int>::from_data(input_indices.as_slice(), &self.device));
            targets.push(Tensor::<B, 1, Int>::from_data(target_indices.as_slice(), &self.device));
        }
        
        // Batch tensors
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
