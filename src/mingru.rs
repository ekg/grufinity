use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor, activation},
    nn::{Linear, LinearConfig},
    config::Config,
};
use crate::parallel_scan::{parallel_scan_log};

/// Configuration for MinGRU module
#[derive(Config, Debug)]
pub struct MinGRUConfig {
    input_size: usize,
    hidden_size: usize,
    #[config(default = "1.0")]
    expansion_factor: f64,
    #[config(default = "true")]
    proj_out: bool,
}

impl MinGRUConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MinGRU<B> {
        let dim_inner = (self.hidden_size as f64 * self.expansion_factor) as usize;
        let proj_out = self.proj_out || self.expansion_factor != 1.0;
        
        // Hidden and gate projections
        let to_hidden_and_gate = LinearConfig::new(self.input_size, dim_inner * 2)
            .with_bias(false)
            .init(device);
        
        // Output projection if needed
        let to_out = if proj_out {
            LinearConfig::new(dim_inner, self.hidden_size)
                .with_bias(false)
                .init(device)
        } else {
            // Create a dummy linear layer (will not be used)
            LinearConfig::new(1, 1)
                .with_bias(false)
                .init(device)
        };
        
        MinGRU {
            to_hidden_and_gate,
            to_out,
            expansion_factor: self.expansion_factor,
            proj_out,
        }
    }
}

/// MinGRU implementation following the paper
#[derive(Module, Debug)]
pub struct MinGRU<B: Backend> {
    to_hidden_and_gate: Linear<B>,
    to_out: Linear<B>,
    #[module(skip)]
    expansion_factor: f64,
    #[module(skip)]
    proj_out: bool,
}

impl<B: Backend> MinGRU<B> {
    /// Forward pass of MinGRU
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [batch_size, seq_len, input_size]
    /// * `prev_hidden` - Optional previous hidden state [batch_size, hidden_size]
    ///
    /// # Returns
    ///
    /// * `output` - Output tensor [batch_size, seq_len, hidden_size]
    /// * `next_hidden` - Next hidden state [batch_size, hidden_size]
    pub fn forward(&self, x: Tensor<B, 3>, prev_hidden: Option<Tensor<B, 2>>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [batch_size, seq_len, _] = x.dims();
        let _device = x.device();
    
        // Process input to get hidden and gate values

        // Project input to get hidden and gate values
        let projected = self.to_hidden_and_gate.forward(x);
        let hidden_dim = projected.dims()[2] / 2;
    
        // Split into hidden and gate components
        let hidden = projected.clone().slice([0..batch_size, 0..seq_len, 0..hidden_dim]);
        let gate = projected.slice([0..batch_size, 0..seq_len, hidden_dim..(hidden_dim*2)]);
    
        let output = if seq_len == 1 && prev_hidden.is_some() {
            // Sequential (recurrent) mode
            self.forward_sequential(hidden, gate, prev_hidden.unwrap())
        } else {
            // Parallel mode using log-space scan
            self.forward_parallel(hidden, gate, prev_hidden)
        };
    
        // Get the last hidden state for next iteration
        let next_hidden = output.clone().slice([0..batch_size, seq_len-1..seq_len, 0..hidden_dim]).squeeze(1);
    
        // Apply output projection if needed
        let output = if self.proj_out {
            self.to_out.forward(output)
        } else {
            output
        };
    
        (output, next_hidden)
    }
    
    /// Forward pass in sequential mode (one step at a time)
    fn forward_sequential(&self, hidden: Tensor<B, 3>, gate: Tensor<B, 3>, prev_hidden: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, _, hidden_dim] = hidden.dims();
        let device = hidden.device();
        
        // Apply g and sigmoid functions
        let hidden = self.g_function(hidden.squeeze(1));
        let gate = activation::sigmoid(gate.squeeze(1));
        
        // h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
        let output = (Tensor::ones([batch_size, hidden_dim], &device) - gate.clone()) * prev_hidden + gate * hidden;
        
        // Add sequence dimension back
        output.unsqueeze()
    }
    
    /// Forward pass in parallel mode using associative scan
    fn forward_parallel(&self, hidden: Tensor<B, 3>, gate: Tensor<B, 3>, prev_hidden: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        // Log-space coefficients: log(1 - z_t)
        let log_coeffs = -activation::softplus(gate.clone(), 1.0);
        
        // Log-space values: log(z_t) + log(g(h_tilde))
        let log_z = -activation::softplus(-gate, 1.0);
        let log_tilde_h = self.log_g_function(hidden);
        let log_values = log_z + log_tilde_h;
        
        // Always use sequential scan regardless of sequence length
        parallel_scan_log(log_coeffs, log_values, prev_hidden)
    }
    
    /// g(x) function: x + 0.5 for x >= 0, sigmoid(x) for x < 0
    fn g_function(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let zeros = Tensor::zeros_like(&x);
        let x_positive = x.clone().greater_equal(zeros.clone()).float();
        let x_negative = x.clone().lower(zeros).float();
        
        (x_positive * (x.clone() + 0.5)) + (x_negative * activation::sigmoid(x))
    }
    
    /// log(g(x)) function: log(x + 0.5) for x >= 0, log(sigmoid(x)) for x < 0
    fn log_g_function(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, hidden_dim] = x.dims();
        let device = x.device();
        let zeros = Tensor::zeros([batch_size, seq_len, hidden_dim], &device);
        
        let x_positive = x.clone().greater_equal(zeros.clone()).float();
        let x_negative = x.clone().lower(zeros).float();
        
        let log_g_pos = (activation::relu(x.clone()) + 0.5).log();
        let log_g_neg = -activation::softplus(-x, 1.0);
        
        (x_positive * log_g_pos) + (x_negative * log_g_neg)
    }
}
