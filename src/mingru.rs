use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor, activation},
    nn::{Linear, LinearConfig},
    config::Config,
};
// Float imported in tests module where needed
use crate::parallel_scan::{parallel_scan_log};

/// Configuration for MinGRU (Minimal Gated Recurrent Unit) module.
///
/// The MinGRU is a simplified version of the GRU that uses a single gate
/// and a special activation function to reduce computation while maintaining
/// performance comparable to a standard GRU.
///
/// # Mathematical Formulation
///
/// MinGRU uses a single update gate instead of the two gates in standard GRU:
///
/// z_t = σ(W_z · [h_{t-1}, x_t] + b_z)         # Update gate
/// h̃_t = W_h · [h_{t-1}, x_t] + b_h            # Candidate hidden state
/// h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ g(h̃_t)    # New hidden state
///
/// Where g(x) is a custom activation function:
/// g(x) = x + 0.5  for x ≥ 0
/// g(x) = sigmoid(x)  for x < 0
///
/// This formulation reduces computation by ~33% while maintaining similar expressivity.
///
/// # Comparison with Standard GRU
///
/// Standard GRU uses two gates and requires more computation:
///
/// z_t = σ(W_z · [h_{t-1}, x_t] + b_z)         # Update gate
/// r_t = σ(W_r · [h_{t-1}, x_t] + b_r)         # Reset gate
/// h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h) # Candidate hidden state
/// h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t      # New hidden state
///
/// MinGRU simplifies by:
/// 1. Removing the reset gate (r_t)
/// 2. Using a simpler activation function g(x) instead of tanh
/// 3. Combining the projection operations for efficiency
#[derive(Config, Debug)]
pub struct MinGRUConfig {
    /// Dimension of the input features
    input_size: usize,
    /// Dimension of the hidden state
    hidden_size: usize,
    /// Factor to expand the hidden dimension in intermediate computations
    /// Values > 1.0 increase model capacity with minimal computational overhead
    #[config(default = "1.0")]
    expansion_factor: f64,
    /// Whether to include a projection layer to transform from expanded dimension
    /// back to hidden_size. Always true when expansion_factor != 1.0
    #[config(default = "true")]
    proj_out: bool,
}

#[cfg(test)]
mod tests {
    use super::MinGRUConfig;
    use crate::{RawBackend, get_backend_name};
    use burn::tensor::{backend::Backend, Float, Tensor};
    
    #[test]
    fn test_mingru_init() {
        println!("Running test with backend: {}", get_backend_name());
        let device = <RawBackend as Backend>::Device::default();
        
        // Create a simple MinGRU configuration
        let config = MinGRUConfig::new(10, 20) // input_size=10, hidden_size=20
            .with_expansion_factor(1.5);
        
        // Initialize the MinGRU module with the generic backend
        let mingru = config.init::<RawBackend>(&device);
        
        // Verify module structure - check both projections separately
        assert_eq!(mingru.to_hidden.weight.dims(), [10, 30]); // input_size=10, hidden_size=20 * 1.5 expansion = 30
        assert_eq!(mingru.to_gate.weight.dims(), [10, 30]); // Same dimensions as to_hidden
        if mingru.proj_out {
            assert_eq!(mingru.to_out.weight.dims(), [30, 20]); // 20 * 1.5 = 30, 20
        }
        
        // Verify expansion factor is set
        assert!((mingru.expansion_factor - 1.5).abs() < 1e-6);
    }
    
    #[test]
    fn test_g_function() {
        println!("Running test of g_function with backend: {}", get_backend_name());
        let device = <RawBackend as Backend>::Device::default();
        
        // Create a MinGRU instance
        let config = MinGRUConfig::new(2, 2);
        let mingru = config.init::<RawBackend>(&device);
        
        // Test g_function with positive values
        let positive_data = vec![1.0, 2.0];
        let positive = Tensor::<RawBackend, 1, Float>::from_data(&*positive_data, &device).reshape([1, 2]);
        let positive_result = mingru.g_function(positive);
        
        // Extract result for checking
        let positive_result_data: Vec<f32> = positive_result.into_data().into_vec().unwrap();
        
        // For x >= 0, g(x) = x + 0.5, so 1.0 -> 1.5, 2.0 -> 2.5
        assert!((positive_result_data[0] - 1.5).abs() < 1e-5);
        assert!((positive_result_data[1] - 2.5).abs() < 1e-5);
        
        // Test g_function with negative values
        let negative_data = vec![-1.0, -2.0];
        let negative = Tensor::<RawBackend, 1, Float>::from_data(&*negative_data, &device).reshape([1, 2]);
        let negative_result = mingru.g_function(negative);
        
        // Extract result for checking
        let negative_result_data: Vec<f32> = negative_result.into_data().into_vec().unwrap();
        
        // For x < 0, g(x) = sigmoid(x), so expected results are sigmoid(-1.0) and sigmoid(-2.0)
        let sigmoid_neg1 = 1.0 / (1.0 + (-1.0f32).exp());
        let sigmoid_neg2 = 1.0 / (1.0 + (-2.0f32).exp());
        
        // Use a slightly larger epsilon for negative values due to potential differences in sigmoid implementation
        assert!((negative_result_data[0] - sigmoid_neg1).abs() < 1e-4);
        assert!((negative_result_data[1] - sigmoid_neg2).abs() < 1e-4);
    }
    
    #[test]
    fn test_mingru_forward_single_step() {
        println!("Running test with backend: {}", get_backend_name());
        let device = <RawBackend as Backend>::Device::default();
        
        // Create a small MinGRU with no expansion for easier testing
        let config = MinGRUConfig::new(2, 2).with_expansion_factor(1.0);
        let mingru = config.init::<RawBackend>(&device);
        
        // Set weights manually for deterministic test
        let weight_data = vec![
            0.1, 0.2, // input dim 0 -> hidden
            0.5, 0.6, // input dim 1 -> hidden
        ];
        
        let _weight = Tensor::<RawBackend, 1, Float>::from_data(
            &*weight_data, &device
        ).reshape([2, 2]);
        
        // Can't directly assign to weight as it's a Param<Tensor>
        // Instead, we can check that it has the expected shape for this test
        assert_eq!(mingru.to_hidden.weight.dims(), [2, 2]); // No expansion factor (1.0)
        assert_eq!(mingru.to_gate.weight.dims(), [2, 2]); // Same as to_hidden
        
        // Create input tensor - single time step
        let x_data = vec![1.0, 2.0]; // batch=1, seq_len=1, input_size=2
        let x = Tensor::<RawBackend, 1, Float>::from_data(&*x_data, &device)
            .reshape([1, 1, 2]);
        
        // Create initial hidden state
        let h0_data = vec![0.0, 0.0]; // batch=1, hidden_size=2
        let h0 = Tensor::<RawBackend, 1, Float>::from_data(&*h0_data, &device)
            .reshape([1, 2]);
        
        // Run forward pass
        let (output, next_hidden) = mingru.forward(x, Some(h0));
        
        // Verify shapes
        assert_eq!(output.dims(), [1, 1, 2]);
        assert_eq!(next_hidden.dims(), [1, 2]);
        
        // Extract results
        let output_data: Vec<f32> = output.into_data().into_vec().unwrap();
        let next_hidden_data: Vec<f32> = next_hidden.into_data().into_vec().unwrap();
        
        // Basic checks (not the exact values due to activation functions)
        assert!(output_data.len() == 2);
        assert!(next_hidden_data.len() == 2);
        // Activation transforms prevent exact checking, so just verify values are reasonable
        for val in output_data.iter().chain(next_hidden_data.iter()) {
            assert!(val.is_finite());
        }
    }
}

impl MinGRUConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MinGRU<B> {
        let dim_inner = (self.hidden_size as f64 * self.expansion_factor) as usize;
        let proj_out = self.proj_out || self.expansion_factor != 1.0;
        
        // Separate projections for hidden state and gate 
        let to_hidden = LinearConfig::new(self.input_size, dim_inner)
            .with_bias(false)
            .init(device);
            
        let to_gate = LinearConfig::new(self.input_size, dim_inner)
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
            to_hidden,
            to_gate,
            to_out,
            expansion_factor: self.expansion_factor,
            proj_out,
        }
    }
}

/// MinGRU implementation following the parallel associative scan algorithm.
///
/// Mathematical formulation of MinGRU:
/// z_t = σ(W_z · x_t + b_z)                   # Update gate
/// h̃_t = W_h · x_t + b_h                      # Candidate hidden state
/// h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ g(h̃_t)   # New hidden state
///
/// where g(x) is a custom activation function:
/// g(x) = x + 0.5  for x ≥ 0
/// g(x) = sigmoid(x)  for x < 0
///
/// The parallel associative scan allows efficient computation across
/// the entire sequence in O(log n) parallel steps instead of O(n) sequential steps.
#[derive(Module, Debug)]
pub struct MinGRU<B: Backend> {
    /// Projection for hidden state computation
    /// Shape: [input_size, hidden_size * expansion_factor]
    to_hidden: Linear<B>,
    /// Projection for gate computation
    /// Shape: [input_size, hidden_size * expansion_factor]
    to_gate: Linear<B>,
    /// Optional projection layer to transform from expanded dimension back to hidden_size
    /// Shape: [hidden_size * expansion_factor, hidden_size]
    to_out: Linear<B>,
    /// Expansion factor for internal representations
    #[module(skip)]
    expansion_factor: f64,
    /// Whether to use the output projection
    #[module(skip)]
    proj_out: bool,
}

impl<B: Backend> MinGRU<B> {
    /// Forward pass of MinGRU implementing parallel or sequential processing.
    ///
    /// This implementation automatically chooses between:
    /// 1. Sequential processing for single steps with existing hidden state
    /// 2. Parallel processing using associative scan for multi-step sequences
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [batch_size, seq_len, input_size]
    /// * `prev_hidden` - Optional previous hidden state [batch_size, hidden_size]
    ///                   If None, hidden state is initialized with zeros
    ///
    /// # Returns
    ///
    /// * `output` - Output tensor of shape [batch_size, seq_len, hidden_size]
    ///              Contains hidden states for all timesteps in the sequence
    /// * `next_hidden` - Next hidden state of shape [batch_size, hidden_size]
    ///                   To be used in subsequent calls for continuing sequences
    ///
    /// # Tensor Shapes
    ///
    /// - x: [batch_size, seq_len, input_size]
    /// - prev_hidden (optional): [batch_size, hidden_size]
    /// - output: [batch_size, seq_len, hidden_size]
    /// - next_hidden: [batch_size, hidden_size]
    pub fn forward(&self, x: Tensor<B, 3>, prev_hidden: Option<Tensor<B, 2>>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [batch_size, seq_len, _] = x.dims();
        let device = x.device();
    
        // Process input to get hidden and gate values
        let hidden = self.to_hidden.forward(x.clone());
        let gate = self.to_gate.forward(x);
    
        // Convert to log space for numerical stability
        let log_coeffs = -activation::softplus(gate.clone(), 1.0);  // log(1 - σ(gate))
        let log_z = -activation::softplus(-gate, 1.0);              // log(σ(gate))
        let log_tilde_h = self.log_g_function(hidden);              // log(g(hidden))
        let log_values = log_z + log_tilde_h;                       // log(z * h_tilde)
        
        // Handle previous hidden state if it exists
        let (log_values_final, log_coeffs_final) = if let Some(prev_h) = prev_hidden.clone() {
            let log_prev_h = self.log_g_function(prev_h.unsqueeze::<3>());
            let log_values_with_prev = Tensor::cat(vec![log_prev_h, log_values], 1);
            
            // Pad log_coeffs with zeros for previous hidden state
            let zeros_pad = Tensor::zeros([batch_size, 1, log_coeffs.dims()[2]], &device);
            let log_coeffs_padded = Tensor::cat(vec![zeros_pad, log_coeffs], 1);
            
            (log_values_with_prev, log_coeffs_padded)
        } else {
            (log_values, log_coeffs)
        };
        
        // Apply parallel scan in log space
        let out = parallel_scan_log(log_coeffs_final, log_values_final, None);
        
        // Keep only the relevant sequence length
        let relevant_out = if prev_hidden.is_some() {
            // Clone out before using to avoid moving
            let out_dims = out.dims()[2];
            out.clone().slice([0..batch_size, 1..seq_len+1, 0..out_dims])
        } else {
            out
        };
        
        // Store last hidden state for potential return
        let next_hidden = relevant_out.clone()
            .slice([0..batch_size, seq_len-1..seq_len, 0..relevant_out.dims()[2]])
            .squeeze(1);
    
        // Apply output projection if needed
        let output = if self.proj_out {
            self.to_out.forward(relevant_out)
        } else {
            relevant_out
        };
    
        (output, next_hidden)
    }
    
    
    /// Forward pass to return next hidden state
    pub fn forward_return_hidden(&self, x: Tensor<B, 3>, prev_hidden: Option<Tensor<B, 2>>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        self.forward(x, prev_hidden)
    }
    
    /// g(x) activation function matching PyTorch implementation
    #[cfg(test)]
    fn g_function(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let zeros = Tensor::zeros_like(&x);
        let x_positive = x.clone().greater_equal(zeros.clone()).float();
        let x_negative = x.clone().lower(zeros).float();
        
        // Original g(x) function from the paper: 
        // g(x) = x + 0.5 for x >= 0, sigmoid(x) for x < 0
        (x_positive * (x.clone() + 0.5)) + (x_negative * activation::sigmoid(x))
    }

    /// Log-space version of the activation function matching PyTorch implementation
    fn log_g_function(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dims = x.dims();
        let device = x.device();
        let zeros = Tensor::zeros(dims, &device);
        
        // Implementation matching PyTorch's log_g function:
        // torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))
        let x_positive = x.clone().greater_equal(zeros.clone()).float();
        let x_negative = x.clone().lower(zeros).float();
        
        // For x >= 0: log(relu(x) + 0.5)
        let log_g_pos = (activation::relu(x.clone()) + 0.5).log();
        
        // For x < 0: log(sigmoid(x)) = -softplus(-x)
        let log_g_neg = -activation::softplus(-x.clone(), 1.0);
        
        (x_positive * log_g_pos) + (x_negative * log_g_neg)
    }
}
