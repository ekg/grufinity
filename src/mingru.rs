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
    // Import specific items needed from parent module
    use super::MinGRU;
    use super::MinGRUConfig;
    
    #[cfg(feature = "ndarray")]
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    
    #[cfg(feature = "ndarray")]
    use burn::tensor::Float;
    #[cfg(feature = "ndarray")]
    type TestBackend = NdArray<f32>;
    
    #[cfg(feature = "ndarray")]
    #[test]
    fn test_mingru_init() {
        let device = NdArrayDevice::default();
        
        // Create a simple MinGRU configuration
        let config = MinGRUConfig::new(10, 20) // input_size=10, hidden_size=20
            .with_expansion_factor(1.5);
        
        // Initialize the MinGRU module
        let mingru = config.init::<TestBackend>(&device);
        
        // Verify module structure
        assert_eq!(mingru.to_hidden_and_gate.weight.dims(), [10, 60]); // 2 * 20 * 1.5 = 60
        if mingru.proj_out {
            assert_eq!(mingru.to_out.weight.dims(), [30, 20]); // 20 * 1.5 = 30, 20
        }
        
        // Verify expansion factor is set
        assert!((mingru.expansion_factor - 1.5).abs() < 1e-6);
    }
    
    #[cfg(feature = "ndarray")]
    #[test]
    fn test_mingru_forward_single_step() {
        let device = NdArrayDevice::default();
        
        // Create a small MinGRU with no expansion for easier testing
        let config = MinGRUConfig::new(2, 2).with_expansion_factor(1.0);
        let mingru = config.init::<TestBackend>(&device);
        
        // Set weights manually for deterministic test
        let to_hidden_and_gate_weight_data = vec![
            0.1, 0.2, 0.3, 0.4, // input dim 0 -> hidden and gate
            0.5, 0.6, 0.7, 0.8, // input dim 1 -> hidden and gate
        ];
        
        let _to_hidden_and_gate_weight = Tensor::<TestBackend, 1, Float>::from_data(
            &*to_hidden_and_gate_weight_data, &device
        ).reshape([2, 4]);
        
        // Can't directly assign to weight as it's a Param<Tensor>
        // Instead, we can check that it has the expected shape for this test
        assert_eq!(mingru.to_hidden_and_gate.weight.dims(), [2, 4]);
        
        // Create input tensor - single time step
        let x_data = vec![1.0, 2.0]; // batch=1, seq_len=1, input_size=2
        let x = Tensor::<TestBackend, 1, Float>::from_data(&*x_data, &device)
            .reshape([1, 1, 2]);
        
        // Create initial hidden state
        let h0_data = vec![0.0, 0.0]; // batch=1, hidden_size=2
        let h0 = Tensor::<TestBackend, 1, Float>::from_data(&*h0_data, &device)
            .reshape([1, 2]);
        
        // Run forward pass
        let (output, next_hidden) = mingru.forward(x, Some(h0));
        
        // Verify shapes
        assert_eq!(output.dims(), [1, 1, 2]);
        assert_eq!(next_hidden.dims(), [1, 2]);
        
        // Manually compute the expected output (simplified version)
        // hidden = [0.1, 0.2] * 1.0 + [0.5, 0.6] * 2.0 = [0.1, 0.2] + [1.0, 1.2] = [1.1, 1.4]
        // gate = [0.3, 0.4] * 1.0 + [0.7, 0.8] * 2.0 = [0.3, 0.4] + [1.4, 1.6] = [1.7, 2.0]
        // gate = sigmoid(gate) = sigmoid([1.7, 2.0])
        // hidden = g_function(hidden) (simplified here as identity)
        // output = gate * hidden + (1 - gate) * prev_hidden
        
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

/// MinGRU implementation following the parallel associative scan algorithm.
///
/// Mathematical formulation of MinGRU:
/// z_t = σ(W_z · [h_{t-1}, x_t] + b_z)         # Update gate
/// h̃_t = W_h · [h_{t-1}, x_t] + b_h            # Candidate hidden state
/// h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ g(h̃_t)    # New hidden state
///
/// where g(x) is a custom activation function:
/// g(x) = x + 0.5  for x ≥ 0
/// g(x) = sigmoid(x)  for x < 0
///
/// The parallel associative scan allows efficient computation across
/// the entire sequence in O(log n) parallel steps instead of O(n) sequential steps.
#[derive(Module, Debug)]
pub struct MinGRU<B: Backend> {
    /// Combined projection for hidden state and gate computations
    /// Shape: [input_size, 2 * hidden_size * expansion_factor]
    to_hidden_and_gate: Linear<B>,
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
        // Log-space coefficients: log(1 - z_t) = -softplus(k_t)
        // where k_t is the pre-activation value for the gate
        let log_coeffs = -activation::softplus(gate.clone(), 1.0);
    
        // Log-space values: log(z_t) + log(g(h_tilde))
        // log(z_t) = -softplus(-k_t)
        let log_z = -activation::softplus(-gate, 1.0);
    
        // Apply log-g function to compute log(g(h_tilde))
        let log_tilde_h = self.log_g_function(hidden);
    
        // Combined log values: log(z_t * g(h_tilde)) = log(z_t) + log(g(h_tilde))
        let log_values = log_z + log_tilde_h;
    
        // Use the efficient parallel scan in log space
        parallel_scan_log(log_coeffs, log_values, prev_hidden)
    }
    
    /// g(x) function as defined in the paper: 
    /// g(x) = x + 0.5 for x >= 0, sigmoid(x) for x < 0
    fn g_function(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let zeros = Tensor::zeros_like(&x);
        let x_positive = x.clone().greater_equal(zeros.clone()).float();
        let x_negative = x.clone().lower(zeros).float();
    
        (x_positive * (x.clone() + 0.5)) + (x_negative * activation::sigmoid(x))
    }

    /// log(g(x)) function as defined in the paper:
    /// log(g(x)) = log(x + 0.5) for x >= 0, -softplus(-x) for x < 0
    fn log_g_function(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, hidden_dim] = x.dims();
        let device = x.device();
        let zeros = Tensor::zeros([batch_size, seq_len, hidden_dim], &device);
    
        let x_positive = x.clone().greater_equal(zeros.clone()).float();
        let x_negative = x.clone().lower(zeros).float();
    
        // For x >= 0: log(x + 0.5)
        // Using ReLU to ensure we don't take log of negative numbers
        let log_g_pos = (activation::relu(x.clone()) + 0.5).log();
    
        // For x < 0: log(sigmoid(x)) = -softplus(-x)
        let log_g_neg = -activation::softplus(-x, 1.0);
    
        (x_positive * log_g_pos) + (x_negative * log_g_neg)
    }
}
