use burn::tensor::{backend::Backend, Tensor};

/// Implementation of the parallel associative scan algorithm 
/// for efficient computation of recurrence: h_t = a_t ⊙ h_{t-1} + b_t
pub fn parallel_scan<B: Backend>(
    coeffs: Tensor<B, 3>,  // a_t coefficients (batch_size, seq_len, hidden_dim)
    values: Tensor<B, 3>,  // b_t values (batch_size, seq_len, hidden_dim)
    h0: Option<Tensor<B, 2>>, // Initial hidden state (batch_size, hidden_dim)
) -> Tensor<B, 3> {
    let [batch_size, seq_len, hidden_dim] = coeffs.dims();
    
    // Create a tensor to hold the accumulated/prefixed coefficients (a_star)
    // a_star[t] = a_1 * a_2 * ... * a_t
    // Initialize with 1s to start the product
    let device = coeffs.device();
    let mut a_star = Tensor::ones([batch_size, seq_len + 1, hidden_dim], &device);
    
    // Copy coeffs to a_star[1:] with proper broadcasting
    let coeffs_expanded = coeffs.clone();
    a_star = a_star.slice_assign([0..batch_size, 1..(seq_len+1), 0..hidden_dim], coeffs_expanded);
    
    // Compute cumulative product in-place along the sequence dimension
    for t in 1..seq_len {
        let a_prev = a_star.clone().slice([0..batch_size, t..t+1, 0..hidden_dim]);
        let a_curr = a_star.clone().slice([0..batch_size, t+1..t+2, 0..hidden_dim]);
        let a_new = a_prev.mul(a_curr);
        a_star = a_star.slice_assign([0..batch_size, t+1..t+2, 0..hidden_dim], a_new);
    }
    
    // Now handle the initial hidden state (h0)
    let b_star = match h0 {
        Some(h0) => {
            // Prepend h0 to values (handled as b_0)
            let mut all_values = Tensor::zeros([batch_size, seq_len + 1, hidden_dim], &device);
            all_values = all_values.slice_assign([0..batch_size, 0..1, 0..hidden_dim], h0.unsqueeze::<3>(1));
            all_values = all_values.slice_assign([0..batch_size, 1..(seq_len+1), 0..hidden_dim], values);
            all_values
        },
        None => {
            // Initialize b_0 with zeros
            let mut all_values = Tensor::zeros([batch_size, seq_len + 1, hidden_dim], &device);
            all_values = all_values.slice_assign([0..batch_size, 1..(seq_len+1), 0..hidden_dim], values);
            all_values
        }
    };
    
    // Compute h_0 + b_star, which is used for the inclusive prefix sum
    // b_star_prefix_sum[t] = (b_0 / a_0) + (b_1 / a_1) + ... + (b_t / a_t)
    let mut b_star_prefix_sum = Tensor::zeros([batch_size, seq_len + 1, hidden_dim], &device);
    for t in 0..seq_len + 1 {
        let b_t = b_star.clone().slice([0..batch_size, t..t+1, 0..hidden_dim]);
        let a_t = if t == 0 {
            Tensor::ones([batch_size, 1, hidden_dim], &device)
        } else {
            a_star.clone().slice([0..batch_size, 0..t, 0..hidden_dim])
        };
        
        // Compute (b_t / a_t) and add to the prefix sum
        let term = b_t.div(a_t);
        
        if t == 0 {
            b_star_prefix_sum = b_star_prefix_sum.slice_assign(
                [0..batch_size, t..t+1, 0..hidden_dim],
                term
            );
        } else {
            let prev_sum = b_star_prefix_sum.clone().slice([0..batch_size, t-1..t, 0..hidden_dim]);
            let new_sum = prev_sum + term;
            b_star_prefix_sum = b_star_prefix_sum.slice_assign(
                [0..batch_size, t..t+1, 0..hidden_dim],
                new_sum
            );
        }
    }
    
    // Compute the final result: h_t = a_star[t] * b_star_prefix_sum[t]
    let h = a_star.mul(b_star_prefix_sum);
    
    // Return h[1:] to get the sequence outputs h_1 to h_T
    h.slice([0..batch_size, 1..(seq_len+1), 0..hidden_dim])
}

/// Log-space implementation of the parallel associative scan algorithm
/// for improved numerical stability
pub fn parallel_scan_log<B: Backend>(
    log_coeffs: Tensor<B, 3>,  // log(a_t) coefficients (batch_size, seq_len, hidden_dim)
    log_values: Tensor<B, 3>,  // log(b_t) values (batch_size, seq_len, hidden_dim)
    h0: Option<Tensor<B, 2>>,  // Initial hidden state (batch_size, hidden_dim)
) -> Tensor<B, 3> {
    let [batch_size, seq_len, hidden_dim] = log_coeffs.dims();
    let device = log_coeffs.device();
    
    // Compute a_star = cumsum(log_coeffs) - cumulative sum of log coefficients
    let a_star_padded = Tensor::zeros([batch_size, seq_len + 1, hidden_dim], &device);
    let log_coeffs_padded = Tensor::zeros([batch_size, seq_len + 1, hidden_dim], &device)
        .slice_assign([0..batch_size, 1..(seq_len+1), 0..hidden_dim], log_coeffs);
    
    let mut a_star = a_star_padded.clone();
    for t in 1..seq_len + 1 {
        let prev_sum = a_star.clone().slice([0..batch_size, t-1..t, 0..hidden_dim]);
        let curr_coeff = log_coeffs_padded.clone().slice([0..batch_size, t..t+1, 0..hidden_dim]);
        let new_sum = prev_sum + curr_coeff;
        a_star = a_star.slice_assign([0..batch_size, t..t+1, 0..hidden_dim], new_sum);
    }
    
    // Now handle the initial hidden state (h0) for log_h0_plus_b_star
    let log_all_values = match h0 {
        Some(h0) => {
            // Prepend log(h0) to log_values
            let mut all_values = Tensor::zeros([batch_size, seq_len + 1, hidden_dim], &device)
                .slice_assign([0..batch_size, 1..(seq_len+1), 0..hidden_dim], log_values);
            
            // Add log(h0) term - need to handle zeros with care
            let epsilon = 1e-10;
            let log_h0 = h0.clamp(epsilon, f32::MAX).log().unsqueeze::<3>(1);  // Explicitly add dimension at index 1 (seq_len)
            all_values = all_values.slice_assign([0..batch_size, 0..1, 0..hidden_dim], log_h0);
            all_values
        },
        None => {
            // If no h0, set the first position to a very negative number (log of near-zero)
            let mut all_values = Tensor::zeros([batch_size, seq_len + 1, hidden_dim], &device)
                .slice_assign([0..batch_size, 1..(seq_len+1), 0..hidden_dim], log_values);
            
            let neg_inf = Tensor::full([batch_size, 1, hidden_dim], -1e5, &device);
            all_values = all_values.slice_assign([0..batch_size, 0..1, 0..hidden_dim], neg_inf);
            all_values
        }
    };
    
    // Compute log_h0_plus_b_star using log-sum-exp trick
    let log_h0_plus_b_star = log_cumsum_exp(log_all_values - a_star.clone());
    
    // Compute log_h = a_star + log_h0_plus_b_star
    let log_h = a_star + log_h0_plus_b_star;
    
    // Return exp(log_h)[1:] to get the sequence outputs h_1 to h_T
    log_h.slice([0..batch_size, 1..(seq_len+1), 0..hidden_dim]).exp()
}

/// Log-space cumulative sum using log-sum-exp trick
fn log_cumsum_exp<B: Backend>(log_x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch_size, seq_len, hidden_dim] = log_x.dims();
    let device = log_x.device();
    
    let mut result = Tensor::zeros([batch_size, seq_len, hidden_dim], &device);
    
    // Initialize first element
    result = result.slice_assign(
        [0..batch_size, 0..1, 0..hidden_dim],
        log_x.clone().slice([0..batch_size, 0..1, 0..hidden_dim])
    );
    
    // Accumulate remaining elements
    for t in 1..seq_len {
        let prev = result.clone().slice([0..batch_size, t-1..t, 0..hidden_dim]);
        let curr = log_x.clone().slice([0..batch_size, t..t+1, 0..hidden_dim]);
        
        // log_sum_exp(a, b) = max(a, b) + log(exp(a - max(a, b)) + exp(b - max(a, b)))
        let max_val = prev.clone().max_pair(curr.clone());
        let sum_exp = (prev - max_val.clone()).exp() + (curr - max_val.clone()).exp();
        let log_sum = max_val + sum_exp.log();
        
        result = result.slice_assign([0..batch_size, t..t+1, 0..hidden_dim], log_sum);
    }
    
    result
}
