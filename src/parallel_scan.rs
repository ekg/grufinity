use burn::tensor::{backend::Backend, Tensor};

/// Implementation of the parallel associative scan algorithm 
/// for efficient computation of recurrence: h_t = a_t ⊙ h_{t-1} + b_t
pub fn parallel_scan<B: Backend>(
    coeffs: Tensor<B, 3>,  // a_t coefficients (batch_size, seq_len, hidden_dim)
    values: Tensor<B, 3>,  // b_t values (batch_size, seq_len, hidden_dim)
    h0: Option<Tensor<B, 2>>, // Initial hidden state (batch_size, hidden_dim)
) -> Tensor<B, 3> {
    parallel_scan_standard_impl(coeffs, values, h0)
}

/// Log-space implementation of the parallel associative scan algorithm
/// for improved numerical stability
pub fn parallel_scan_log<B: Backend>(
    log_coeffs: Tensor<B, 3>,  // log(a_t) coefficients (batch_size, seq_len, hidden_dim)
    log_values: Tensor<B, 3>,  // log(b_t) values (batch_size, seq_len, hidden_dim)
    h0: Option<Tensor<B, 2>>,  // Initial hidden state (batch_size, hidden_dim)
) -> Tensor<B, 3> {
    parallel_scan_log_impl(log_coeffs, log_values, h0)
}

/// Vectorized implementation of the standard parallel scan
fn parallel_scan_standard_impl<B: Backend>(
    coeffs: Tensor<B, 3>, 
    values: Tensor<B, 3>,
    h0: Option<Tensor<B, 2>>,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, hidden_dim] = coeffs.dims();
    let device = coeffs.device();
    
    // Prepare initial state
    let h0_tensor = match h0 {
        Some(h) => h,
        None => Tensor::zeros([batch_size, hidden_dim], &device),
    };
    
    // Create a_star: [1, a_1, a_1*a_2, a_1*a_2*a_3, ...]
    // First, create [1, a_1, a_2, ..., a_T]
    let ones = Tensor::ones([batch_size, 1, hidden_dim], &device);
    let a_padded = Tensor::cat(vec![ones, coeffs], 1);
    
    // Then compute the inclusive scan (cumulative product)
    let a_star = inclusive_scan_mul(a_padded);
    
    // Create b_star: [h_0, b_1, b_2, ..., b_T]
    let h0_expanded = h0_tensor.reshape([batch_size, 1, hidden_dim]);
    let b_padded = Tensor::cat(vec![h0_expanded, values], 1);
    
    // Create terms for the inclusive scan: [h_0, b_1/a_1, b_2/a_2, ..., b_T/a_T]
    let a_star_without_last = a_star.clone().slice([0..batch_size, 0..seq_len+1, 0..hidden_dim]);
    let terms = b_padded.div(a_star_without_last);
    
    // Compute inclusive scan (cumulative sum) of the terms
    let prefix_sum = inclusive_scan_add(terms);
    
    // Compute h_t = a_star[t] * prefix_sum[t] for t = 1...T
    let h_with_h0 = a_star.mul(prefix_sum);
    
    // Return h_1 to h_T (exclude h_0)
    h_with_h0.slice([0..batch_size, 1..seq_len+1, 0..hidden_dim])
}

/// Numerically stable log-space implementation of the parallel scan
fn parallel_scan_log_impl<B: Backend>(
    log_coeffs: Tensor<B, 3>,
    log_values: Tensor<B, 3>,
    h0: Option<Tensor<B, 2>>,
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
            
            // Correctly reshape h0 to [batch_size, 1, hidden_dim]
            // Apply log after reshaping to maintain the correct dimensions
            let log_h0 = h0.clone().clamp(epsilon, f32::MAX).log().reshape([batch_size, 1, hidden_dim]);
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

/// Compute inclusive scan (cumulative product) along dimension 1
fn inclusive_scan_mul<B: Backend>(input: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch_size, seq_len, hidden_dim] = input.dims();
    let _device = input.device();
    
    let mut result = Tensor::zeros_like(&input);
    
    // Copy first element
    let first = input.clone().slice([0..batch_size, 0..1, 0..hidden_dim]);
    result = result.slice_assign([0..batch_size, 0..1, 0..hidden_dim], first);
    
    // Compute cumulative product
    for t in 1..seq_len {
        let prev = result.clone().slice([0..batch_size, t-1..t, 0..hidden_dim]);
        let curr = input.clone().slice([0..batch_size, t..t+1, 0..hidden_dim]);
        let product = prev.mul(curr);
        result = result.slice_assign([0..batch_size, t..t+1, 0..hidden_dim], product);
    }
    
    result
}

/// Compute inclusive scan (cumulative sum) along dimension 1
fn inclusive_scan_add<B: Backend>(input: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch_size, seq_len, hidden_dim] = input.dims();
    let _device = input.device();
    
    let mut result = Tensor::zeros_like(&input);
    
    // Copy first element
    let first = input.clone().slice([0..batch_size, 0..1, 0..hidden_dim]);
    result = result.slice_assign([0..batch_size, 0..1, 0..hidden_dim], first);
    
    // Compute cumulative sum
    for t in 1..seq_len {
        let prev = result.clone().slice([0..batch_size, t-1..t, 0..hidden_dim]);
        let curr = input.clone().slice([0..batch_size, t..t+1, 0..hidden_dim]);
        let sum = prev.add(curr);
        result = result.slice_assign([0..batch_size, t..t+1, 0..hidden_dim], sum);
    }
    
    result
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
        
        // Use the log_sum_exp helper function
        let log_sum = log_sum_exp(prev, curr);
        
        result = result.slice_assign([0..batch_size, t..t+1, 0..hidden_dim], log_sum);
    }
    
    result
}

/// Compute log(exp(a) + exp(b)) in a numerically stable way
#[allow(dead_code)]
fn log_sum_exp<B: Backend>(a: Tensor<B, 3>, b: Tensor<B, 3>) -> Tensor<B, 3> {
    // Use the identity: log(exp(a) + exp(b)) = max(a, b) + log(exp(a - max(a, b)) + exp(b - max(a, b)))
    let max_val = a.clone().max_pair(b.clone());
    
    // Compute exp(a - max_val) + exp(b - max_val)
    let a_shifted = a.sub(max_val.clone());
    let b_shifted = b.sub(max_val.clone());
    let sum_exp = a_shifted.exp().add(b_shifted.exp());
    
    // Add max_val + log(sum_exp)
    max_val.add(sum_exp.log())
}

/// Higher-performance parallel scan that uses true divide-and-conquer approach
/// This can be implemented once the basic version is working and tested
pub fn parallel_scan_divide_conquer<B: Backend>(
    coeffs: Tensor<B, 3>,
    values: Tensor<B, 3>,
    h0: Option<Tensor<B, 2>>,
) -> Tensor<B, 3> {
    // TODO: Implement a true logarithmic-time parallel scan algorithm
    // using the up-sweep/down-sweep approach
    
    // For now, fall back to the standard implementation
    parallel_scan_standard_impl(coeffs, values, h0)
}
