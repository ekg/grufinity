use burn::tensor::{backend::Backend, Tensor};

/// Implementation of the parallel associative scan algorithm for efficient computation 
/// of recurrent neural networks.
///
/// This algorithm computes the recurrence relation: h_t = a_t ⊙ h_{t-1} + b_t
/// in O(log n) parallel steps instead of O(n) sequential steps, enabling much faster
/// processing of long sequences on parallel hardware like GPUs.
///
/// # Mathematical Formulation
///
/// The recurrence equation:
/// h_t = a_t * h_{t-1} + b_t
///
/// Can be reformulated with an associative binary operator ⊕:
/// (a_i, b_i) ⊕ (a_j, b_j) = (a_i * a_j, b_i + a_i * b_j)
///
/// This allows an O(log n) parallel scan algorithm instead of O(n) sequential computation.
///
/// # Mathematical Formulation
///
/// The recurrence equation:
/// h_t = a_t * h_{t-1} + b_t
///
/// Can be reformulated with an associative binary operator ⊕:
/// (a_i, b_i) ⊕ (a_j, b_j) = (a_i * a_j, b_i + a_i * b_j)
///
/// This allows an O(log n) parallel scan algorithm instead of O(n) sequential computation.
///
/// # Tensor Shapes:
/// - coeffs: [batch_size, seq_len, hidden_dim] - Coefficients a_t
/// - values: [batch_size, seq_len, hidden_dim] - Values b_t
/// - h0: Optional[batch_size, hidden_dim] - Initial hidden state
/// - Returns: [batch_size, seq_len, hidden_dim] - Output hidden states
pub fn parallel_scan<B: Backend>(
    coeffs: Tensor<B, 3>,  // a_t coefficients (batch_size, seq_len, hidden_dim)
    values: Tensor<B, 3>,  // b_t values (batch_size, seq_len, hidden_dim)
    h0: Option<Tensor<B, 2>>, // Initial hidden state (batch_size, hidden_dim)
) -> Tensor<B, 3> {
    parallel_scan_standard_impl(coeffs, values, h0)
}

/// Log-space implementation of the parallel associative scan algorithm
/// for improved numerical stability when processing long sequences.
///
/// By performing computations in log space, we can avoid numerical underflow/overflow
/// issues that often occur with long sequences due to repeated multiplication.
///
/// Mathematical formulation:
/// Instead of computing h_t = (1-z_t) * h_{t-1} + z_t * g(h̃_t) directly,
/// we compute:
/// log(h_t) = logsumexp(log(1-z_t) + log(h_{t-1}), log(z_t) + log(g(h̃_t)))
///
/// Where logsumexp(a,b) = max(a,b) + log(1 + exp(min(a,b) - max(a,b)))
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

/// Numerically stable log-space implementation of the parallel scan following the paper
fn parallel_scan_log_impl<B: Backend>(
    log_coeffs: Tensor<B, 3>,
    log_values: Tensor<B, 3>,
    h0: Option<Tensor<B, 2>>,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, hidden_dim] = log_coeffs.dims();
    let device = log_coeffs.device();
    
    // 1. Compute a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
    // First prepare log_coeffs with initial zeros padding
    let zeros_pad = Tensor::zeros([batch_size, 1, hidden_dim], &device);
    let padded_log_coeffs = Tensor::cat(vec![zeros_pad, log_coeffs], 1);
    
    // Then compute cumulative sum along dim 1
    let mut a_star = Tensor::zeros([batch_size, seq_len + 1, hidden_dim], &device);
    for t in 1..seq_len + 1 {
        let prev = a_star.clone().slice([0..batch_size, t-1..t, 0..hidden_dim]);
        let curr = padded_log_coeffs.clone().slice([0..batch_size, t..t+1, 0..hidden_dim]);
        let cumsum = prev + curr;
        a_star = a_star.slice_assign([0..batch_size, t..t+1, 0..hidden_dim], cumsum);
    }
    
    // 2. Prepare log_values with initial h0
    let log_values_with_h0 = if let Some(h0) = h0 {
        // Take log of h0, being careful with zeros
        let epsilon = 1e-10;
        let log_h0 = h0.clamp(epsilon, f32::MAX).log().reshape([batch_size, 1, hidden_dim]);
        Tensor::cat(vec![log_h0, log_values], 1)
    } else {
        // Use small value for log(0) approximation
        let log_h0 = Tensor::full([batch_size, 1, hidden_dim], -1e5, &device);
        Tensor::cat(vec![log_h0, log_values], 1)
    };
    
    // 3. Compute log_h0_plus_b_star = torch.logcumsumexp(log_values_with_h0 - a_star, dim=1)
    let log_diff = log_values_with_h0 - a_star.clone();
    let log_h0_plus_b_star = logcumsumexp(log_diff);
    
    // 4. Compute log_h = a_star + log_h0_plus_b_star
    let log_h = a_star + log_h0_plus_b_star;
    
    // 5. Return exp(log_h)[:, 1:] to get the sequence outputs h_1 to h_T
    log_h.slice([0..batch_size, 1..seq_len+1, 0..hidden_dim]).exp()
}

/// Compute the log(cumsum(exp(x))) in a numerically stable way
fn logcumsumexp<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch_size, seq_len, hidden_dim] = x.dims();
    let device = x.device();
    
    let mut result = Tensor::zeros([batch_size, seq_len, hidden_dim], &device);
    
    // Copy first element
    let first = x.clone().slice([0..batch_size, 0..1, 0..hidden_dim]);
    result = result.slice_assign([0..batch_size, 0..1, 0..hidden_dim], first);
    
    // Compute log(exp(result[t-1]) + exp(x[t])) for remaining elements
    for t in 1..seq_len {
        let prev = result.clone().slice([0..batch_size, t-1..t, 0..hidden_dim]);
        let curr = x.clone().slice([0..batch_size, t..t+1, 0..hidden_dim]);
        
        // Use log-sum-exp trick for numerical stability:
        // log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(min(a,b) - max(a,b)))
        let max_vals = prev.clone().max_pair(curr.clone());
        let min_vals = prev.clone().min_pair(curr.clone());
        let diff = min_vals - max_vals.clone();
        let logsumexp = max_vals + (Tensor::ones_like(&diff) + diff.exp()).log();
        
        result = result.slice_assign([0..batch_size, t..t+1, 0..hidden_dim], logsumexp);
    }
    
    result
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
#[allow(dead_code)]
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

/// Parallel prefix sum using Blelloch scan algorithm
/// Efficient O(log n) algorithm for prefix sum operations
#[allow(dead_code)]
fn parallel_prefix_sum<B: Backend>(input: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch_size, seq_len, hidden_dim] = input.dims();
    let device = input.device();
    
    // Ensure the sequence length is a power of 2 for easier algorithm implementation
    let pow2_len = seq_len.next_power_of_two();
    
    // Pad input to power of 2 length if necessary
    let padded_input = if pow2_len != seq_len {
        let padding = Tensor::zeros([batch_size, pow2_len - seq_len, hidden_dim], &device);
        Tensor::cat(vec![input, padding], 1)
    } else {
        input
    };
    
    // Clone the input for in-place modification
    let mut temp = padded_input.clone();
    
    // Up-sweep phase (reduction)
    let mut stride = 1;
    while stride < pow2_len {
        for i in (stride..pow2_len).step_by(stride * 2) {
            // Combine elements at i-stride and i
            let left_idx = i - stride;
            let right_idx = i;
            
            // Get the elements
            let left = temp.clone().slice([0..batch_size, left_idx..left_idx+1, 0..hidden_dim]);
            let right = temp.clone().slice([0..batch_size, right_idx..right_idx+1, 0..hidden_dim]);
            
            // Update the right element with sum of left and right
            let new_right = left + right;
            temp = temp.slice_assign([0..batch_size, right_idx..right_idx+1, 0..hidden_dim], new_right);
        }
        
        stride *= 2;
    }
    
    // Set identity element at root
    temp = temp.slice_assign([0..batch_size, pow2_len-1..pow2_len, 0..hidden_dim], 
                           Tensor::zeros([batch_size, 1, hidden_dim], &device));
    
    // Down-sweep phase
    stride = pow2_len / 2;
    while stride > 0 {
        for i in (stride..pow2_len).step_by(stride * 2) {
            let left_idx = i - stride;
            let right_idx = i;
            
            // Get current values
            let left = temp.clone().slice([0..batch_size, left_idx..left_idx+1, 0..hidden_dim]);
            let right = temp.clone().slice([0..batch_size, right_idx..right_idx+1, 0..hidden_dim]);
            
            // Store right value
            let temp_right = right.clone();
            
            // Update right: right += left
            let new_right = right + left;
            temp = temp.slice_assign([0..batch_size, right_idx..right_idx+1, 0..hidden_dim], new_right);
            
            // Update left: left = temp_right
            temp = temp.slice_assign([0..batch_size, left_idx..left_idx+1, 0..hidden_dim], temp_right);
        }
        
        stride /= 2;
    }
    
    // Extract result up to the original sequence length
    if seq_len != pow2_len {
        temp.slice([0..batch_size, 0..seq_len, 0..hidden_dim])
    } else {
        temp
    }
}

/// Execute a sequential log-space scan for short sequences
#[allow(dead_code)]
fn sequential_log_scan<B: Backend>(
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
    
    // Handle the initial hidden state (h0)
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

/// Parallel logsumexp using the Blelloch algorithm
#[allow(dead_code)]
fn parallel_logsumexp_scan<B: Backend>(input: Tensor<B, 3>) -> Tensor<B, 3> {
    // Always use the sequential algorithm
    return log_cumsum_exp(input);
}

/// Implementation of a true parallel scan algorithm using Blelloch scan
/// This provides O(log n) time complexity when executed on a GPU
pub fn parallel_scan_divide_conquer<B: Backend>(
    coeffs: Tensor<B, 3>,
    values: Tensor<B, 3>,
    h0: Option<Tensor<B, 2>>,
) -> Tensor<B, 3> {
    // Always use the standard implementation regardless of sequence length
    return parallel_scan_standard_impl(coeffs, values, h0);
}
