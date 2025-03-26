# GRUfinity Code Improvement TODO List

## Documentation Improvements
- [ ] Add shape documentation for tensors in function signatures
- [ ] Add mathematical formulas for key algorithms like MinGRU and parallel scan
- [ ] Include usage examples for core functions
- [ ] Improve inline comments for complex logic

## Code Structure and Modularity
- [ ] Break up large functions in train.rs and tbptt.rs
- [ ] Extract reusable components to separate modules
- [ ] Refactor repetitive code patterns
- [ ] Create helper functions for common operations

## Error Handling
- [ ] Implement proper error types using thiserror
- [ ] Replace unwrap() and expect() with Result handling
- [ ] Add context to error messages
- [ ] Implement recovery strategies for non-fatal errors

## CLI Argument Handling
- [x] Replace manual CLI parsing with clap
- [x] Add proper command structure (train, generate, etc.)
- [x] Improve help messages and examples
- [x] Add validation for command line arguments

## Feature Flag Management
- [ ] Simplify feature flag conditionals
- [ ] Create more hierarchical approach to feature detection
- [ ] Add documentation for feature combinations
- [ ] Ensure feature flags have sensible defaults

## Testing and Benchmarking
- [ ] Add unit tests for core modules
- [ ] Implement integration tests for end-to-end workflows
- [ ] Add benchmarks for performance-critical code
- [ ] Create test fixtures and helper functions

## Data Handling Improvements
- [ ] Enhance validation in batchers
- [ ] Improve error handling in dataset loading
- [ ] Add more flexible data preprocessing options
- [ ] Implement better padding and masking strategies

## Model Architecture Improvements
- [ ] Follow Burn's pattern for model implementation
- [ ] Add type bounds and constraints consistently
- [ ] Improve parameter initialization
- [ ] Add more model hyperparameter options

## Training Metrics and Monitoring
- [ ] Implement structured metrics collection
- [ ] Add custom metrics like perplexity
- [ ] Improve progress reporting
- [ ] Add export of training metrics to visualization formats

## Backend Selection and Configuration
- [ ] Create a more structured backend selection mechanism
- [ ] Add runtime detection of available backends
- [ ] Implement fallback logic for unavailable backends
- [ ] Add consistent device handling across backends

## Memory and Performance Optimizations
- [ ] Improve chunking for long sequences
- [ ] Add explicit memory management
- [ ] Optimize critical paths in forward/backward passes
- [ ] Implement better batching strategies
