pub mod parallel_scan;
pub mod mingru;
pub mod model;
pub mod dataset;
pub mod tbptt;

pub use parallel_scan::*;
pub use mingru::*;
pub use model::*;
pub use dataset::*;
pub use tbptt::*;

// Re-export essential types for convenience
pub use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
    config::Config,
    record::{Record, Recorder, BinFileRecorder, FullPrecisionSettings}
};

// Import backend types at the crate level
#[cfg(feature = "cuda")]
pub use burn::backend::cuda::{Cuda, CudaDevice};

#[cfg(feature = "wgpu")]
pub use burn::backend::wgpu::{Wgpu, WgpuDevice};

#[cfg(feature = "candle")]
pub use burn::backend::candle::{Candle, CandleDevice};

#[cfg(feature = "tch")]
pub use burn::backend::libtorch::{LibTorch, LibTorchDevice};

#[cfg(feature = "ndarray")]
pub use burn::backend::ndarray::{NdArray, NdArrayDevice};

#[cfg(feature = "autodiff")]
pub use burn::backend::autodiff::Autodiff;

// Define the backend types conditionally - only one will be active at a time
// based on the feature flags and their priority

// CUDA backend (highest priority)
#[cfg(all(feature = "cuda", feature = "autodiff"))]
pub type BackendWithAutodiff = Autodiff<Cuda>;
#[cfg(all(feature = "cuda", not(feature = "autodiff")))]
pub type BackendWithAutodiff = Cuda;
#[cfg(feature = "cuda")]
pub type RawBackend = Cuda;
#[cfg(feature = "cuda")]
pub type BackendDevice = CudaDevice;

// WGPU backend (second priority, only used if cuda is not enabled)
#[cfg(all(feature = "wgpu", feature = "autodiff", not(feature = "cuda")))]
pub type BackendWithAutodiff = Autodiff<Wgpu<f32, i32>>;
#[cfg(all(feature = "wgpu", not(feature = "autodiff"), not(feature = "cuda")))]
pub type BackendWithAutodiff = Wgpu<f32, i32>;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type RawBackend = Wgpu<f32, i32>;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type BackendDevice = WgpuDevice;

// Candle backend (third priority, only used if cuda and wgpu are not enabled)
#[cfg(all(feature = "candle", feature = "autodiff", not(any(feature = "cuda", feature = "wgpu"))))]
pub type BackendWithAutodiff = Autodiff<Candle<f32>>;
#[cfg(all(feature = "candle", not(feature = "autodiff"), not(any(feature = "cuda", feature = "wgpu"))))]
pub type BackendWithAutodiff = Candle<f32>;
#[cfg(all(feature = "candle", not(any(feature = "cuda", feature = "wgpu"))))]
pub type RawBackend = Candle<f32>;
#[cfg(all(feature = "candle", not(any(feature = "cuda", feature = "wgpu"))))]
pub type BackendDevice = CandleDevice;

// LibTorch backend (fourth priority)
#[cfg(all(feature = "tch", feature = "autodiff", not(any(feature = "cuda", feature = "wgpu", feature = "candle"))))]
pub type BackendWithAutodiff = Autodiff<LibTorch<f32>>;
#[cfg(all(feature = "tch", not(feature = "autodiff"), not(any(feature = "cuda", feature = "wgpu", feature = "candle"))))]
pub type BackendWithAutodiff = LibTorch<f32>;
#[cfg(all(feature = "tch", not(any(feature = "cuda", feature = "wgpu", feature = "candle"))))]
pub type RawBackend = LibTorch<f32>;
#[cfg(all(feature = "tch", not(any(feature = "cuda", feature = "wgpu", feature = "candle"))))]
pub type BackendDevice = LibTorchDevice;

// NdArray backend (lowest priority)
#[cfg(all(feature = "ndarray", feature = "autodiff", not(any(feature = "cuda", feature = "wgpu", feature = "candle", feature = "tch"))))]
pub type BackendWithAutodiff = Autodiff<NdArray<f32>>;
#[cfg(all(feature = "ndarray", not(feature = "autodiff"), not(any(feature = "cuda", feature = "wgpu", feature = "candle", feature = "tch"))))]
pub type BackendWithAutodiff = NdArray<f32>;
#[cfg(all(feature = "ndarray", not(any(feature = "cuda", feature = "wgpu", feature = "candle", feature = "tch"))))]
pub type RawBackend = NdArray<f32>;
#[cfg(all(feature = "ndarray", not(any(feature = "cuda", feature = "wgpu", feature = "candle", feature = "tch"))))]
pub type BackendDevice = NdArrayDevice;

/// Run with the appropriate backend based on configured features
#[macro_export]
macro_rules! use_configured_backend {
    () => {
        // Determine which backend to use based on features
        #[cfg(all(feature = "cuda", feature = "fusion", feature = "autodiff"))]
        {
            // For reporting
            const BACKEND_NAME: &str = "cuda-fusion";
            println!("Using CUDA backend with fusion optimization");
        }
        
        #[cfg(all(feature = "cuda", feature = "autodiff", not(feature = "fusion")))]
        {
            // For reporting
            const BACKEND_NAME: &str = "cuda";
            println!("Using CUDA backend");
        }
        
        #[cfg(all(feature = "wgpu", feature = "fusion", feature = "autodiff", not(feature = "cuda")))]
        {
            // For reporting
            const BACKEND_NAME: &str = "wgpu-fusion";
            println!("Using WGPU backend with fusion optimization");
        }
        
        #[cfg(all(feature = "wgpu", feature = "autodiff", not(feature = "fusion"), not(feature = "cuda")))]
        {
            // For reporting
            const BACKEND_NAME: &str = "wgpu";
            println!("Using WGPU backend");
        }
        
        #[cfg(all(feature = "candle", feature = "fusion", feature = "autodiff", not(any(feature = "cuda", feature = "wgpu"))))]
        {
            // For reporting
            const BACKEND_NAME: &str = "candle-fusion";
            println!("Using Candle CPU backend with fusion optimization");
        }
        
        #[cfg(all(feature = "candle", feature = "autodiff", not(feature = "fusion"), not(any(feature = "cuda", feature = "wgpu"))))]
        {
            // For reporting
            const BACKEND_NAME: &str = "candle";
            println!("Using Candle CPU backend");
        }
        
        #[cfg(all(feature = "tch", feature = "autodiff", not(any(feature = "cuda", feature = "wgpu", feature = "candle"))))]
        {
            // For reporting
            const BACKEND_NAME: &str = "libtorch";
            println!("Using LibTorch backend");
        }
        
        #[cfg(all(feature = "ndarray", feature = "autodiff", not(any(feature = "cuda", feature = "wgpu", feature = "candle", feature = "tch"))))]
        {
            // For reporting
            const BACKEND_NAME: &str = "ndarray";
            println!("Using NdArray backend");
        }
        
        // Default fallback to WGPU if no specific combination is enabled but wgpu is available
        #[cfg(all(feature = "wgpu", not(feature = "autodiff"), not(feature = "cuda")))]
        {
            // For reporting
            const BACKEND_NAME: &str = "wgpu-basic";
            println!("Using basic WGPU backend (autodiff not available)");
        }
        
        // We need to ensure one backend is always selected
        #[cfg(not(any(
            feature = "cuda",
            feature = "wgpu", 
            feature = "candle",
            feature = "tch",
            feature = "ndarray"
        )))]
        compile_error!("At least one backend feature must be enabled: 'cuda', 'wgpu', 'candle', 'tch', or 'ndarray'");
        
        // Ensure autodiff is available for training
        #[cfg(not(feature = "autodiff"))]
        {
            println!("WARNING: 'autodiff' feature is not enabled. Training will not work properly.");
            println!("Consider adding the autodiff feature: cargo add burn --features autodiff");
        }
    };
}
