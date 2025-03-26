pub mod parallel_scan;
pub mod mingru;
pub mod model;
pub mod dataset;
pub mod tbptt;
pub mod errors;

pub use parallel_scan::*;
pub use mingru::*;
pub use model::*;
pub use dataset::*;
// Only export specific items from tbptt module that are needed
pub use tbptt::{TBPTTConfig, train_with_tbptt, LRSchedulerType};
pub use errors::{GRUfinityError, Result, ResultExt};

// Re-export essential types for convenience
pub use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
    config::Config,
    record::{Record, Recorder, BinFileRecorder, FullPrecisionSettings}
};

// Import backend types at the crate level
// Only import CUDA types when explicitly requested
#[cfg(feature = "cuda")]
pub use burn::backend::Cuda;
#[cfg(feature = "cuda")]
pub use burn::backend::cuda::CudaDevice;

#[cfg(feature = "wgpu")]
pub use burn::backend::wgpu::{Wgpu, WgpuDevice};

#[cfg(any(feature = "candle", feature = "candle-cuda", feature = "candle-metal"))]
pub use burn::backend::candle::{Candle, CandleDevice};

#[cfg(feature = "tch")]
pub use burn::backend::libtorch::{LibTorch, LibTorchDevice};

#[cfg(feature = "ndarray")]
pub use burn::backend::ndarray::{NdArray, NdArrayDevice};

#[cfg(feature = "autodiff")]
pub use burn::backend::autodiff::Autodiff;

// Define backend types using a macro to reduce repetition
macro_rules! define_backend_types {
    ($backend:ty, $device:ty, $feature:literal, $not_features:expr) => {
        // Define BackendWithAutodiff with autodiff
        #[cfg(all(feature = $feature, feature = "autodiff", $not_features))]
        pub type BackendWithAutodiff = Autodiff<$backend>;
        
        // Define BackendWithAutodiff without autodiff
        #[cfg(all(feature = $feature, not(feature = "autodiff"), $not_features))]
        pub type BackendWithAutodiff = $backend;
        
        // Define RawBackend
        #[cfg(all(feature = $feature, $not_features))]
        pub type RawBackend = $backend;
        
        // Define BackendDevice
        #[cfg(all(feature = $feature, $not_features))]
        pub type BackendDevice = $device;
    };
}

// Import Vulkan backend when the feature is enabled (via wgpu)
#[cfg(feature = "vulkan")]
pub use burn::backend::wgpu::{Vulkan, WgpuDevice as VulkanDevice};

// Define the backend types in priority order
// 1. CUDA (highest priority)
define_backend_types!(
    Cuda<f32>, 
    CudaDevice, 
    "cuda", 
    not(any())
);

// 2. Candle CUDA 
define_backend_types!(
    Candle<f32>, 
    CandleDevice, 
    "candle-cuda", 
    not(feature = "cuda")
);

// 3. Vulkan
define_backend_types!(
    Vulkan<f32, i32>, 
    VulkanDevice, 
    "vulkan", 
    not(any(feature = "cuda", feature = "candle-cuda", feature = "candle-metal", feature = "candle"))
);

// 4. WGPU
define_backend_types!(
    Wgpu<f32, i32>, 
    WgpuDevice, 
    "wgpu", 
    not(any(feature = "cuda", feature = "candle-cuda", feature = "candle-metal", feature = "candle", feature = "vulkan"))
);

// 5. Candle Metal
define_backend_types!(
    Candle<f32>, 
    CandleDevice, 
    "candle-metal", 
    not(any(feature = "cuda", feature = "wgpu", feature = "candle-cuda"))
);

// 6. Candle CPU
define_backend_types!(
    Candle<f32>, 
    CandleDevice, 
    "candle", 
    not(any(feature = "cuda", feature = "wgpu", feature = "candle-metal", feature = "candle-cuda"))
);

// 7. LibTorch
define_backend_types!(
    LibTorch<f32>, 
    LibTorchDevice, 
    "tch", 
    not(any(feature = "cuda", feature = "wgpu", feature = "candle-metal", feature = "candle", feature = "candle-cuda"))
);

// 8. NdArray (lowest priority)
define_backend_types!(
    NdArray<f32>, 
    NdArrayDevice, 
    "ndarray", 
    not(any(feature = "cuda", feature = "wgpu", feature = "candle-metal", feature = "candle", feature = "tch", feature = "candle-cuda"))
);

/// Run with the appropriate backend based on configured features
#[macro_export]
macro_rules! use_configured_backend {
    () => {
        // Determine which backend to use based on features
        #[cfg(all(feature = "cuda", feature = "fusion", feature = "autodiff"))]
        {
            // For reporting
            const BACKEND_NAME: &str = "cuda";
            println!("Using CUDA backend with fusion optimization");
        }
        
        #[cfg(all(feature = "cuda", feature = "autodiff", not(feature = "fusion")))]
        {
            // For reporting
            const BACKEND_NAME: &str = "cuda-basic";
            println!("Using CUDA backend");
        }
        
        #[cfg(all(feature = "candle-cuda", feature = "fusion", feature = "autodiff", not(feature = "cuda")))]
        {
            // For reporting
            const BACKEND_NAME: &str = "candle-cuda";
            println!("Using Candle CUDA backend with fusion optimization");
        }
        
        #[cfg(all(feature = "candle-cuda", feature = "autodiff", not(feature = "fusion"), not(feature = "cuda")))]
        {
            // For reporting
            const BACKEND_NAME: &str = "candle-cuda";
            println!("Using Candle CUDA backend");
        }
        
        #[cfg(all(feature = "candle-metal", feature = "fusion", feature = "autodiff", 
                  not(feature = "cuda"), not(feature = "candle-cuda")))]
        {
            // For reporting
            const BACKEND_NAME: &str = "candle-metal";
            println!("Using Candle Metal backend with fusion optimization");
        }
        
        #[cfg(all(feature = "candle-metal", feature = "autodiff", not(feature = "fusion"), 
                  not(feature = "cuda"), not(feature = "candle-cuda")))]
        {
            // For reporting
            const BACKEND_NAME: &str = "candle-metal";
            println!("Using Candle Metal backend");
        }
        
        #[cfg(all(feature = "vulkan", feature = "fusion", feature = "autodiff", 
                  not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal")))]
        {
            // For reporting
            const BACKEND_NAME: &str = "vulkan-fusion";
            println!("Using Vulkan backend with fusion optimization");
        }
        
        #[cfg(all(feature = "vulkan", feature = "autodiff", not(feature = "fusion"), 
                  not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal")))]
        {
            // For reporting
            const BACKEND_NAME: &str = "vulkan";
            println!("Using Vulkan backend");
        }
        
        #[cfg(all(feature = "wgpu", feature = "fusion", feature = "autodiff", 
                  not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), 
                  not(feature = "vulkan")))]
        {
            // For reporting
            const BACKEND_NAME: &str = "wgpu-fusion";
            println!("Using WGPU backend with fusion optimization");
        }
        
        #[cfg(all(feature = "wgpu", feature = "autodiff", not(feature = "fusion"), 
                  not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"),
                  not(feature = "vulkan")))]
        {
            // For reporting
            const BACKEND_NAME: &str = "wgpu";
            println!("Using WGPU backend");
        }
        
        #[cfg(all(feature = "candle", feature = "fusion", feature = "autodiff", 
                  not(feature = "cuda"), not(feature = "candle-cuda"), 
                  not(feature = "candle-metal"), not(feature = "wgpu")))]
        {
            // For reporting
            const BACKEND_NAME: &str = "candle-fusion";
            println!("Using Candle CPU backend with fusion optimization");
        }
        
        #[cfg(all(feature = "candle", feature = "autodiff", not(feature = "fusion"), 
                  not(feature = "cuda"), not(feature = "candle-cuda"),
                  not(feature = "candle-metal"), not(feature = "wgpu")))]
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
        #[cfg(all(feature = "wgpu", not(feature = "autodiff"), not(any(feature = "cuda", all(feature = "candle", feature = "candle-cuda")))))]
        {
            // For reporting
            const BACKEND_NAME: &str = "wgpu-basic";
            println!("Using basic WGPU backend (autodiff not available)");
        }
        
        // We need to ensure one backend is always selected
        #[cfg(not(any(
            feature = "cuda",
            feature = "wgpu",
            feature = "vulkan", 
            feature = "candle",
            feature = "tch",
            feature = "ndarray"
        )))]
        compile_error!("At least one backend feature must be enabled: 'cuda', 'vulkan', 'wgpu', 'candle', 'tch', or 'ndarray'");
        
        // Ensure autodiff is available for training
        #[cfg(not(feature = "autodiff"))]
        {
            println!("WARNING: 'autodiff' feature is not enabled. Training will not work properly.");
            println!("Consider adding the autodiff feature: cargo add burn --features autodiff");
        }
    };
}
