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

// Define element type based on the f16 feature flag
#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

// Re-export essential types for convenience
pub use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
    config::Config,
    record::{Record, Recorder, BinFileRecorder, FullPrecisionSettings}
};

// Export specialized softplus function
pub use parallel_scan::specialized_softplus;

// Import f16 type when feature is enabled
#[cfg(feature = "f16")]
pub use burn::tensor::f16;

// Import backend types at the crate level
// Only import CUDA types when explicitly requested
#[cfg(feature = "cuda")]
pub use burn::backend::Cuda;
#[cfg(feature = "cuda")]
pub use burn::backend::cuda::CudaDevice;

#[cfg(feature = "vulkan")]
pub use burn::backend::wgpu::{Vulkan, WgpuDevice as VulkanDevice};

#[cfg(feature = "wgpu")]
pub use burn::backend::wgpu::{Wgpu, WgpuDevice};

#[cfg(any(feature = "candle", feature = "candle-cuda", feature = "candle-metal"))]
pub use burn::backend::candle::{Candle, CandleDevice};

#[cfg(feature = "tch")]
pub use burn::backend::libtorch::{LibTorch, LibTorchDevice};

// For checking CUDA availability with LibTorch
// Convenience function to create a LibTorch device with the right type
#[cfg(feature = "tch")]
pub fn create_libtorch_device(device_id: usize) -> LibTorchDevice {
    #[cfg(all(feature = "tch-gpu", not(target_os = "macos")))]
    {
        println!("LibTorch: Using CUDA device {}", device_id);
        return LibTorchDevice::Cuda(device_id);
    }
    
    #[cfg(all(feature = "tch-gpu", target_os = "macos"))]
    {
        println!("LibTorch: Using MPS device on Apple Silicon");
        return LibTorchDevice::Mps;
    }
    
    #[cfg(not(feature = "tch-gpu"))]
    {
        println!("LibTorch: Using CPU device");
        LibTorchDevice::Cpu
    }
}

#[cfg(feature = "ndarray")]
pub use burn::backend::ndarray::{NdArray, NdArrayDevice};

#[cfg(feature = "autodiff")]
pub use burn::backend::autodiff::Autodiff;

// Define backend types based on feature flags (in priority order)

// 1. CUDA (highest priority)
#[cfg(all(feature = "cuda", feature = "autodiff"))]
pub type BackendWithAutodiff = Autodiff<Cuda<Elem>>;
#[cfg(all(feature = "cuda", not(feature = "autodiff")))]
pub type BackendWithAutodiff = Cuda<Elem>;
#[cfg(feature = "cuda")]
pub type RawBackend = Cuda<Elem>;
#[cfg(feature = "cuda")]
pub type BackendDevice = CudaDevice;

// 2. Candle CUDA backend (second priority)
#[cfg(all(feature = "candle-cuda", feature = "autodiff", not(feature = "cuda")))]
pub type BackendWithAutodiff = Autodiff<Candle<Elem>>;
#[cfg(all(feature = "candle-cuda", not(feature = "autodiff"), not(feature = "cuda")))]
pub type BackendWithAutodiff = Candle<Elem>;
#[cfg(all(feature = "candle-cuda", not(feature = "cuda")))]
pub type RawBackend = Candle<Elem>;
#[cfg(all(feature = "candle-cuda", not(feature = "cuda")))]
pub type BackendDevice = CandleDevice;

// 3. Vulkan backend (third priority)
#[cfg(all(feature = "vulkan", feature = "autodiff", 
          not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"),
          not(feature = "candle")))]
pub type BackendWithAutodiff = Autodiff<Vulkan<Elem, i32>>;
#[cfg(all(feature = "vulkan", not(feature = "autodiff"), 
          not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"),
          not(feature = "candle")))]
pub type BackendWithAutodiff = Vulkan<Elem, i32>;
#[cfg(all(feature = "vulkan", 
          not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"),
          not(feature = "candle")))]
pub type RawBackend = Vulkan<Elem, i32>;
#[cfg(all(feature = "vulkan", 
          not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"),
          not(feature = "candle")))]
pub type BackendDevice = burn::backend::wgpu::WgpuDevice; // Vulkan uses WgpuDevice as its device type

// 4. WGPU backend (fourth priority)
#[cfg(all(feature = "wgpu", feature = "autodiff", 
          not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"),
          not(feature = "candle"), not(feature = "vulkan")))]
pub type BackendWithAutodiff = Autodiff<Wgpu<Elem, i32>>;
#[cfg(all(feature = "wgpu", not(feature = "autodiff"), 
          not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"),
          not(feature = "candle"), not(feature = "vulkan")))]
pub type BackendWithAutodiff = Wgpu<Elem, i32>;
#[cfg(all(feature = "wgpu", 
          not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"),
          not(feature = "candle"), not(feature = "vulkan")))]
pub type RawBackend = Wgpu<Elem, i32>;
#[cfg(all(feature = "wgpu", 
          not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"),
          not(feature = "candle"), not(feature = "vulkan")))]
pub type BackendDevice = WgpuDevice;

// 5. Candle Metal backend (fifth priority)
#[cfg(all(feature = "candle-metal", feature = "autodiff", 
          not(feature = "cuda"), not(feature = "candle-cuda"), 
          not(feature = "wgpu"), not(feature = "vulkan")))]
pub type BackendWithAutodiff = Autodiff<Candle<Elem>>;
#[cfg(all(feature = "candle-metal", not(feature = "autodiff"), 
          not(feature = "cuda"), not(feature = "candle-cuda"), 
          not(feature = "wgpu"), not(feature = "vulkan")))]
pub type BackendWithAutodiff = Candle<Elem>;
#[cfg(all(feature = "candle-metal", 
          not(feature = "cuda"), not(feature = "candle-cuda"), 
          not(feature = "wgpu"), not(feature = "vulkan")))]
pub type RawBackend = Candle<Elem>;
#[cfg(all(feature = "candle-metal", 
          not(feature = "cuda"), not(feature = "candle-cuda"), 
          not(feature = "wgpu"), not(feature = "vulkan")))]
pub type BackendDevice = CandleDevice;

// 6. Candle CPU backend (sixth priority)
#[cfg(all(feature = "candle", feature = "autodiff", 
         not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), 
         not(feature = "wgpu"), not(feature = "vulkan")))]
pub type BackendWithAutodiff = Autodiff<Candle<Elem>>;
#[cfg(all(feature = "candle", not(feature = "autodiff"), 
         not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), 
         not(feature = "wgpu"), not(feature = "vulkan")))]
pub type BackendWithAutodiff = Candle<Elem>;
#[cfg(all(feature = "candle", 
         not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), 
         not(feature = "wgpu"), not(feature = "vulkan")))]
pub type RawBackend = Candle<Elem>;
#[cfg(all(feature = "candle", 
         not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), 
         not(feature = "wgpu"), not(feature = "vulkan")))]
pub type BackendDevice = CandleDevice;

// 7. LibTorch backend (seventh priority)
#[cfg(all(feature = "tch", feature = "autodiff", 
         not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), 
         not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle")))]
pub type BackendWithAutodiff = Autodiff<LibTorch<Elem>>;
#[cfg(all(feature = "tch", not(feature = "autodiff"), 
         not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), 
         not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle")))]
pub type BackendWithAutodiff = LibTorch<Elem>;
#[cfg(all(feature = "tch", 
         not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), 
         not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle")))]
pub type RawBackend = LibTorch<Elem>;
#[cfg(all(feature = "tch", 
         not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), 
         not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle")))]
pub type BackendDevice = LibTorchDevice;

// 8. NdArray backend (lowest priority)
#[cfg(all(feature = "ndarray", feature = "autodiff", 
         not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), 
         not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle"), 
         not(feature = "tch")))]
pub type BackendWithAutodiff = Autodiff<NdArray<Elem>>;
#[cfg(all(feature = "ndarray", not(feature = "autodiff"), 
         not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), 
         not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle"), 
         not(feature = "tch")))]
pub type BackendWithAutodiff = NdArray<Elem>;
#[cfg(all(feature = "ndarray", 
         not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), 
         not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle"), 
         not(feature = "tch")))]
pub type RawBackend = NdArray<Elem>;
#[cfg(all(feature = "ndarray", 
         not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), 
         not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle"), 
         not(feature = "tch")))]
pub type BackendDevice = NdArrayDevice;

/// Returns a string identifying which backend is currently being used
pub fn get_backend_name() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        "CUDA"
    }
    #[cfg(all(feature = "candle-cuda", not(feature = "cuda")))]
    {
        "Candle CUDA"
    }
    #[cfg(all(feature = "vulkan", 
          not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"),
          not(feature = "candle")))]
    {
        "Vulkan"
    }
    #[cfg(all(feature = "wgpu", 
          not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"),
          not(feature = "candle"), not(feature = "vulkan")))]
    {
        "WebGPU"
    }
    #[cfg(all(feature = "candle-metal", 
          not(feature = "cuda"), not(feature = "candle-cuda"), 
          not(feature = "wgpu"), not(feature = "vulkan")))]
    {
        "Candle Metal"
    }
    #[cfg(all(feature = "candle", 
         not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), 
         not(feature = "wgpu"), not(feature = "vulkan")))]
    {
        "Candle CPU"
    }
    #[cfg(all(feature = "tch", 
         not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), 
         not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle")))]
    {
        "LibTorch"
    }
    #[cfg(all(feature = "ndarray", 
         not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), 
         not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle"), 
         not(feature = "tch")))]
    {
        "NdArray"
    }
    
    // Fallback for unknown combinations
    #[cfg(not(any(
        feature = "cuda",
        all(feature = "candle-cuda", not(feature = "cuda")),
        all(feature = "vulkan", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "candle")),
        all(feature = "wgpu", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "candle"), not(feature = "vulkan")),
        all(feature = "candle-metal", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "wgpu"), not(feature = "vulkan")),
        all(feature = "candle", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "wgpu"), not(feature = "vulkan")),
        all(feature = "tch", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle")),
        all(feature = "ndarray", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle"), not(feature = "tch"))
    )))]
    {
        "Unknown Backend"
    }
}

// Test-only backend implementations
// These are only available when running tests and should be used preferentially over
// feature-specific backends when testing
#[cfg(all(test, feature = "ndarray", not(any(
    all(feature = "cuda", feature = "autodiff"),
    all(feature = "candle-cuda", feature = "autodiff", not(feature = "cuda")),
    all(feature = "vulkan", feature = "autodiff", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "candle")),
    all(feature = "wgpu", feature = "autodiff", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "candle"), not(feature = "vulkan")),
    all(feature = "candle-metal", feature = "autodiff", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "wgpu"), not(feature = "vulkan")),
    all(feature = "candle", feature = "autodiff", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "wgpu"), not(feature = "vulkan")),
    all(feature = "tch", feature = "autodiff", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle")),
    all(feature = "ndarray", feature = "autodiff", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle"), not(feature = "tch"))
))))]
pub use burn::backend::ndarray::{NdArray, NdArrayDevice};

#[cfg(all(test, feature = "autodiff", not(any(
    all(feature = "cuda", feature = "autodiff"),
    all(feature = "candle-cuda", feature = "autodiff", not(feature = "cuda")),
    all(feature = "vulkan", feature = "autodiff", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "candle")),
    all(feature = "wgpu", feature = "autodiff", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "candle"), not(feature = "vulkan")),
    all(feature = "candle-metal", feature = "autodiff", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "wgpu"), not(feature = "vulkan")),
    all(feature = "candle", feature = "autodiff", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "wgpu"), not(feature = "vulkan")),
    all(feature = "tch", feature = "autodiff", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle")),
    all(feature = "ndarray", feature = "autodiff", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle"), not(feature = "tch"))
))))]
pub type BackendWithAutodiff = Autodiff<NdArray<f32>>;

#[cfg(all(test, not(any(
    feature = "cuda",
    all(feature = "candle-cuda", not(feature = "cuda")),
    all(feature = "vulkan", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "candle")),
    all(feature = "wgpu", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "candle"), not(feature = "vulkan")),
    all(feature = "candle-metal", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "wgpu"), not(feature = "vulkan")),
    all(feature = "candle", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "wgpu"), not(feature = "vulkan")),
    all(feature = "tch", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle")),
    all(feature = "ndarray", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle"), not(feature = "tch"))
))))]
pub type RawBackend = NdArray<Elem>;

#[cfg(all(test, not(any(
    feature = "cuda",
    all(feature = "candle-cuda", not(feature = "cuda")),
    all(feature = "vulkan", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "candle")),
    all(feature = "wgpu", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "candle"), not(feature = "vulkan")),
    all(feature = "candle-metal", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "wgpu"), not(feature = "vulkan")),
    all(feature = "candle", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "wgpu"), not(feature = "vulkan")),
    all(feature = "tch", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle")),
    all(feature = "ndarray", not(feature = "cuda"), not(feature = "candle-cuda"), not(feature = "candle-metal"), not(feature = "wgpu"), not(feature = "vulkan"), not(feature = "candle"), not(feature = "tch"))
))))]
pub type BackendDevice = NdArrayDevice;

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
            #[cfg(feature = "tch-gpu")]
            {
                #[cfg(not(target_os = "macos"))]
                println!("Using LibTorch backend with CUDA support");
                #[cfg(target_os = "macos")]
                println!("Using LibTorch backend with MPS support");
            }
            #[cfg(not(feature = "tch-gpu"))]
            {
                println!("Using LibTorch backend (CPU only)");
            }
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
