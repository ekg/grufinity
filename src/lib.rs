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

/// Run with the appropriate backend based on configured features
#[macro_export]
macro_rules! use_configured_backend {
    () => {
        // Determine which backend to use based on features
        #[cfg(all(feature = "wgpu", feature = "fusion", feature = "autodiff"))]
        {
            use burn::backend::wgpu::{Wgpu, WgpuDevice};
            use burn::backend::autodiff::Autodiff;
            
            type BackendDevice = WgpuDevice;
            type RawBackend = Wgpu<f32, i32>;
            type BackendWithAutodiff = Autodiff<RawBackend>;
            
            // For reporting
            const BACKEND_NAME: &str = "wgpu-fusion";
            println!("Using WGPU backend with fusion optimization");
            
            let device = WgpuDevice::default();
        }
        
        #[cfg(all(feature = "wgpu", feature = "autodiff", not(feature = "fusion")))]
        {
            use burn::backend::wgpu::{Wgpu, WgpuDevice};
            use burn::backend::autodiff::Autodiff;
            
            type BackendDevice = WgpuDevice;
            type RawBackend = Wgpu<f32, i32>;
            type BackendWithAutodiff = Autodiff<RawBackend>;
            
            // For reporting
            const BACKEND_NAME: &str = "wgpu";
            println!("Using WGPU backend");
            
            let device = WgpuDevice::default();
        }
        
        #[cfg(all(feature = "candle", feature = "candle-cuda", feature = "fusion", feature = "autodiff"))]
        {
            use burn::backend::candle::{Candle, CandleDevice};
            use burn::backend::autodiff::Autodiff;
            
            type BackendDevice = CandleDevice;
            type RawBackend = Candle<f32>;
            type BackendWithAutodiff = Autodiff<RawBackend>;
            
            // For reporting
            const BACKEND_NAME: &str = "candle-cuda-fusion";
            println!("Using Candle+CUDA backend with fusion optimization");
            
            let device = match CandleDevice::cuda_if_available(0) {
                Ok(device) => {
                    println!("CUDA device found and will be used");
                    device
                },
                Err(_) => {
                    println!("WARNING: CUDA device requested but not available, falling back to CPU");
                    CandleDevice::Cpu
                }
            };
        }
        
        #[cfg(all(feature = "candle", feature = "candle-cuda", feature = "autodiff", not(feature = "fusion")))]
        {
            use burn::backend::candle::{Candle, CandleDevice};
            use burn::backend::autodiff::Autodiff;
            
            type BackendDevice = CandleDevice;
            type RawBackend = Candle<f32>;
            type BackendWithAutodiff = Autodiff<RawBackend>;
            
            // For reporting
            const BACKEND_NAME: &str = "candle-cuda";
            println!("Using Candle+CUDA backend");
            
            let device = match CandleDevice::cuda_if_available(0) {
                Ok(device) => {
                    println!("CUDA device found and will be used");
                    device
                },
                Err(_) => {
                    println!("WARNING: CUDA device requested but not available, falling back to CPU");
                    CandleDevice::Cpu
                }
            };
        }
        
        #[cfg(all(feature = "candle", feature = "autodiff", not(feature = "candle-cuda"), not(feature = "fusion")))]
        {
            use burn::backend::candle::{Candle, CandleDevice};
            use burn::backend::autodiff::Autodiff;
            
            type BackendDevice = CandleDevice;
            type RawBackend = Candle<f32>;
            type BackendWithAutodiff = Autodiff<RawBackend>;
            
            // For reporting
            const BACKEND_NAME: &str = "candle";
            println!("Using Candle CPU backend");
            
            let device = CandleDevice::Cpu;
        }
        
        // Default fallback to WGPU if no specific combination is enabled but wgpu is available
        #[cfg(all(feature = "wgpu", not(feature = "autodiff")))]
        {
            use burn::backend::wgpu::{Wgpu, WgpuDevice};
            
            type BackendDevice = WgpuDevice;
            type RawBackend = Wgpu<f32, i32>;
            // Create a dummy type for BackendWithAutodiff since autodiff isn't available
            type BackendWithAutodiff = Wgpu<f32, i32>;
            
            // For reporting
            const BACKEND_NAME: &str = "wgpu-basic";
            println!("Using basic WGPU backend (autodiff not available)");
            
            let device = WgpuDevice::default();
        }
        
        // We need to ensure one backend is always selected
        #[cfg(not(any(
            feature = "wgpu", 
            feature = "candle"
        )))]
        compile_error!("At least one backend feature must be enabled: 'wgpu' or 'candle'");
        
        // Ensure autodiff is available for training
        #[cfg(not(feature = "autodiff"))]
        {
            println!("WARNING: 'autodiff' feature is not enabled. Training will not work properly.");
            println!("Consider adding the autodiff feature: cargo add burn --features autodiff");
        }
    };
}
