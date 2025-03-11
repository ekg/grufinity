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
        #[cfg(feature = "wgpu-fusion")]
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
        
        #[cfg(all(feature = "wgpu", not(feature = "wgpu-fusion")))]
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
        
        #[cfg(feature = "candle-cuda-fusion")]
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
        
        #[cfg(all(feature = "candle-cuda", not(feature = "candle-cuda-fusion")))]
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
        
        #[cfg(all(feature = "candle", not(any(feature = "candle-cuda", feature = "candle-cuda-fusion"))))]
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
        
        // We need to ensure one backend is always selected
        #[cfg(not(any(
            feature = "wgpu", 
            feature = "wgpu-fusion", 
            feature = "candle", 
            feature = "candle-cuda", 
            feature = "candle-cuda-fusion"
        )))]
        compile_error!("At least one backend feature must be enabled: 'wgpu', 'wgpu-fusion', 'candle', 'candle-cuda', or 'candle-cuda-fusion'");
    };
}
