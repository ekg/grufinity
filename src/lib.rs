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
            
            pub type BackendDevice = WgpuDevice;
            pub type RawBackend = Wgpu<f32, i32>;
            pub type BackendWithAutodiff = Autodiff<RawBackend>;
            
            // For reporting
            const BACKEND_NAME: &str = "wgpu-fusion";
            println!("Using WGPU backend with fusion optimization");
            
            let device = WgpuDevice::default();
        }
        
        #[cfg(all(feature = "wgpu", feature = "autodiff", not(feature = "fusion")))]
        {
            use burn::backend::wgpu::{Wgpu, WgpuDevice};
            use burn::backend::autodiff::Autodiff;
            
            pub type BackendDevice = WgpuDevice;
            pub type RawBackend = Wgpu<f32, i32>;
            pub type BackendWithAutodiff = Autodiff<RawBackend>;
            
            // For reporting
            const BACKEND_NAME: &str = "wgpu";
            println!("Using WGPU backend");
            
            let device = WgpuDevice::default();
        }
        
        #[cfg(all(feature = "candle", feature = "fusion", feature = "autodiff"))]
        {
            use burn::backend::candle::{Candle, CandleDevice};
            use burn::backend::autodiff::Autodiff;
            
            pub type BackendDevice = CandleDevice;
            pub type RawBackend = Candle<f32>;
            pub type BackendWithAutodiff = Autodiff<RawBackend>;
            
            // For reporting
            const BACKEND_NAME: &str = "candle-fusion";
            println!("Using Candle CPU backend with fusion optimization");
            
            let device = CandleDevice::Cpu;
        }
        
        #[cfg(all(feature = "candle", feature = "autodiff", not(feature = "fusion")))]
        {
            use burn::backend::candle::{Candle, CandleDevice};
            use burn::backend::autodiff::Autodiff;
            
            pub type BackendDevice = CandleDevice;
            pub type RawBackend = Candle<f32>;
            pub type BackendWithAutodiff = Autodiff<RawBackend>;
            
            // For reporting
            const BACKEND_NAME: &str = "candle";
            println!("Using Candle CPU backend");
            
            let device = CandleDevice::Cpu;
        }
        
        #[cfg(all(feature = "tch", feature = "autodiff"))]
        {
            use burn::backend::libtorch::{LibTorch, LibTorchDevice};
            use burn::backend::autodiff::Autodiff;
            
            pub type BackendDevice = LibTorchDevice;
            pub type RawBackend = LibTorch<f32>;
            pub type BackendWithAutodiff = Autodiff<RawBackend>;
            
            // For reporting
            const BACKEND_NAME: &str = "libtorch";
            println!("Using LibTorch backend");
            
            let device = LibTorchDevice::Cpu;
        }
        
        #[cfg(all(feature = "ndarray", feature = "autodiff"))]
        {
            use burn::backend::ndarray::{NdArray, NdArrayDevice};
            use burn::backend::autodiff::Autodiff;
            
            pub type BackendDevice = NdArrayDevice;
            pub type RawBackend = NdArray<f32>;
            pub type BackendWithAutodiff = Autodiff<RawBackend>;
            
            // For reporting
            const BACKEND_NAME: &str = "ndarray";
            println!("Using NdArray backend");
            
            let device = NdArrayDevice;
        }
        
        // Default fallback to WGPU if no specific combination is enabled but wgpu is available
        #[cfg(all(feature = "wgpu", not(feature = "autodiff")))]
        {
            use burn::backend::wgpu::{Wgpu, WgpuDevice};
            
            pub type BackendDevice = WgpuDevice;
            pub type RawBackend = Wgpu<f32, i32>;
            // Create a dummy type for BackendWithAutodiff since autodiff isn't available
            pub type BackendWithAutodiff = Wgpu<f32, i32>;
            
            // For reporting
            const BACKEND_NAME: &str = "wgpu-basic";
            println!("Using basic WGPU backend (autodiff not available)");
            
            let device = WgpuDevice::default();
        }
        
        // We need to ensure one backend is always selected
        #[cfg(not(any(
            feature = "wgpu", 
            feature = "candle",
            feature = "tch",
            feature = "ndarray"
        )))]
        compile_error!("At least one backend feature must be enabled: 'wgpu', 'candle', 'tch', or 'ndarray'");
        
        // Ensure autodiff is available for training
        #[cfg(not(feature = "autodiff"))]
        {
            println!("WARNING: 'autodiff' feature is not enabled. Training will not work properly.");
            println!("Consider adding the autodiff feature: cargo add burn --features autodiff");
        }
    };
}
