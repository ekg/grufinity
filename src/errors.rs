use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// Custom error types for GRUfinity
#[derive(Error, Debug)]
pub enum GRUfinityError {
    /// IO errors that occur during file operations
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// JSON serialization/deserialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Model loading errors
    #[error("Failed to load model from {path}: {reason}")]
    ModelLoad {
        path: PathBuf,
        reason: String,
    },

    /// Configuration loading errors
    #[error("Failed to load configuration from {path}: {reason}")]
    ConfigLoad {
        path: PathBuf,
        reason: String,
    },

    /// Vocabulary loading errors
    #[error("Failed to load vocabulary from {path}: {reason}")]
    VocabLoad {
        path: PathBuf,
        reason: String,
    },

    /// Tensor dimension errors
    #[error("Tensor dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        expected: String,
        actual: String,
    },

    /// Device initialization errors
    #[error("Failed to initialize device: {0}")]
    DeviceInit(String),
    
    /// Parameter validation errors
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Generic errors with custom message
    #[error("{0}")]
    Generic(String),
}

/// Type alias for GRUfinity's Result type
pub type Result<T> = std::result::Result<T, GRUfinityError>;

/// Utility function to convert from any error implementing std::error::Error to GRUfinityError
pub fn into_grufinity_error<E: std::error::Error>(error: E) -> GRUfinityError {
    GRUfinityError::Generic(error.to_string())
}

/// Extension trait for Result to add context to errors
pub trait ResultExt<T, E> {
    /// Add context to an error
    fn with_context<C, F>(self, context: F) -> std::result::Result<T, GRUfinityError>
    where
        F: FnOnce() -> C,
        C: std::fmt::Display;
}

impl<T, E: std::error::Error + 'static> ResultExt<T, E> for std::result::Result<T, E> {
    fn with_context<C, F>(self, context: F) -> std::result::Result<T, GRUfinityError>
    where
        F: FnOnce() -> C,
        C: std::fmt::Display,
    {
        self.map_err(|e| {
            GRUfinityError::Generic(format!("{}: {}", context(), e))
        })
    }
}
