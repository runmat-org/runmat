use std::fmt;
use thiserror::Error;

pub mod runtime_error;
pub use runtime_error::{runtime_error, ErrorContext, RuntimeError, RuntimeErrorBuilder};

/// Narrow set of interaction kinds used for host I/O hooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionKind {
    Line { echo: bool },
    KeyPress,
    /// Internal suspension used while waiting for a WebGPU map/readback completion.
    /// The prompt/label carries debug info.
    GpuMapRead,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingInteraction {
    pub prompt: String,
    pub kind: InteractionKind,
}

/// Typed channel for runtime control flow. This replaces string sentinels like
/// `__RUNMAT_PENDING_INTERACTION__`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeControlFlow<E = String> {
    Suspend(PendingInteraction),
    Error(E),
}

impl From<String> for RuntimeControlFlow<String> {
    fn from(value: String) -> Self {
        RuntimeControlFlow::Error(value)
    }
}

impl From<&str> for RuntimeControlFlow<String> {
    fn from(value: &str) -> Self {
        RuntimeControlFlow::Error(value.to_string())
    }
}

impl<E: fmt::Display> fmt::Display for RuntimeControlFlow<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeControlFlow::Suspend(pending) => write!(f, "suspend: {}", pending.prompt),
            RuntimeControlFlow::Error(message) => write!(f, "{message}"),
        }
    }
}

/// Marker error type used to bubble an internal suspension through `anyhow::Error` without
/// encoding it as a string.
#[derive(Debug, Error)]
#[error("{prompt}")]
pub struct SuspendMarker {
    pub kind: InteractionKind,
    pub prompt: String,
}

impl SuspendMarker {
    pub fn internal(prompt: impl Into<String>) -> Self {
        Self {
            kind: InteractionKind::GpuMapRead,
            prompt: prompt.into(),
        }
    }
}


