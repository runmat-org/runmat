use thiserror::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionKind {
    Line { echo: bool },
    KeyPress,
    /// Internal suspension point (e.g. wasm WebGPU readback). The prompt/label carries debug info.
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingInteraction {
    pub prompt: String,
    pub kind: InteractionKind,
}

/// Typed channel for runtime control flow. This replaces string sentinels like
/// `__RUNMAT_PENDING_INTERACTION__`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeControlFlow {
    Suspend(PendingInteraction),
    Error(String),
}

impl From<String> for RuntimeControlFlow {
    fn from(value: String) -> Self {
        RuntimeControlFlow::Error(value)
    }
}

impl From<&str> for RuntimeControlFlow {
    fn from(value: &str) -> Self {
        RuntimeControlFlow::Error(value.to_string())
    }
}

impl fmt::Display for RuntimeControlFlow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeControlFlow::Suspend(pending) => {
                write!(f, "suspend: {}", pending.prompt)
            }
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
            kind: InteractionKind::Internal,
            prompt: prompt.into(),
        }
    }
}


