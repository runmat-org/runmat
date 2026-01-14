use std::fmt;

use thiserror::Error;

/// Narrow set of interaction kinds that can suspend execution.
///
/// Note: this is transitional scaffolding for Phase 1. It will evolve into a richer typed
/// awaitable/pending-request model once `ExecuteFuture` lands.
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
            RuntimeControlFlow::Suspend(pending) => write!(f, "suspend: {}", pending.prompt),
            RuntimeControlFlow::Error(message) => write!(f, "{message}"),
        }
    }
}

impl From<RuntimeControlFlow> for String {
    fn from(value: RuntimeControlFlow) -> Self {
        match value {
            RuntimeControlFlow::Error(e) => e,
            RuntimeControlFlow::Suspend(pending) => {
                let kind = match pending.kind {
                    InteractionKind::Line { .. } => "line",
                    InteractionKind::KeyPress => "keypress",
                    InteractionKind::GpuMapRead => "internal",
                };
                format!("__RUNMAT_SUSPEND__:{kind}:{}", pending.prompt)
            }
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


