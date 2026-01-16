use std::fmt;
use std::sync::{Mutex, OnceLock};
use std::task::Waker;

use thiserror::Error;

pub mod runtime_error;
pub use runtime_error::{runtime_error, ErrorContext, RuntimeError, RuntimeErrorBuilder};

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

// ---- Internal waker hooks (Phase 2) -----------------------------------------
//
// These allow runtime-internal subsystems (e.g. wasm WebGPU map_async callbacks) to wake the
// poll-driven `ExecuteFuture` without any host-side polling loops.
//
// This intentionally starts narrow (GpuMapRead only). We'll generalize once ExecuteFuture is the
// only execution path.

static GPU_MAP_READ_WAKER: OnceLock<Mutex<Option<Waker>>> = OnceLock::new();

fn gpu_map_read_waker_slot() -> &'static Mutex<Option<Waker>> {
    GPU_MAP_READ_WAKER.get_or_init(|| Mutex::new(None))
}

/// Register (or refresh) the waker for a task currently suspended on a GPU map/readback.
pub fn register_gpu_map_read_waker(waker: &Waker) {
    let mut slot = gpu_map_read_waker_slot()
        .lock()
        .unwrap_or_else(|_| panic!("GPU_MAP_READ_WAKER poisoned"));
    // Always replace: wakers can change between polls.
    *slot = Some(waker.clone());
}

/// Wake any task currently waiting on a GPU map/readback.
pub fn wake_gpu_map_read() {
    let waker = gpu_map_read_waker_slot()
        .lock()
        .ok()
        .and_then(|mut slot| slot.take());
    if let Some(waker) = waker {
        waker.wake();
    }
}


