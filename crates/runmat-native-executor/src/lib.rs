//! Native IR invocation, process-owned entrypoints, and semantic host execution.

pub mod deopt;
mod error;
#[cfg(not(target_arch = "wasm32"))]
pub mod executable;
#[cfg(not(target_arch = "wasm32"))]
pub mod execute;
pub mod memory;
#[cfg(not(target_arch = "wasm32"))]
pub mod osr;
mod profile;
#[cfg(not(target_arch = "wasm32"))]
mod region;
pub mod specialization;

pub use error::{NativeExecutorError, NativeExecutorResult};
#[cfg(not(target_arch = "wasm32"))]
pub use executable::NativeExecutable;
#[cfg(not(target_arch = "wasm32"))]
pub use execute::{
    NativeExecution, NativeExecutor, NativeExecutorOptions, NativeInvocationRequest,
};
pub use profile::RepresentationProfile;
