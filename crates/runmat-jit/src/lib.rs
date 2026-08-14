//! Native execution policy and lifecycle for verified RunMat Native IR.

#[cfg(not(target_arch = "wasm32"))]
pub mod compile;
pub mod deopt;
#[cfg(not(target_arch = "wasm32"))]
pub mod entry;
mod error;
#[cfg(not(target_arch = "wasm32"))]
pub mod execute;
pub mod invalidation;
pub mod memory;
pub mod specialization;

pub use error::*;

#[cfg(not(target_arch = "wasm32"))]
pub use compile::{CompiledExecutable, GenericCompiler};
#[cfg(not(target_arch = "wasm32"))]
pub use execute::{GenericExecution, GenericExecutor};
