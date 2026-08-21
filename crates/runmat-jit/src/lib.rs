//! Native execution policy and lifecycle for verified RunMat Native IR.

#[cfg(not(target_arch = "wasm32"))]
pub mod compile;
#[cfg(not(target_arch = "wasm32"))]
pub mod entry;
mod error;
pub mod invalidation;
pub mod tiering;

pub use error::*;

#[cfg(not(target_arch = "wasm32"))]
pub use compile::GenericCompiler;
