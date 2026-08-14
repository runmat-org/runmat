//! Verified, deterministic generic Native IR for RunMat.

pub mod abi;
#[cfg(not(target_arch = "wasm32"))]
pub mod cranelift;
mod error;
pub mod ir;
pub mod lowering;
mod target;

pub use error::*;
pub use ir::*;
pub use lowering::*;
pub use target::*;

pub const NATIVE_IR_SCHEMA_VERSION: u16 = 2;
