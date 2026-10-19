//! Dependency-free, versioned C ABI for native RunMat extensions.
//!
//! This crate contains ABI vocabulary only. It does not expose Rust runtime
//! values, VM bytecode, generated-code internals, or adapter policy.

#![deny(unsafe_op_in_unsafe_fn)]

mod call;
mod capability;
mod context;
mod error;
mod handle;
mod header;
mod layout;
mod result;
mod symbol;
mod value;
mod version;
mod vtable;

pub use call::*;
pub use capability::*;
pub use context::*;
pub use error::*;
pub use handle::*;
pub use header::*;
pub use result::*;
pub use symbol::*;
pub use value::*;
pub use version::*;
pub use vtable::*;
