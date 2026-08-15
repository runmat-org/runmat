//! Host-only orchestration for RunMat native AOT products.

pub mod archive;
pub mod compile;
mod error;
pub mod link;
mod object;

pub use error::{AotError, AotResult};
pub use object::{emit_native_object, NativeObjectOptions};
