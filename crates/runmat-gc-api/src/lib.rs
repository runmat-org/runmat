#![forbid(unsafe_op_in_unsafe_fn)]

mod handle;
mod trace;

pub use handle::GcHandle;
pub use trace::{Trace, Tracer};
