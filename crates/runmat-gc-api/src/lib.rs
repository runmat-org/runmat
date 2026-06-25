#![forbid(unsafe_op_in_unsafe_fn)]

mod handle;
mod root;
mod trace;

pub use handle::GcHandle;
pub use root::{GcRoot, RootId, RootInfo, RootScannerStats};
pub use trace::{Trace, Tracer};
