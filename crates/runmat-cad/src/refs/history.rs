//! The history record every kernel operation must emit.
//!
//! Defined once, kernel-neutral, in [`runmat_geometry_kernel::history`] (contract rules 1 to 8
//! are documented there); re-exported here under the design layer's path.

pub use runmat_geometry_kernel::history::*;
