//! New builtin set. Builtins are organised by category and re-exported from this module.
#[macro_use]
pub mod common;
pub mod acceleration;
pub mod array;
pub mod cells;
pub mod constants;
pub mod containers;
pub mod diagnostics;
pub mod image;
pub mod introspection;
pub mod io;
pub mod logical;
pub mod math;
pub mod stats;
pub mod strings;
pub mod structs;
pub mod timing;

// Temporary: expose legacy modules while migration is in progress.
pub mod legacy {
    pub use crate::arrays;
    #[cfg(feature = "blas-lapack")]
    pub use crate::blas;
    pub use crate::comparison;
    pub use crate::concatenation;
    pub use crate::elementwise;
    pub use crate::indexing;
    #[cfg(feature = "blas-lapack")]
    pub use crate::lapack;
    pub use crate::matrix;
}
