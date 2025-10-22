//! New builtin set. Builtins are organised by category and re-exported from this module.
#[macro_use]
pub mod common;
pub mod acceleration;
pub mod array;
pub mod introspection;
pub mod io;
pub mod logical;
pub mod math;
pub mod strings;
pub mod structs;

// Temporary: expose legacy modules while migration is in progress.
pub mod legacy {
    pub use crate::arrays;
    #[cfg(feature = "blas-lapack")]
    pub use crate::blas;
    pub use crate::comparison;
    pub use crate::concatenation;
    pub use crate::constants;
    pub use crate::elementwise;
    pub use crate::indexing;
    pub use crate::introspection;
    pub use crate::io;
    #[cfg(feature = "blas-lapack")]
    pub use crate::lapack;
    pub use crate::matrix;
}
