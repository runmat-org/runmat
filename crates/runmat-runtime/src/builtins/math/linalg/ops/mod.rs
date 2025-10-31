//! Linear algebra operations builtins.

pub(crate) mod ctranspose;
pub(crate) mod mldivide;
pub(crate) mod mrdivide;
pub(crate) mod mtimes;
pub(crate) mod trace;
pub(crate) mod transpose;

pub use mldivide::mldivide_host_real_for_provider;
pub use mrdivide::mrdivide_host_real_for_provider;
