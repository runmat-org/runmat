//! Matrix arithmetic identities and their shared runtime services.

pub(crate) mod mldivide;
pub(crate) mod mpower;
pub(crate) mod mrdivide;
pub(crate) mod mtimes;
mod solve;

pub use mldivide::mldivide_host_real_for_provider;
pub use mrdivide::mrdivide_host_real_for_provider;
