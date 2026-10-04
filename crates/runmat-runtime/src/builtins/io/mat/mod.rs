//! MATLAB MAT-file I/O builtins.

pub(crate) mod format;
pub(crate) mod authority;
mod cooperation;
pub mod load;
pub mod matfile;
pub mod save;
#[cfg(test)]
pub(crate) mod tests;
