//! MATLAB MAT-file I/O builtins.

mod admission;
pub(crate) mod authority;
mod cooperation;
mod encoding;
pub(crate) mod format;
pub mod load;
pub mod matfile;
pub(crate) mod records;
pub mod save;
#[cfg(test)]
pub(crate) mod tests;
