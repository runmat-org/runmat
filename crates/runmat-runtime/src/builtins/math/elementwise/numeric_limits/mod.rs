//! MATLAB-compatible numeric-limit query builtins.

mod errors;
mod execute;
pub(crate) mod flintmax;
mod floating;
mod integer;
pub(crate) mod intmax;
pub(crate) mod intmin;
mod operation;
pub(crate) mod realmax;
pub(crate) mod realmin;
mod syntax;

#[cfg(test)]
mod tests;
