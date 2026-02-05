//! Diagnostics-oriented builtins (error handling, warnings, assertions, ...).

pub mod assert;
pub mod error;
pub mod warning;
pub(crate) mod type_resolvers;
