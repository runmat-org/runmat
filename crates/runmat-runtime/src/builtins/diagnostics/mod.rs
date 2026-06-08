//! Diagnostics-oriented builtins (error handling, warnings, assertions, ...).

pub mod assert;
pub mod error;
pub mod rethrow;
pub(crate) mod type_resolvers;
pub mod warning;
