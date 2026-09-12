//! Shared helpers for builtin implementations.
//!
//! This module hosts small utility subsystems that builtins
//! can depend on.

pub mod arg_tokens;
pub(crate) mod binary;
pub mod broadcast;
pub mod concatenation;
pub(crate) mod control_flow_error;
pub mod deal;
pub mod elementwise;
pub mod env;
pub(crate) mod exact_logical;
pub mod format;
pub mod fs;
pub mod gpu_helpers;
pub mod identifiers;
pub mod indexing;
pub mod integer_capability;
pub(crate) mod integer_conversion;
pub(crate) mod integer_value;
pub mod json;
pub mod linalg;
pub(crate) mod mapped_callable;
pub mod matrix;
pub mod path_search;
pub mod path_state;
pub(crate) mod provider_restore;
pub mod random;
pub mod random_args;
pub mod residency;
pub(crate) mod resident_output;
pub mod shape;
pub mod spec;
pub mod tensor;
pub(crate) mod uniform_scalar_output;
pub mod validation;

#[cfg(test)]
pub mod test_support;

pub(crate) use control_flow_error::map_control_flow_with_builtin;
