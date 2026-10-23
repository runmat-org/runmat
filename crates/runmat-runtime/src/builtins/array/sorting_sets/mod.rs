//! Sorting and set-related array builtins.

pub mod argsort;
pub(super) mod float_order;
pub(super) mod integer_order;
pub mod intersect;
pub mod ismember;
pub mod ismembertol;
pub mod issorted;
pub mod issortedrows;
pub mod setdiff;
pub mod setxor;
pub mod sort;
pub mod sortrows;
mod support;
mod provider_result;
mod set_policy;
#[cfg(test)]
mod mixed_precision_tests;
#[cfg(test)]
pub(crate) mod type_resolvers;
pub mod union;
pub mod unique;

use support::{
    is_unsupported_set_gpu_integer, output_provider, restore_set_outputs, set_output_provider,
};
