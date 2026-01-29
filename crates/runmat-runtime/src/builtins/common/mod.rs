//! Shared helpers for builtin implementations.
//!
//! This module hosts small utility subsystems that builtins
//! can depend on.

pub mod broadcast;
pub mod format;
pub mod fs;
pub mod gpu_helpers;
pub mod linalg;
pub mod path_search;
pub mod path_state;
pub mod random;
pub mod random_args;
pub mod residency;
pub mod shape;
pub mod spec;
pub mod tensor;

#[cfg(test)]
pub mod test_support;

pub(crate) fn map_control_flow_with_builtin(
    mut err: crate::RuntimeError,
    builtin: &str,
) -> crate::RuntimeError {
    if err.context.builtin.is_none() {
        err.context = err.context.with_builtin(builtin);
    }
    err
}
