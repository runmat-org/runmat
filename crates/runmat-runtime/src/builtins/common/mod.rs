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

pub use spec::builtin_doc_texts;
pub use spec::DocTextInventory;

#[cfg(test)]
pub mod test_support;

pub(crate) fn map_control_flow_with_builtin(
    flow: crate::RuntimeControlFlow,
    builtin: &'static str,
) -> crate::RuntimeControlFlow {
    match flow {
        crate::RuntimeControlFlow::Suspend(pending) => crate::RuntimeControlFlow::Suspend(pending),
        crate::RuntimeControlFlow::Error(mut err) => {
            if err.context.builtin.is_none() {
                err.context = err.context.with_builtin(builtin);
            }
            crate::RuntimeControlFlow::Error(err)
        }
    }
}
