//! Public interpreter API surface.
//!
//! Port target:
//! - top-level interpreter entrypoints and public runtime state types from
//!   `runmat-ignition/src/vm.rs`

pub use crate::runtime::call_stack::{
    set_call_stack_limit, set_error_namespace, DEFAULT_CALLSTACK_LIMIT, DEFAULT_ERROR_NAMESPACE,
};
pub use crate::runtime::workspace::{
    push_pending_workspace, take_updated_workspace_state, PendingWorkspaceGuard,
};
