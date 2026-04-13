pub use runmat_ignition::vm::{
    interpret, interpret_with_vars, push_pending_workspace, set_call_stack_limit,
    set_error_namespace, take_updated_workspace_state, InterpreterOutcome, InterpreterState,
    PendingWorkspaceGuard, DEFAULT_CALLSTACK_LIMIT, DEFAULT_ERROR_NAMESPACE,
};
