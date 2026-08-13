#![allow(clippy::result_large_err)]

pub mod accel;
pub(crate) mod bytecode;
pub(crate) mod call;
pub(crate) mod compiler;
pub mod coverage;
pub mod indexing;
pub(crate) mod instr {
    pub use crate::bytecode::instr::{ArgSpec, EndExpr, Instr};
}
pub(crate) mod interpreter;
pub(crate) mod layout;
pub(crate) mod object;
pub(crate) mod ops;
mod program_execution;
pub(crate) mod runtime;

pub use bytecode::{compile, compile_semantic_function_registry};
pub use bytecode::{
    ArgSpec, AsyncMetadata, AwaitSite, Bytecode, EmitLabel, EndExpr, FunctionBytecode,
    FunctionRegistry, Instr, SpawnSite, StackEffect,
};
#[cfg(feature = "native-accel")]
pub use bytecode::{
    FusionCandidateGroup, FusionInstructionKind, FusionInstructionWindow, FusionMetadata,
};
pub use call::builtins::{push_dynamic_eval_options, set_dynamic_eval_options};
pub use compiler::CompileError;
pub use interpreter::api::{
    set_call_stack_limit, set_error_namespace, DEFAULT_CALLSTACK_LIMIT, DEFAULT_ERROR_NAMESPACE,
};
pub use interpreter::runner::{
    interpret, interpret_function, interpret_function_with_counts, interpret_with_vars,
    interpret_with_vars_in_context, invoke_semantic_function_value,
    invoke_semantic_function_value_in_context,
};
pub use interpreter::state::{InterpreterOutcome, InterpreterState};
pub use layout::{
    derive_layout, LayoutError, VmAssemblyLayout, VmEntrypointLayout, VmFunctionLayout, VmSlotId,
};
pub use program_execution::{execute_program_request, materialize_deferred_call};
pub use runtime::workspace::{
    push_pending_workspace, take_updated_workspace_assigned_report, take_updated_workspace_state,
    PendingWorkspaceGuard, WorkspaceAssignedReport,
};

#[doc(hidden)]
pub fn expand_cell_indices_for_call(
    cell: &runmat_value::CellArray,
    indices: &[runmat_value::Value],
) -> Result<Vec<runmat_value::Value>, runmat_runtime::RuntimeError> {
    ops::cells::expand_cell_indices(cell, indices)
}

#[doc(hidden)]
pub fn expand_all_cell_for_call(
    cell: &runmat_value::CellArray,
) -> Result<Vec<runmat_value::Value>, runmat_runtime::RuntimeError> {
    ops::cells::expand_all_cell_values(cell)
}

#[doc(hidden)]
pub fn reset_thread_state_for_tests() {
    runtime::call_stack::reset_thread_state_for_tests();
    runmat_runtime::debug_context::reset_for_tests();
    runmat_runtime::builtins::introspection::debugging::reset_lock_registry_for_tests();
    runtime::globals::reset_thread_state_for_tests();
    runtime::workspace::reset_thread_state_for_tests();
}

pub async fn call_method_or_member_index_named_with_outputs(
    base: runmat_value::Value,
    name: String,
    args: Vec<runmat_value::Value>,
    requested_outputs: usize,
    _fallback_policy: runmat_hir::CallableFallbackPolicy,
) -> Result<runmat_value::Value, runmat_runtime::RuntimeError> {
    call::closures::call_method_or_member_index_named_with_outputs(
        base,
        name,
        args,
        requested_outputs,
        None,
    )
    .await
}
