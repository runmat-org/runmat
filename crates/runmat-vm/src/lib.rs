#![allow(clippy::result_large_err)]

pub mod accel;
pub(crate) mod bytecode;
pub(crate) mod call;
pub(crate) mod compiler;
pub(crate) mod instr {
    pub use crate::bytecode::instr::Instr;
}
pub(crate) mod interpreter;
pub(crate) mod layout;
pub(crate) mod object;
pub(crate) mod ops;
mod program_execution;
pub(crate) mod runtime;

pub use bytecode::{compile, compile_semantic_function_registry};
pub use bytecode::{
    AsyncMetadata, AwaitSite, Bytecode, BytecodeRegion, BytecodeRegionBoundary, EmitLabel,
    FunctionBytecode, FunctionRegistry, Instr, SpawnSite, StackEffect, BYTECODE_SCHEMA_VERSION,
    FUNCTION_REGISTRY_SCHEMA_VERSION,
};
#[cfg(feature = "native-accel")]
pub use bytecode::{
    FusionCandidateGroup, FusionInstructionKind, FusionInstructionWindow, FusionMetadata,
};
pub use call::builtins::{push_dynamic_eval_options, set_dynamic_eval_options};
pub use compiler::CompileError;
pub use interpreter::runner::{
    interpret, interpret_function, interpret_function_with_counts, interpret_with_vars,
    interpret_with_vars_in_context, invoke_semantic_function_value,
    invoke_semantic_function_value_in_context,
};
pub use interpreter::runner::{interpret_resume_in_context, prepare_native_execution_metadata};
pub use interpreter::state::{InterpreterOutcome, InterpreterResumeState, InterpreterState};
pub use layout::{
    derive_layout, remap_layout_function_ids, LayoutError, VmAssemblyLayout, VmEntrypointLayout,
    VmFunctionLayout, VmSlotId, VM_LAYOUT_SCHEMA_VERSION,
};
pub use program_execution::{execute_program_request, materialize_deferred_call};
pub use runtime::workspace::{
    push_pending_workspace, take_updated_workspace_assigned_report, take_updated_workspace_state,
    PendingWorkspaceGuard, WorkspaceAssignedReport,
};

#[doc(hidden)]
pub fn reset_thread_state_for_tests() {
    runmat_runtime::debug_context::reset_for_tests();
    runmat_runtime::builtins::introspection::debugging::reset_lock_registry_for_tests();
    runtime::globals::reset_thread_state_for_tests();
    runtime::workspace::reset_thread_state_for_tests();
}
