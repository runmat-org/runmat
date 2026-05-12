#![allow(clippy::result_large_err)]

pub mod accel;
pub(crate) mod bytecode;
pub(crate) mod call;
pub(crate) mod compiler;
pub mod indexing;
pub(crate) mod instr {
    pub use crate::bytecode::instr::{ArgSpec, EmitLabel, EndExpr, Instr};
}
pub(crate) mod interpreter;
pub(crate) mod layout;
pub(crate) mod object;
pub(crate) mod ops;
pub(crate) mod runtime;

pub use bytecode::compile;
pub use bytecode::{
    ArgSpec, Bytecode, CallFrame, EmitLabel, EndExpr, ExecutionContext, Instr, LegacyUserFunction,
    SemanticFunctionRegistry, StackEffect,
};
pub use compiler::CompileError;
pub use interpreter::api::{
    set_call_stack_limit, set_error_namespace, DEFAULT_CALLSTACK_LIMIT, DEFAULT_ERROR_NAMESPACE,
};
pub use interpreter::dispatch::{
    legacy_user_dispatch_fallback_count, reset_legacy_user_dispatch_fallback_count,
};
pub use interpreter::runner::{
    execute_legacy_user_function_isolated, interpret, interpret_function,
    interpret_function_with_counts, interpret_with_vars, invoke_semantic_function_value,
};
pub use interpreter::state::{InterpreterOutcome, InterpreterState};
pub use layout::{
    derive_layout, LayoutError, VmAssemblyLayout, VmEntrypointLayout, VmFunctionLayout, VmSlotId,
};
pub use runtime::workspace::{
    push_pending_workspace, take_updated_workspace_state, PendingWorkspaceGuard,
};
