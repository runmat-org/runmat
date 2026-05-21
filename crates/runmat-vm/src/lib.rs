#![allow(clippy::result_large_err)]

pub mod accel;
pub(crate) mod bytecode;
pub(crate) mod call;
pub(crate) mod compiler;
pub mod indexing;
pub(crate) mod instr {
    pub use crate::bytecode::instr::{ArgSpec, EndExpr, Instr};
}
pub(crate) mod interpreter;
pub(crate) mod layout;
pub(crate) mod object;
pub(crate) mod ops;
pub(crate) mod runtime;

pub use bytecode::{compile, compile_semantic_function_registry};
pub use bytecode::{
    ArgSpec, Bytecode, EmitLabel, EndExpr, Instr, SemanticAsyncMetadata, SemanticAwaitSite,
    SemanticFunctionBytecode, SemanticFunctionRegistry, SemanticSpawnSite, StackEffect,
};
#[cfg(feature = "native-accel")]
pub use bytecode::{
    SemanticFusionCandidateGroup, SemanticFusionInstructionKind, SemanticFusionInstructionWindow,
    SemanticFusionMetadata,
};
pub use compiler::CompileError;
pub use interpreter::api::{
    set_call_stack_limit, set_error_namespace, DEFAULT_CALLSTACK_LIMIT, DEFAULT_ERROR_NAMESPACE,
};
pub use interpreter::runner::{
    interpret, interpret_function, interpret_function_with_counts, interpret_with_vars,
    invoke_semantic_function_value,
};
pub use interpreter::state::{InterpreterOutcome, InterpreterState};
pub use layout::{
    derive_layout, LayoutError, VmAssemblyLayout, VmEntrypointLayout, VmFunctionLayout, VmSlotId,
};
pub use runtime::workspace::{
    push_pending_workspace, take_updated_workspace_state, PendingWorkspaceGuard,
};

pub async fn call_method_or_member_index_named_with_outputs(
    base: runmat_builtins::Value,
    name: String,
    args: Vec<runmat_builtins::Value>,
    requested_outputs: usize,
    fallback_policy: runmat_hir::CallableFallbackPolicy,
) -> Result<runmat_builtins::Value, runmat_runtime::RuntimeError> {
    call::closures::call_method_or_member_index_named_with_outputs(
        base,
        name,
        args,
        requested_outputs,
        fallback_policy,
    )
    .await
}
