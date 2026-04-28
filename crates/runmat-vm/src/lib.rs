#![allow(clippy::result_large_err)]

pub mod accel;
pub mod bytecode;
pub mod call;
pub mod compiler;
pub mod functions {
    pub use crate::bytecode::program::{Bytecode, CallFrame, ExecutionContext, UserFunction};
}
pub mod indexing;
pub mod instr {
    pub use crate::bytecode::instr::{ArgSpec, EmitLabel, EndExpr, Instr, StackEffect};
}
pub mod interpreter;
pub mod object;
pub mod ops;
pub mod runtime;

pub use bytecode::compile;
pub use bytecode::{
    ArgSpec, Bytecode, CallFrame, EmitLabel, EndExpr, ExecutionContext, Instr, StackEffect,
    UserFunction,
};
pub use compiler::CompileError;
pub use interpreter::api::{
    set_call_stack_limit, set_error_namespace, DEFAULT_CALLSTACK_LIMIT, DEFAULT_ERROR_NAMESPACE,
};
pub use interpreter::runner::{
    interpret, interpret_function, interpret_function_with_counts, interpret_with_vars,
};
pub use interpreter::state::{InterpreterOutcome, InterpreterState};
pub use runtime::workspace::{
    push_pending_workspace, take_updated_workspace_state, PendingWorkspaceGuard,
};

use runmat_builtins::Value;
use runmat_hir::HirProgram;
use runmat_runtime::RuntimeError;
use std::collections::HashMap;

pub async fn execute(program: &HirProgram) -> Result<Vec<Value>, RuntimeError> {
    let bc = compile(program, &HashMap::new()).map_err(RuntimeError::from)?;
    interpret(&bc).await
}
