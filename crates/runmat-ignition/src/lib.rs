#![allow(clippy::result_large_err)]

#[cfg(feature = "native-accel")]
pub mod accel_graph;
pub mod bytecode;
pub mod compiler;
pub mod functions;
pub mod gc_roots;
pub mod instr;

pub use bytecode::compile;
pub use runmat_vm::CompileError;
pub use functions::{Bytecode, ExecutionContext, UserFunction};
pub use instr::Instr;
pub use runmat_vm::{interpret, interpret_function, interpret_function_with_counts, interpret_with_vars};
pub use runmat_vm::interpreter::api::{
    push_pending_workspace, set_call_stack_limit, set_error_namespace,
    take_updated_workspace_state, DEFAULT_CALLSTACK_LIMIT, DEFAULT_ERROR_NAMESPACE,
};
pub use runmat_vm::interpreter::state::{InterpreterOutcome, InterpreterState};
pub use runmat_vm::runtime::workspace::PendingWorkspaceGuard;

use runmat_builtins::Value;
use runmat_hir::HirProgram;
use runmat_runtime::RuntimeError;
use std::collections::HashMap;

pub async fn execute(program: &HirProgram) -> Result<Vec<Value>, RuntimeError> {
    let bc = compile(program, &HashMap::new()).map_err(RuntimeError::from)?;
    interpret(&bc).await
}
