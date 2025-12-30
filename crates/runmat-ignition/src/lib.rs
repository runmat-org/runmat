#[cfg(feature = "native-accel")]
pub mod accel_graph;
pub mod bytecode;
pub mod compiler;
pub mod functions;
pub mod gc_roots;
pub mod instr;
pub mod vm;

pub use bytecode::{compile, compile_with_functions};
pub use functions::{Bytecode, ExecutionContext, UserFunction};
pub use instr::Instr;
pub use vm::{
    interpret, interpret_with_vars, push_pending_workspace, resume_with_state,
    take_updated_workspace_state, InterpreterOutcome, InterpreterState, PendingExecution,
    PendingWorkspaceGuard,
};

use runmat_builtins::Value;
use runmat_hir::HirProgram;

pub fn execute(program: &HirProgram) -> Result<Vec<Value>, String> {
    let bc = compile(program)?;
    interpret(&bc)
}
