#![allow(clippy::result_large_err)]

pub mod accel;
pub mod bytecode;
pub mod call;
pub mod compiler;
pub mod indexing;
pub mod interpreter;
pub mod object;
pub mod ops;
pub mod runtime;

pub use runmat_ignition::bytecode::compile;
pub use runmat_ignition::functions::{Bytecode, ExecutionContext, UserFunction};
pub use runmat_ignition::instr::{ArgSpec, EmitLabel, EndExpr, Instr, StackEffect};
pub use runmat_ignition::vm::{
    interpret, interpret_with_vars, push_pending_workspace, set_call_stack_limit,
    set_error_namespace, take_updated_workspace_state, InterpreterOutcome, InterpreterState,
    PendingWorkspaceGuard, DEFAULT_CALLSTACK_LIMIT, DEFAULT_ERROR_NAMESPACE,
};
pub use runmat_ignition::{execute, CompileError};
