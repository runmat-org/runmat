//! Invocation lifecycle and semantic host for generic Native IR.

mod aggregate;
mod awaiting;
mod call;
mod call_suspension;
mod callbacks;
mod candidate;
mod control;
mod deoptimization;
mod executor;
mod global_target;
mod indexing;
mod invocation;
mod mutation;
mod operation_completion;
mod operand;
mod operator;
mod osr;
mod region;
mod short_circuit;
mod site;
mod state;
mod subscript_path;
mod workspace;

pub use workspace::{
    NativeWorkspaceBinding, NativeWorkspaceGlobalBinding, NativeWorkspaceInput,
    NativeWorkspaceSnapshot,
};

pub use executor::{
    NativeExecution, NativeExecutor, NativeExecutorOptions, NativeInvocationRequest,
    RootedNativeExecution, RootedNativeInvocationRequest,
};
pub use invocation::{NativeInvocation, NativeInvocationStep};
