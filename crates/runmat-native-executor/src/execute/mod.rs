//! Invocation lifecycle and semantic host for generic Native IR.

mod aggregate;
mod awaiting;
mod call;
mod callbacks;
mod candidate;
mod deoptimization;
mod executor;
mod indexing;
mod invocation;
mod mutation;
mod operand;
mod operator;
mod osr;
mod region;
mod site;
mod state;
mod sync;
mod workspace;

pub use workspace::{NativeWorkspaceBinding, NativeWorkspaceInput, NativeWorkspaceSnapshot};

pub use executor::{
    NativeExecution, NativeExecutor, NativeExecutorOptions, NativeInvocationRequest,
};
pub use invocation::{NativeInvocation, NativeInvocationStep};
