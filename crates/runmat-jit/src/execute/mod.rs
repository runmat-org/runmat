//! Invocation lifecycle and semantic host for generic Native IR.

mod aggregate;
mod awaiting;
mod call;
mod callbacks;
mod deoptimization;
mod executor;
mod indexing;
mod invocation;
mod mutation;
mod operand;
mod operator;
mod site;
mod state;
mod sync;
mod workspace;

pub use executor::{GenericExecution, GenericExecutor};
pub use invocation::{GenericInvocation, GenericInvocationStep};
