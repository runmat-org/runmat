//! Invocation lifecycle and semantic host for generic Native IR.

mod aggregate;
mod call;
mod callbacks;
mod executor;
mod indexing;
mod mutation;
mod operand;
mod operator;
mod site;
mod state;
mod sync;
mod workspace;

pub use executor::{GenericExecution, GenericExecutor};
