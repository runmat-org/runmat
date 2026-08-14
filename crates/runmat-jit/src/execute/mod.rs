//! Invocation lifecycle and semantic host for generic Native IR.

mod aggregate;
mod call;
mod callbacks;
mod executor;
mod operand;
mod operator;
mod site;
mod state;

pub use executor::{GenericExecution, GenericExecutor};
