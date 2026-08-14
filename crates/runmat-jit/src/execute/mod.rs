//! Invocation lifecycle and semantic host for generic Native IR.

mod callbacks;
mod executor;
mod operand;
mod site;
mod state;

pub use executor::{GenericExecution, GenericExecutor};
