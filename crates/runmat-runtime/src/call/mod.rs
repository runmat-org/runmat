//! Executor-neutral callable resolution, argument, output, and invocation semantics.

pub mod arguments;
pub(crate) mod admission;
pub(crate) mod catalog;
pub mod closures;
pub mod descriptor;
pub(crate) mod feval;
pub mod function_abi;
mod function_arguments;
pub mod identity;
pub mod lexical;
pub use function_arguments::{CapturedValueLease, RootedFunctionArguments};
