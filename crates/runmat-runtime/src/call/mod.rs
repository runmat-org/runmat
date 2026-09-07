//! Executor-neutral callable resolution, argument, output, and invocation semantics.

pub mod arguments;
pub(crate) mod catalog;
pub mod closures;
pub mod descriptor;
pub mod function_abi;
pub mod identity;
pub mod lexical;
