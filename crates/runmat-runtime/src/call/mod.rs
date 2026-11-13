//! Executor-neutral callable resolution, argument, output, and invocation semantics.

pub mod arguments;
pub mod argument_text;
pub(crate) mod admission;
pub(crate) mod catalog;
pub mod closures;
pub(crate) mod closure_binding;
pub mod descriptor;
pub(crate) mod feval;
pub mod function_abi;
mod function_arguments;
pub mod identity;
pub mod lexical;
pub use function_arguments::{
    ArgumentRangeError, BuiltinArguments, BuiltinArgumentsIntoIter, CapturedValueLease,
    FunctionArgumentChunksExact, FunctionArgumentIter, FunctionArgumentView,
    RetainedCallableTarget, RootedArgumentValue, RootedFunctionArguments,
    RootedFunctionArgumentsIntoIter,
};
