//! One selected parenthesis read/call, under the existing instruction owner.
use runmat_runtime::{context::RuntimeContext, RuntimeError};

pub(super) async fn read(
    num_indices: usize,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    function_registry: &crate::bytecode::FunctionRegistry,
    runtime: &RuntimeContext,
) -> Result<bool, RuntimeError> {
    let mut raw_indices = Vec::with_capacity(num_indices);
    for _ in 0..num_indices {
        raw_indices.push(stack.pop().ok_or(crate::interpreter::errors::mex(
            "StackUnderflow",
            "stack underflow",
        ))?);
    }
    raw_indices.reverse();
    let base = stack.pop().ok_or(crate::interpreter::errors::mex(
        "StackUnderflow",
        "stack underflow",
    ))?;
    super::super::paren_index_value(runtime, base, raw_indices, 1, function_registry)
        .await?
        .with_owned(|value| stack.try_push(value))?;
    Ok(true)
}
