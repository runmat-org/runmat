use crate::interpreter::errors::mex;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;

#[inline]
pub fn pop_value(stack: &mut Vec<Value>) -> Result<Value, RuntimeError> {
    stack
        .pop()
        .ok_or_else(|| mex("StackUnderflow", "stack underflow"))
}

#[inline]
pub fn pop2(stack: &mut Vec<Value>) -> Result<(Value, Value), RuntimeError> {
    let b = pop_value(stack)?;
    let a = pop_value(stack)?;
    Ok((a, b))
}

#[inline]
pub fn pop_args(stack: &mut Vec<Value>, argc: usize) -> Result<Vec<Value>, RuntimeError> {
    let mut args = Vec::with_capacity(argc);
    for _ in 0..argc {
        args.push(pop_value(stack)?);
    }
    args.reverse();
    Ok(args)
}
