//! Retain original operands until runtime dispatch selects the implementation.
use runmat_runtime::{context::RuntimeContext, RuntimeError};
use runmat_value::Value;
#[cfg(test)]
mod tests;

pub struct PreparedBuiltinCall(Vec<Value>);

pub async fn prepare_builtin_args(
    _name: &str,
    args: &[Value],
    _runtime: &RuntimeContext,
) -> Result<PreparedBuiltinCall, RuntimeError> {
    Ok(PreparedBuiltinCall(args.to_vec()))
}

pub async fn call_prepared_builtin(
    name: &str,
    prepared: PreparedBuiltinCall,
    outputs: usize,
) -> Result<Value, RuntimeError> {
    runmat_runtime::call_builtin_async_with_outputs(name, &prepared.0, outputs).await
}
